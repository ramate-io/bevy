//! Retained uploads for the CPU-authored, shader-read-only binned inputs.

use core::ops::Deref;
use std::sync::OnceLock;

use bytemuck::NoUninit;
use wgpu::BufferUsages;

use crate::{
    render_resource::{BufferId, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

/// A tracked buffer for the three CPU-authored, shader-read-only binned inputs.
///
/// Enabled only by `native_binned_dirty_uploads`. This fork prototype deliberately
/// changes the corresponding public batch-set field types. It supports read-only
/// [`RawBufferVec`] access through [`Deref`], but no mutable dereference: every CPU
/// mutation must invalidate the retained submission before lending mutable access.
/// Typed mutable `RawBufferVec` callers and struct literals require adaptation.
///
/// `BEVY_NATIVE_BINNED_UPLOADS=dirty` skips unchanged full uploads. The process-wide
/// mode is frozen on first construction and defaults to `dirty` when this feature
/// is explicitly enabled. `force` and unknown values perform the original
/// unconditional writes. Neither mode depends on tracing features.
/// External queue or GPU writes to the exposed buffer invalidate this contract;
/// this wrapper is only for the audited read-only binned inputs, not generic storage.
pub struct BinnedInputBuffer<T: NoUninit> {
    raw: RawBufferVec<T>,
    state: UploadState<BufferId>,
    mode: UploadMode,
    usage: BufferUsages,
    device: Option<RenderDevice>,
    #[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
    last_write: BinnedInputUploadOutcome,
    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    submitted: SubmittedBytes,
}

impl<T: NoUninit> BinnedInputBuffer<T> {
    /// Creates a buffer using the process-wide force/dirty upload mode.
    pub fn new(usage: BufferUsages) -> Self {
        let mode = UploadMode::runtime();
        bevy_log::info_once!("native_binned_dirty_uploads mode={}", mode.name());
        Self::with_mode(usage, mode)
    }

    fn with_mode(usage: BufferUsages, mode: UploadMode) -> Self {
        Self {
            raw: RawBufferVec::new(usage),
            state: UploadState::default(),
            mode,
            usage,
            device: None,
            #[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
            last_write: BinnedInputUploadOutcome::default(),
            #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
            submitted: SubmittedBytes::default(),
        }
    }

    /// Appends a value, invalidating the last full submission first.
    pub fn push(&mut self, value: T) -> usize {
        self.state.changed();
        self.raw.push(value)
    }

    /// Replaces an existing value, conservatively invalidating even equal values.
    pub fn set(&mut self, index: u32, value: T) {
        self.state.changed();
        self.raw.set(index, value);
    }

    /// Invalidates before exposing arbitrary mutable CPU values.
    pub fn values_mut(&mut self) -> &mut Vec<T> {
        self.state.changed();
        self.raw.values_mut()
    }

    /// Removes a value, moving the last value into its slot.
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.state.changed();
        self.raw.swap_remove(index)
    }

    /// Updates the label. The next nonempty write processes it before skipping.
    pub fn set_label(&mut self, label: Option<&str>) {
        self.raw.set_label(label);
    }

    /// Returns the last write decision for externally sampled diagnostics.
    ///
    /// This does not emit a span or event per buffer. A queued outcome describes
    /// a CPU write request after the Rust API returned, not GPU completion.
    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    pub fn last_upload_outcome(&self) -> BinnedInputUploadOutcome {
        self.last_write
    }

    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    pub(crate) fn upload_mode_name() -> &'static str {
        UploadMode::runtime().name()
    }

    /// Writes the entire nonempty buffer according to the selected upload mode.
    ///
    /// Dirty mode reserves before comparing destination identity, so growth and
    /// same-capacity label changes cannot reuse a stale acknowledged destination.
    /// Empty and skipped writes never acknowledge a new submission.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        #[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
        {
            self.last_write = BinnedInputUploadOutcome::default();
        }
        if self.mode == UploadMode::Force {
            // Preserve the stock force baseline, without dirty checks or byte copies.
            self.request_upload(device, queue);
            return;
        }
        if self.raw.is_empty() {
            return;
        }
        self.prepare_device(device);
        self.raw.reserve(self.raw.len(), device);
        let destination = self.raw.buffer().map(|buffer| buffer.id());
        if !self.state.needs_upload(destination, self.raw.len()) {
            #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
            {
                self.last_write.shadow_checked = true;
                if !self
                    .submitted
                    .matches(bytemuck::cast_slice(self.raw.values()))
                {
                    self.last_write.shadow_mismatch = true;
                    self.state.changed();
                    bevy_log::error!(
                        "native_binned_dirty_mismatch: unchanged generation differs from last submitted bytes; forcing upload"
                    );
                    self.request_upload(device, queue);
                    return;
                }
            }
            #[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
            {
                self.last_write.skipped = true;
            }
            return;
        }
        self.request_upload(device, queue);
    }

    fn prepare_device(&mut self, device: &RenderDevice) {
        if self
            .device
            .as_ref()
            .is_some_and(|previous| !previous.same_render_device(device))
        {
            self.state.invalidate_destination();
            let mut replacement = RawBufferVec::new(self.usage);
            replacement.set_label(self.raw.get_label());
            core::mem::swap(replacement.values_mut(), self.raw.values_mut());
            self.raw = replacement;
        }
        if self
            .device
            .as_ref()
            .is_none_or(|previous| !previous.same_render_device(device))
        {
            self.device = Some(device.clone());
        }
    }

    fn request_upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
        {
            self.last_write.queued_equal = !self.raw.is_empty()
                && self
                    .submitted
                    .matches(bytemuck::cast_slice(self.raw.values()));
        }
        self.raw.write_buffer(device, queue);
        let destination = self.raw.buffer().map(|buffer| buffer.id());
        let queued = !self.raw.is_empty() && destination.is_some();
        #[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
        {
            self.last_write.queued = queued;
        }
        if queued {
            self.state.queued(destination, self.raw.len());
            #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
            self.submitted
                .submitted(bytemuck::cast_slice(self.raw.values()));
        }
    }
}

impl<T: NoUninit> Deref for BinnedInputBuffer<T> {
    type Target = RawBufferVec<T>;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

/// The latest full-upload decision, available only for diagnostic sampling.
#[cfg(any(test, feature = "native_binned_dirty_upload_diagnostics"))]
#[derive(Clone, Copy, Default, Debug)]
pub struct BinnedInputUploadOutcome {
    /// A nonempty full CPU queue write returned from its Rust API.
    pub queued: bool,
    /// Diagnostic bytes were equal despite a conservative dirty indication.
    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    pub queued_equal: bool,
    /// An unchanged nonempty destination was retained without a queue write.
    pub skipped: bool,
    /// Independent last-submitted bytes were checked on this would-skip path.
    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    pub shadow_checked: bool,
    /// Independent bytes differed. An error was logged and a full write forced.
    #[cfg(feature = "native_binned_dirty_upload_diagnostics")]
    pub shadow_mismatch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadMode {
    Force,
    Dirty,
}

impl UploadMode {
    fn parse(raw: Option<&str>) -> Self {
        if raw.is_none() || raw == Some("dirty") {
            Self::Dirty
        } else {
            Self::Force
        }
    }

    fn from_env(raw: Result<String, std::env::VarError>) -> Self {
        match raw {
            Ok(raw) => Self::parse(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(std::env::VarError::NotUnicode(_)) => Self::Force,
        }
    }

    fn runtime() -> Self {
        static MODE: OnceLock<UploadMode> = OnceLock::new();
        *MODE.get_or_init(|| Self::from_env(std::env::var("BEVY_NATIVE_BINNED_UPLOADS")))
    }

    fn name(self) -> &'static str {
        match self {
            Self::Force => "force",
            Self::Dirty => "dirty",
        }
    }
}

// Dropping the acknowledgement before each mutation prevents revision-wrap aliases
// and covers panics after partial Vec mutation or before a mutable borrow returns.
struct UploadState<D> {
    revision: u64,
    queued: Option<(u64, D, usize)>,
}

impl<D> Default for UploadState<D> {
    fn default() -> Self {
        Self {
            revision: 0,
            queued: None,
        }
    }
}

impl<D: Copy + Eq> UploadState<D> {
    fn changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.queued = None;
    }

    fn invalidate_destination(&mut self) {
        self.queued = None;
    }

    fn needs_upload(&self, destination: Option<D>, len: usize) -> bool {
        len > 0
            && (destination.is_none()
                || self.queued != destination.map(|destination| (self.revision, destination, len)))
    }

    // Acknowledges only a full nonempty queue request after its Rust API returned.
    fn queued(&mut self, destination: Option<D>, len: usize) {
        if len > 0 {
            self.queued = destination.map(|destination| (self.revision, destination, len));
        }
    }
}

#[cfg(feature = "native_binned_dirty_upload_diagnostics")]
#[derive(Default)]
struct SubmittedBytes {
    bytes: Option<Vec<u8>>,
}

#[cfg(feature = "native_binned_dirty_upload_diagnostics")]
impl SubmittedBytes {
    fn matches(&self, bytes: &[u8]) -> bool {
        self.bytes.as_deref() == Some(bytes)
    }

    fn submitted(&mut self, bytes: &[u8]) {
        let submitted = self.bytes.get_or_insert_with(Vec::new);
        submitted.clear();
        submitted.extend_from_slice(bytes);
    }
}

#[cfg(test)]
#[path = "binned_input_tests.rs"]
mod tests;
