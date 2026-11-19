use core::{hash::Hash, num::NonZeroU64};

use bevy_ecs::resource::Resource;
use bevy_platform::collections::{HashMap, hash_map::Entry};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutId, BindingResource, Buffer,
        BufferBinding, BufferId,
    },
    renderer::RenderDevice,
};
use smallvec::SmallVec;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum JobBindGroupRecipe {
    UnpackIndexed,
    UnpackNonIndexed,
    AllocateIndexed,
    AllocateNonIndexed,
}

impl JobBindGroupRecipe {
    fn label(self) -> &'static str {
        match self {
            Self::UnpackIndexed => "bin unpacking indexed bind group",
            Self::UnpackNonIndexed => "bin unpacking non-indexed bind group",
            Self::AllocateIndexed => "uniform allocation indexed bind group",
            Self::AllocateNonIndexed => "uniform allocation non-indexed bind group",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct JobBufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

impl<'a> JobBufferBinding<'a> {
    pub fn entire(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            size: None,
        }
    }

    fn key(self) -> JobBindingKey {
        JobBindingKey {
            buffer: self.buffer.id(),
            offset: self.offset,
            size: self.size,
        }
    }

    fn resource(self) -> BindingResource<'a> {
        BindingResource::Buffer(BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
        })
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct JobBindingKey {
    buffer: BufferId,
    offset: u64,
    size: Option<NonZeroU64>,
}

/// Bindings are sequential and ordered. The layout ID includes binding type
/// and dynamic-offset semantics. Neither job family supplies dynamic offsets.
#[derive(Clone, Hash, PartialEq, Eq)]
struct JobBindGroupKey {
    recipe: JobBindGroupRecipe,
    layout: BindGroupLayoutId,
    bindings: SmallVec<[JobBindingKey; 5]>,
}

struct RetainedGroup<V> {
    last_seen_frame: u64,
    value: V,
}

struct FrameCache<K, V> {
    groups: HashMap<K, RetainedGroup<V>>,
    frame: u64,
}

impl<K, V> Default for FrameCache<K, V> {
    fn default() -> Self {
        Self {
            groups: HashMap::default(),
            frame: 0,
        }
    }
}

impl<K: Eq + Hash, V> FrameCache<K, V> {
    fn get_or_create(&mut self, key: K, create: impl FnOnce() -> V) -> (&V, bool) {
        match self.groups.entry(key) {
            Entry::Occupied(entry) => {
                let group = entry.into_mut();
                group.last_seen_frame = self.frame;
                (&group.value, true)
            }
            Entry::Vacant(entry) => {
                let group = entry.insert(RetainedGroup {
                    last_seen_frame: self.frame,
                    value: create(),
                });
                (&group.value, false)
            }
        }
    }

    fn finish_frame(&mut self) {
        // collect_buffers_for_phase alternates the phase's work-item and
        // indirect-parameter buffer ownership. A group may be used every
        // other frame on a stable view, so allow one missed preparation.
        self.groups
            .retain(|_, group| self.frame.wrapping_sub(group.last_seen_frame) <= 1);
        self.frame = self.frame.wrapping_add(1);
    }
}

#[cfg(feature = "trace")]
#[derive(Default)]
struct JobGroupSample {
    frame: u32,
    requests: u64,
    reused: u64,
}

/// Internal resource for the per-job bind groups. RenderStartup replaces it
/// when the render device is recreated.
#[doc(hidden)]
#[derive(Resource)]
pub struct RetainedJobBindGroupCache {
    enabled: bool,
    groups: FrameCache<JobBindGroupKey, BindGroup>,
    #[cfg(feature = "trace")]
    sample_enabled: bool,
    #[cfg(feature = "trace")]
    sample: Option<JobGroupSample>,
}

impl Default for RetainedJobBindGroupCache {
    fn default() -> Self {
        let toggle = std::env::var_os("BEVY_EXPERIMENT_REUSE_PREPROCESS_JOB_BIND_GROUPS");
        Self {
            enabled: toggle.as_deref().is_some_and(|value| value == "1"),
            groups: FrameCache::default(),
            #[cfg(feature = "trace")]
            sample_enabled: toggle.is_some_and(|value| value == "0" || value == "1"),
            #[cfg(feature = "trace")]
            sample: None,
        }
    }
}

impl RetainedJobBindGroupCache {
    #[cfg(feature = "trace")]
    pub(super) fn begin_sample(&mut self, frame: u32) -> bool {
        self.sample = (self.sample_enabled && frame.is_multiple_of(180)).then(|| JobGroupSample {
            frame,
            ..Default::default()
        });
        self.sample.is_some()
    }

    pub(super) fn get_or_create<const N: usize>(
        &mut self,
        device: &RenderDevice,
        recipe: JobBindGroupRecipe,
        layout: &BindGroupLayout,
        bindings: [JobBufferBinding<'_>; N],
    ) -> BindGroup {
        let create = || {
            let entries: [BindGroupEntry<'_>; N] = core::array::from_fn(|index| BindGroupEntry {
                binding: index as u32,
                resource: bindings[index].resource(),
            });
            device.create_bind_group(recipe.label(), layout, &entries)
        };
        let (group, _reused) = if self.enabled {
            let key = JobBindGroupKey {
                recipe,
                layout: layout.id(),
                bindings: bindings.iter().map(|binding| binding.key()).collect(),
            };
            let (group, reused) = self.groups.get_or_create(key, create);
            (group.clone(), reused)
        } else {
            (create(), false)
        };
        #[cfg(feature = "trace")]
        if let Some(sample) = self.sample.as_mut() {
            sample.requests += 1;
            sample.reused += u64::from(_reused);
        }
        group
    }

    pub(super) fn finish_frame(&mut self) {
        self.groups.finish_frame();
        #[cfg(feature = "trace")]
        if let Some(sample) = self.sample.take() {
            let created = sample.requests - sample.reused;
            bevy_log::info!(target: "bevy_pbr::preprocess_job_bind_groups",
                "preprocess_job_bind_group_census enabled={} frame={} requests={} reused={} created={}",
                self.enabled, sample.frame, sample.requests, sample.reused, created);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> JobBindGroupKey {
        JobBindGroupKey {
            recipe: JobBindGroupRecipe::UnpackIndexed,
            layout: BindGroupLayoutId::new(),
            bindings: SmallVec::from_iter([
                JobBindingKey {
                    buffer: BufferId::new(),
                    offset: 256,
                    size: NonZeroU64::new(64),
                },
                JobBindingKey {
                    buffer: BufferId::new(),
                    offset: 0,
                    size: None,
                },
            ]),
        }
    }

    #[test]
    fn job_bind_group_reuses_alternating_buffers_and_expires_removed_entries() {
        use std::{cell::Cell, rc::Rc};

        struct DropSpy {
            slot: u8,
            dropped: Rc<Cell<u8>>,
        }
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.dropped.set(self.dropped.get() + 1);
            }
        }

        let mut cache = FrameCache::default();
        let a = key();
        let mut b = a.clone();
        b.bindings[1].buffer = BufferId::new();
        let dropped = Rc::new(Cell::new(0));
        for (frame, key, slot) in [(0, &a, 0), (1, &b, 1), (2, &a, 0), (3, &b, 1)] {
            let (group, reused) = cache.get_or_create(key.clone(), || DropSpy {
                slot,
                dropped: dropped.clone(),
            });
            assert_eq!((group.slot, reused), (slot, frame >= 2));
            assert!(cache.get_or_create(key.clone(), || unreachable!()).1);
            cache.finish_frame();
            assert_eq!(dropped.get(), 0);
        }
        cache.finish_frame();
        assert_eq!(dropped.get(), 1);
        cache.finish_frame();
        assert_eq!(dropped.get(), 2);
        assert!(
            !cache
                .get_or_create(a, || DropSpy {
                    slot: 0,
                    dropped: dropped.clone(),
                })
                .1
        );
        drop(cache);
        assert_eq!(dropped.get(), 3);
    }

    #[test]
    fn job_bind_group_misses_when_layout_buffer_range_or_binding_order_changes() {
        let mut cache = FrameCache::default();
        let key = key();
        let copy = || key.clone();
        assert_eq!(cache.get_or_create(copy(), || 0), (&0, false));
        cache.finish_frame();

        let mut new_layout = copy();
        new_layout.layout = BindGroupLayoutId::new();
        assert_eq!(cache.get_or_create(new_layout, || 1), (&1, false));
        let mut resized_buffer = copy();
        resized_buffer.bindings[0].buffer = BufferId::new();
        assert_eq!(cache.get_or_create(resized_buffer, || 2), (&2, false));
        let mut new_offset = copy();
        new_offset.bindings[0].offset = 512;
        assert_eq!(cache.get_or_create(new_offset, || 3), (&3, false));
        let mut entire_binding = copy();
        entire_binding.bindings[0].size = None;
        assert_eq!(cache.get_or_create(entire_binding, || 4), (&4, false));
        let mut reordered = copy();
        reordered.bindings.swap(0, 1);
        assert_eq!(cache.get_or_create(reordered, || 5), (&5, false));
        let mut non_indexed = copy();
        non_indexed.recipe = JobBindGroupRecipe::UnpackNonIndexed;
        assert_eq!(cache.get_or_create(non_indexed, || 6), (&6, false));
    }
}
