use super::{BinnedInputBuffer, UploadMode, UploadState};
use crate::{render_resource::BufferId, test_utils::create_dummy_device};
use wgpu::BufferUsages;

fn acknowledge(buffer: &mut BinnedInputBuffer<u32>, id: BufferId) {
    buffer.state.queued(Some(id), buffer.len());
}

#[test]
fn mutations_cover_equal_length_changes_and_swap_displacement() {
    let mut buffer = BinnedInputBuffer::with_mode(BufferUsages::STORAGE, UploadMode::Dirty);
    let id = BufferId::new();
    buffer.push(10);
    buffer.push(20);
    acknowledge(&mut buffer, id);
    assert!(!buffer.state.needs_upload(Some(id), 2));
    buffer.set(0, 11);
    assert!(buffer.state.needs_upload(Some(id), 2));
    acknowledge(&mut buffer, id);
    assert_eq!(buffer.swap_remove(0), 11);
    assert_eq!(buffer.values(), &[20]);
    assert!(buffer.state.needs_upload(Some(id), 1));
    buffer.push(30);
    acknowledge(&mut buffer, id);
    buffer.values_mut()[0] = 21;
    assert!(buffer.state.needs_upload(Some(id), 2));
}

#[test]
fn clear_refill_and_mutable_metadata_updates_are_owner_local() {
    let mut first = BinnedInputBuffer::with_mode(BufferUsages::STORAGE, UploadMode::Dirty);
    let mut second = BinnedInputBuffer::with_mode(BufferUsages::STORAGE, UploadMode::Dirty);
    let id = BufferId::new();
    first.push(1);
    second.push(2);
    acknowledge(&mut first, id);
    acknowledge(&mut second, id);
    first.values_mut().clear();
    first.values_mut().resize(1, 3);
    assert!(first.state.needs_upload(Some(id), 1));
    assert!(!second.state.needs_upload(Some(id), 1));
    acknowledge(&mut first, id);
    first.values_mut()[0] += 1;
    assert!(first.state.needs_upload(Some(id), 1));
    assert_eq!(second.values(), &[2]);
}

#[test]
fn missing_replaced_resized_and_invalidated_destinations_need_uploads() {
    let mut state = UploadState::default();
    assert!(state.needs_upload(Some(11), 3));
    state.queued(Some(11), 3);
    assert!(!state.needs_upload(Some(11), 3));
    assert!(state.needs_upload(None, 3));
    assert!(state.needs_upload(Some(12), 3));
    assert!(state.needs_upload(Some(11), 4));
    state.invalidate_destination();
    assert!(state.needs_upload(Some(11), 3));
}

#[test]
fn empty_and_wrapped_revisions_never_acknowledge_dirty_data() {
    let mut state = UploadState::default();
    state.queued(Some(7), 1);
    state.changed();
    assert!(!state.needs_upload(Some(7), 0));
    state.queued(Some(7), 0);
    assert!(state.needs_upload(Some(7), 1));
    state.revision = u64::MAX;
    state.queued(Some(7), 1);
    state.changed();
    assert_eq!(state.revision, 0);
    assert!(state.needs_upload(Some(7), 1));
}

#[test]
fn process_mode_defaults_dirty_only_after_explicit_feature_activation() {
    assert_eq!(UploadMode::parse(None), UploadMode::Dirty);
    assert_eq!(UploadMode::parse(Some("force")), UploadMode::Force);
    assert_eq!(UploadMode::parse(Some("unknown")), UploadMode::Force);
    assert_eq!(UploadMode::parse(Some("dirty")), UploadMode::Dirty);
    assert_eq!(
        UploadMode::from_env(Err(std::env::VarError::NotPresent)),
        UploadMode::Dirty
    );
    assert_eq!(
        UploadMode::from_env(Err(std::env::VarError::NotUnicode(
            std::ffi::OsString::new()
        ))),
        UploadMode::Force
    );
}

#[test]
fn noop_device_writes_skip_unchanged_but_force_and_mutations_write() {
    let (device, queue) = create_dummy_device();
    for mode in [UploadMode::Force, UploadMode::Dirty] {
        let mut buffer = BinnedInputBuffer::with_mode(BufferUsages::STORAGE, mode);
        buffer.push(10u32);
        buffer.write_buffer(&device, &queue);
        assert!(buffer.last_write.queued);
        buffer.write_buffer(&device, &queue);
        assert_eq!(buffer.last_write.queued, mode == UploadMode::Force);
        assert_eq!(buffer.last_write.skipped, mode == UploadMode::Dirty);
        buffer.set(0, 20);
        buffer.write_buffer(&device, &queue);
        assert!(buffer.last_write.queued);
        assert!(!buffer.last_write.skipped);
    }
}

#[test]
fn noop_device_label_growth_replacement_and_empty_refill_force_writes() {
    let (device, queue) = create_dummy_device();
    let mut buffer = BinnedInputBuffer::with_mode(
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        UploadMode::Dirty,
    );
    buffer.set_label(Some("initial"));
    buffer.push(10u32);
    buffer.write_buffer(&device, &queue);
    let initial = buffer.buffer().map(|buffer| buffer.id());
    assert!(initial.is_some());
    buffer.set_label(Some("replacement"));
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(|buffer| buffer.id()), initial);
    let labeled = buffer.buffer().map(|buffer| buffer.id());
    buffer.push(20);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(|buffer| buffer.id()), labeled);
    let grown = buffer.buffer().map(|buffer| buffer.id());
    buffer.values_mut().clear();
    buffer.write_buffer(&device, &queue);
    assert!(!buffer.last_write.queued);
    assert!(!buffer.last_write.skipped);
    assert!(buffer.state.queued.is_none());
    buffer.values_mut().extend([30, 40]);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_eq!(buffer.buffer().map(|buffer| buffer.id()), grown);

    let (replacement_device, replacement_queue) = create_dummy_device();
    buffer.write_buffer(&replacement_device, &replacement_queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(|buffer| buffer.id()), grown);
    assert_eq!(buffer.values(), &[30, 40]);
    assert_eq!(buffer.get_label(), Some("replacement"));
    assert!(buffer.buffer().is_some_and(|buffer| buffer
        .usage()
        .contains(BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)));
    buffer.write_buffer(&replacement_device, &replacement_queue);
    assert!(buffer.last_write.skipped);
}

#[cfg(feature = "native_binned_dirty_upload_diagnostics")]
#[test]
fn independent_shadow_checks_every_skip_and_fails_closed_on_mismatch() {
    let (device, queue) = create_dummy_device();
    let mut buffer = BinnedInputBuffer::with_mode(BufferUsages::STORAGE, UploadMode::Dirty);
    buffer.push(1u32);
    buffer.write_buffer(&device, &queue);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.skipped && buffer.last_write.shadow_checked);
    buffer.set(0, 1);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued && buffer.last_write.queued_equal);
    // Intentionally bypass the wrapper's mutation tracking from this private
    // test to prove that the independent shadow does not trust its generation.
    buffer.raw.values_mut()[0] = 2;
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.shadow_checked && buffer.last_write.shadow_mismatch);
    assert!(buffer.last_write.queued && !buffer.last_write.skipped);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.shadow_checked && !buffer.last_write.shadow_mismatch);
    assert!(buffer.last_write.skipped);
}
