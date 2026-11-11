use super::{BinnedInputBuffer, UploadMode, UploadState};
use crate::{
    render_resource::{Buffer, BufferId},
    test_utils::create_dummy_device,
};
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
    let initial = buffer.buffer().map(Buffer::id);
    assert!(initial.is_some());
    buffer.set_label(Some("replacement"));
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(Buffer::id), initial);
    let labeled = buffer.buffer().map(Buffer::id);
    buffer.push(20);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(Buffer::id), labeled);
    let grown = buffer.buffer().map(Buffer::id);
    buffer.values_mut().clear();
    buffer.write_buffer(&device, &queue);
    assert!(!buffer.last_write.queued);
    assert!(!buffer.last_write.skipped);
    assert!(buffer.state.queued.is_none());
    buffer.values_mut().extend([30, 40]);
    buffer.write_buffer(&device, &queue);
    assert!(buffer.last_write.queued);
    assert_eq!(buffer.buffer().map(Buffer::id), grown);

    let cloned_device = device.clone();
    assert!(device.same_render_device(&cloned_device));
    buffer.write_buffer(&cloned_device, &queue);
    assert!(buffer.last_write.skipped);
    assert_eq!(buffer.buffer().map(Buffer::id), grown);

    let rewrapped_device = crate::renderer::RenderDevice::from(device.wgpu_device().clone());
    assert!(!device.same_render_device(&rewrapped_device));
    buffer.write_buffer(&rewrapped_device, &queue);
    assert!(buffer.last_write.queued);
    let rewrapped = buffer.buffer().map(Buffer::id);
    assert_ne!(rewrapped, grown);

    let (replacement_device, replacement_queue) = create_dummy_device();
    assert!(!device.same_render_device(&replacement_device));
    buffer.write_buffer(&replacement_device, &replacement_queue);
    assert!(buffer.last_write.queued);
    assert_ne!(buffer.buffer().map(Buffer::id), rewrapped);
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

fn create_metal_device(
) -> Result<(crate::renderer::RenderDevice, crate::renderer::RenderQueue), Box<dyn std::error::Error>>
{
    use bevy_platform::future::block_on;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    Ok((device.into(), crate::renderer::RenderQueue::new(queue)))
}

fn check_metal_readback(
    input: &BinnedInputBuffer<u32>,
    device: &crate::renderer::RenderDevice,
    queue: &crate::renderer::RenderQueue,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = (input.len() * size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("binned input test readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let source = input
        .buffer()
        .ok_or_else(|| std::io::Error::other("missing uploaded buffer"))?;
    let scope = device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    let submitted = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submitted),
        timeout: Some(std::time::Duration::from_secs(10)),
    })?;
    receiver.recv_timeout(std::time::Duration::from_secs(1))??;
    if let Some(error) = bevy_platform::future::block_on(scope.pop()) {
        return Err(error.into());
    }
    let bytes = staging.slice(..).get_mapped_range()?;
    assert_eq!(&*bytes, bytemuck::cast_slice::<u32, u8>(input.values()));
    drop(bytes);
    staging.unmap();
    Ok(())
}

#[test]
#[ignore = "requires a Metal GPU; run explicitly for device replacement validation"]
fn metal_independent_instances_reupload_and_read_back_current_bytes(
) -> Result<(), Box<dyn std::error::Error>> {
    let (device, queue) = create_metal_device()?;
    let mut input = BinnedInputBuffer::with_mode(
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        UploadMode::Dirty,
    );
    input.values_mut().extend([30u32, 40]);
    input.write_buffer(&device, &queue);
    check_metal_readback(&input, &device, &queue)?;
    let original = input.buffer().map(Buffer::id);
    let (replacement, replacement_queue) = create_metal_device()?;
    eprintln!(
        "independent_metal_wgpu_devices_equal={}",
        device.wgpu_device() == replacement.wgpu_device()
    );
    input.write_buffer(&replacement, &replacement_queue);
    assert!(
        input.last_write.queued,
        "new instance/device must receive a full upload"
    );
    assert_ne!(input.buffer().map(Buffer::id), original);
    check_metal_readback(&input, &replacement, &replacement_queue)?;
    Ok(())
}
