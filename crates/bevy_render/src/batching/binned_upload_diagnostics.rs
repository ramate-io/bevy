//! Bounded, completed writer summaries; compiled out of clean upload builds.
use std::sync::OnceLock;

use bytemuck::NoUninit;

use crate::render_phase::BinnedInputBuffer;

struct Config {
    enabled: bool,
    sample_frames: u32,
    max_samples: u32,
}

impl Config {
    fn runtime() -> &'static Self {
        static CONFIG: OnceLock<Config> = OnceLock::new();
        CONFIG.get_or_init(|| Self {
            enabled: std::env::var("BEVY_NATIVE_BINNED_UPLOAD_DIAGNOSTICS").as_deref() == Ok("1"),
            sample_frames: std::env::var("BEVY_NATIVE_BINNED_UPLOAD_SAMPLE_FRAMES")
                .ok()
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(120)
                .max(1),
            max_samples: std::env::var("BEVY_NATIVE_BINNED_UPLOAD_MAX_SAMPLES")
                .ok()
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(64)
                .clamp(1, 1024),
        })
    }

    fn select(&self, frame: u32, cadence: &mut (u32, Option<u32>)) -> Sample {
        if !self.enabled || !frame.is_multiple_of(self.sample_frames) || cadence.1 == Some(frame) {
            return Sample::Skip;
        }
        cadence.1 = Some(frame);
        if cadence.0 >= self.max_samples {
            if cadence.0 == self.max_samples {
                cadence.0 += 1;
                return Sample::Limit;
            }
            return Sample::Skip;
        }
        cadence.0 += 1;
        Sample::Capture(cadence.0)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Sample {
    Skip,
    Limit,
    Capture(u32),
}

#[derive(Default)]
struct Counts {
    requests: u64,
    logical_bytes: u64,
    queue_calls: u64,
    upload_bytes: u64,
    skipped_uploads: u64,
    skipped_bytes: u64,
    shadow_checks: u64,
    shadow_mismatches: u64,
    queued_equal_uploads: u64,
    queued_equal_bytes: u64,
}

impl Counts {
    fn record<T: NoUninit>(&mut self, buffer: &BinnedInputBuffer<T>) {
        let bytes = (buffer.len() * size_of::<T>()) as u64;
        let outcome = buffer.last_upload_outcome();
        self.requests += u64::from(!buffer.is_empty());
        self.logical_bytes += bytes;
        self.queue_calls += u64::from(outcome.queued);
        self.upload_bytes += if outcome.queued { bytes } else { 0 };
        self.skipped_uploads += u64::from(outcome.skipped);
        self.skipped_bytes += if outcome.skipped { bytes } else { 0 };
        self.shadow_checks += u64::from(outcome.shadow_checked);
        self.shadow_mismatches += u64::from(outcome.shadow_mismatch);
        self.queued_equal_uploads += u64::from(outcome.queued_equal);
        self.queued_equal_bytes += if outcome.queued_equal { bytes } else { 0 };
    }

    fn conservation_valid(&self) -> bool {
        self.requests == self.queue_calls + self.skipped_uploads
            && self.logical_bytes == self.upload_bytes + self.skipped_bytes
    }
}

pub(super) struct Census {
    frame: u32,
    phase: &'static str,
    sample_index: u32,
    pub views: usize,
    pub candidate_sets: usize,
    pub eligible_sets: usize,
    pub skipped_mode_sets: usize,
    pub skipped_no_representative: usize,
    pub skipped_no_metadata: usize,
    pub skipped_missing_buffers: usize,
    pub skipped_index_overflow: usize,
    pub indexed_jobs: usize,
    pub non_indexed_jobs: usize,
    pub fan_write_calls: usize,
    buffers: [Counts; 3],
}

impl Census {
    pub fn begin(
        frame: u32,
        phase: &'static str,
        cadence: &mut (u32, Option<u32>),
    ) -> Option<Self> {
        let config = Config::runtime();
        let sample_index = match config.select(frame, cadence) {
            Sample::Skip => return None,
            Sample::Limit => {
                bevy_log::info!(target: "bevy_render::native_binned_uploads",
                    "native_binned_upload_census_limit frame={} phase=\"{}\" emitted_samples={} max_samples={} sample_frames={}",
                    frame, phase, config.max_samples, config.max_samples, config.sample_frames);
                return None;
            }
            Sample::Capture(index) => index,
        };
        Some(Self {
            frame,
            phase,
            sample_index,
            views: 0,
            candidate_sets: 0,
            eligible_sets: 0,
            skipped_mode_sets: 0,
            skipped_no_representative: 0,
            skipped_no_metadata: 0,
            skipped_missing_buffers: 0,
            skipped_index_overflow: 0,
            indexed_jobs: 0,
            non_indexed_jobs: 0,
            fan_write_calls: 0,
            buffers: Default::default(),
        })
    }

    pub fn record<T: NoUninit>(&mut self, kind: usize, buffer: &BinnedInputBuffer<T>) {
        self.buffers[kind].record(buffer);
    }

    // Only called after the native writer completes, never from Drop or mid-loop.
    pub fn finish(self) {
        let config = Config::runtime();
        let mode = BinnedInputBuffer::<u32>::upload_mode_name();
        for (kind, counts) in ["instances", "metadata", "lookup"]
            .into_iter()
            .zip(self.buffers)
        {
            bevy_log::info!(target: "bevy_render::native_binned_uploads",
                "native_binned_upload_census frame={} phase=\"{}\" mode={} buffer={} sample_index={} max_samples={} sample_frames={} views={} candidate_sets={} eligible_sets={} skipped_mode_sets={} skipped_no_representative={} skipped_no_metadata={} skipped_missing_buffers={} skipped_index_overflow={} indexed_jobs={} non_indexed_jobs={} fan_write_calls={} requests={} logical_bytes={} queue_calls={} upload_bytes={} skipped_uploads={} skipped_bytes={} shadow_checks={} shadow_mismatches={} queued_equal_uploads={} queued_equal_bytes={} conservation_valid={}",
                self.frame, self.phase, mode, kind, self.sample_index, config.max_samples,
                config.sample_frames, self.views, self.candidate_sets, self.eligible_sets,
                self.skipped_mode_sets, self.skipped_no_representative, self.skipped_no_metadata,
                self.skipped_missing_buffers, self.skipped_index_overflow, self.indexed_jobs,
                self.non_indexed_jobs, self.fan_write_calls, counts.requests, counts.logical_bytes,
                counts.queue_calls, counts.upload_bytes, counts.skipped_uploads, counts.skipped_bytes,
                counts.shadow_checks, counts.shadow_mismatches, counts.queued_equal_uploads,
                counts.queued_equal_bytes, counts.conservation_valid());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Counts, Sample};
    use crate::{render_phase::BinnedInputBuffer, test_utils::create_dummy_device};
    use wgpu::BufferUsages;

    #[test]
    fn cadence_is_bounded_deduplicated_and_phase_local() {
        let config = Config {
            enabled: true,
            sample_frames: 120,
            max_samples: 2,
        };
        let mut first = (0, None);
        let mut second = (0, None);
        assert_eq!(config.select(1, &mut first), Sample::Skip);
        assert_eq!(config.select(120, &mut first), Sample::Capture(1));
        assert_eq!(config.select(120, &mut first), Sample::Skip);
        assert_eq!(config.select(120, &mut second), Sample::Capture(1));
        assert_eq!(config.select(240, &mut first), Sample::Capture(2));
        assert_eq!(config.select(360, &mut first), Sample::Limit);
        assert_eq!(config.select(480, &mut first), Sample::Skip);
        assert_eq!(config.select(480, &mut second), Sample::Capture(2));
    }

    #[test]
    fn disabled_census_never_advances_cadence() {
        let config = Config {
            enabled: false,
            sample_frames: 1,
            max_samples: 1,
        };
        let mut cadence = (0, None);
        assert_eq!(config.select(0, &mut cadence), Sample::Skip);
        assert_eq!(cadence, (0, None));
    }

    #[test]
    fn actual_buffer_requests_conserve_calls_and_bytes_including_empty() {
        let (device, queue) = create_dummy_device();
        let mut buffer = BinnedInputBuffer::new(BufferUsages::STORAGE);
        let mut counts = Counts::default();
        buffer.write_buffer(&device, &queue);
        counts.record::<u32>(&buffer);
        assert_eq!(counts.requests, 0);
        assert!(counts.conservation_valid());
        buffer.push(12u32);
        buffer.write_buffer(&device, &queue);
        counts.record(&buffer);
        buffer.write_buffer(&device, &queue);
        counts.record(&buffer);
        assert_eq!(counts.requests, 2);
        assert_eq!(counts.logical_bytes, 8);
        assert!(counts.queue_calls >= 1);
        assert!(counts.conservation_valid());
        counts.logical_bytes += 1;
        assert!(!counts.conservation_valid());
    }
}
