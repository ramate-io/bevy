//! Default-off ordering experiment for the stock 3D shared-metadata writers.
use bevy_app::SubApp;
use bevy_core_pipeline::{
    core_3d::{AlphaMask3d, Opaque3d},
    deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
    prepass::{AlphaMask3dPrepass, Opaque3dPrepass},
};
use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule};
use bevy_render::{render_phase::BinnedPhaseBufferWriterSet, Render};

use crate::{render::light::Shadow, render::mesh::MeshPipeline, wireframe::Wireframe3d};

pub(super) fn configure_if_enabled(render_app: &mut SubApp) {
    let enabled =
        std::env::var("BEVY_EXPERIMENT_ORDERED_METADATA_WRITERS").is_ok_and(|value| value == "1");
    if !enabled {
        return;
    }
    render_app.edit_schedule(Render, |schedule| configure_order(schedule, true));
    tracing::info!(
        target: "bevy_pbr::metadata_writer_order",
        "ordered_metadata_writers enabled; stock_3d_writer_labels=8"
    );
}

fn writer_sets() -> [BinnedPhaseBufferWriterSet; 8] {
    [
        BinnedPhaseBufferWriterSet::new::<Opaque3dPrepass, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<AlphaMask3dPrepass, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<Opaque3dDeferred, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<AlphaMask3dDeferred, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<Opaque3d, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<AlphaMask3d, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<Shadow, MeshPipeline>(),
        BinnedPhaseBufferWriterSet::new::<Wireframe3d, MeshPipeline>(),
    ]
}

fn configure_order(schedule: &mut Schedule, enabled: bool) {
    if !enabled {
        return;
    }
    // These writers already conflict on ResMut<SceneUnpackingBuffers>. Order
    // their existing labels rather than adding systems or moving outer stages.
    // Empty optional phase sets collapse in the pinned schedule implementation.
    let [a, b, c, d, e, f, g, h] = writer_sets();
    schedule.configure_sets((a, b, c, d, e, f, g, h).chain());
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{ResMut, Resource, World};
    use bevy_render::RenderSystems;

    #[derive(Default, Resource)]
    struct Writes(Vec<i32>);

    fn clear(mut writes: ResMut<Writes>) {
        writes.0.push(-1);
    }

    fn collect(mut writes: ResMut<Writes>) {
        writes.0.push(99);
    }

    fn check_existing_labels(indices: &[usize]) -> Result<(), String> {
        let mut schedule = Render::base_schedule();
        let mut world = World::new();
        world.insert_resource(Writes::default());
        let sets = writer_sets();
        // Reverse insertion rules out an accidental registration-order pass.
        for &index in indices.iter().rev() {
            schedule.add_systems(
                (move |mut writes: ResMut<Writes>| writes.0.push(index as i32))
                    .in_set(sets[index])
                    .in_set(RenderSystems::PrepareResourcesWritePhaseBuffers),
            );
        }
        schedule.add_systems((
            clear.in_set(RenderSystems::PrepareResources),
            collect.in_set(RenderSystems::PrepareResourcesCollectPhaseBuffers),
        ));
        let registered = schedule.systems_len();
        configure_order(&mut schedule, true);
        assert_eq!(schedule.systems_len(), registered);
        schedule
            .initialize(&mut world)
            .map_err(|error| format!("{error:?}"))?;
        schedule.run(&mut world);
        let mut expected = vec![-1];
        expected.extend(indices.iter().map(|&index| index as i32));
        expected.push(99);
        assert_eq!(
            world.get_resource::<Writes>().ok_or("missing writes")?.0,
            expected
        );
        Ok(())
    }

    #[test]
    fn stock_writer_labels_keep_outer_stages_and_do_not_add_systems() -> Result<(), String> {
        check_existing_labels(&[0, 1, 2, 3, 4, 5, 6, 7])
    }

    #[test]
    fn missing_optional_labels_preserve_transitive_writer_order() -> Result<(), String> {
        check_existing_labels(&[0, 2, 4, 6])
    }

    #[test]
    fn disabled_order_adds_no_dependencies_and_pipeline_identity_is_distinct() {
        let mut schedule = Render::base_schedule();
        let before = (
            schedule.graph().dependency().graph().node_count(),
            schedule.graph().dependency().graph().all_edges().len(),
        );
        configure_order(&mut schedule, false);
        assert_eq!(
            (
                schedule.graph().dependency().graph().node_count(),
                schedule.graph().dependency().graph().all_edges().len(),
            ),
            before
        );
        assert_ne!(
            BinnedPhaseBufferWriterSet::new::<Opaque3d, MeshPipeline>(),
            BinnedPhaseBufferWriterSet::new::<Opaque3d, ()>()
        );
        assert_ne!(
            BinnedPhaseBufferWriterSet::new::<Opaque3d, MeshPipeline>(),
            BinnedPhaseBufferWriterSet::new::<Shadow, MeshPipeline>()
        );
    }
}
