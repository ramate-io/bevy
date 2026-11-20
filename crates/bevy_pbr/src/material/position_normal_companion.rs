use bevy_app::App;
use bevy_ecs::prelude::{Resource, World};
use bevy_material::{
    key::ErasedMaterialPipelineKey, specialize::BaseSpecializeFn, OpaqueRendererMethod,
    RenderPhaseType,
};
use bevy_mesh::{
    Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayout, MeshVertexBufferLayouts,
    VertexBufferLayout,
};
use bevy_render::{
    render_resource::{
        CachedRenderPipelineId, PipelineCache, PrimitiveTopology, TextureFormat, VertexAttribute,
        VertexFormat, VertexStepMode,
    },
    RenderApp, RenderStartup,
};
use core::any::TypeId;

use super::{Material, SpecializationWorkItem};
use crate::MeshPipelineKey;

/// Experimentally prepares one position/normal-only pipeline for this material per render device.
///
/// Register at most one material. Its forward pass must support both an uncompressed UV0 layout
/// and the position/normal-only layout. The request uses the first matching real specialization's
/// key and never changes a mesh's layout or the pipeline returned for the original mesh.
pub fn prepare_position_normal_pipeline_companion<M: Material>(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(RenderStartup, |world: &mut World| {
            world.insert_resource(PositionNormalPipelineCompanion {
                material_type: TypeId::of::<M>(),
                state: CompanionState::Waiting,
            });
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompanionState {
    Waiting,
    Requested(CachedRenderPipelineId),
    Finished,
}

#[derive(Clone, Copy, PartialEq, Eq, Resource)]
pub(super) struct PositionNormalPipelineCompanion {
    material_type: TypeId,
    state: CompanionState,
}

impl PositionNormalPipelineCompanion {
    pub(super) fn accepts(&self, item: &SpecializationWorkItem) -> bool {
        self.state == CompanionState::Waiting
            && item.material_type_id == self.material_type
            && matches!(item.properties.render_phase_type, RenderPhaseType::Opaque)
            && matches!(item.properties.render_method, OpaqueRendererMethod::Forward)
            && eligible_view(item.mesh_key)
            && is_position_normal_uv0(&item.layout.0)
    }

    pub(super) fn request(
        &mut self,
        world: &mut World,
        specialize: BaseSpecializeFn,
        key: ErasedMaterialPipelineKey,
        item: &SpecializationWorkItem,
        source_pipeline: CachedRenderPipelineId,
    ) {
        let layout = world
            .resource_mut::<MeshVertexBufferLayouts>()
            .insert(position_normal_layout());
        // The normal specialization cache owns this request and its shader-loading retries.
        match specialize(world, key, &layout, &item.properties) {
            Ok(id) => {
                self.state = CompanionState::Requested(id);
                tracing::trace!(
                    target: "bevy_pbr.position_normal_companion",
                    source_pipeline = source_pipeline.id(),
                    companion_pipeline = id.id(),
                    source_entity = ?item.visible_entity,
                    "position_normal_companion.requested"
                );
            }
            Err(err) => {
                self.state = CompanionState::Finished;
                tracing::warn!(
                    target: "bevy_pbr.position_normal_companion",
                    %err,
                    "position_normal_companion.specialization_failed"
                );
            }
        }
    }

    pub(super) fn observe(
        &mut self,
        world: &World,
        item: &SpecializationWorkItem,
        id: CachedRenderPipelineId,
    ) {
        if self.state == CompanionState::Requested(id)
            && item.material_type_id == self.material_type
            && item.layout.0.layout().array_stride == 24
        {
            self.state = CompanionState::Finished;
            tracing::trace!(
                target: "bevy_pbr.position_normal_companion",
                companion_pipeline = id.id(),
                consumer_entity = ?item.visible_entity,
                ready = world.get_resource::<PipelineCache>()
                    .and_then(|cache| cache.get_render_pipeline(id)).is_some(),
                "position_normal_companion.reused"
            );
        }
    }
}

fn eligible_view(key: MeshPipelineKey) -> bool {
    use MeshPipelineKey as Key;
    key.msaa_samples() == 1
        && key.target_format() == TextureFormat::Bgra8UnormSrgb
        && key.primitive_topology() == PrimitiveTopology::TriangleList
        && (key & Key::BLEND_RESERVED_BITS) == Key::BLEND_OPAQUE
        && (key & Key::VIEW_PROJECTION_RESERVED_BITS) == Key::VIEW_PROJECTION_PERSPECTIVE
        && key.contains(Key::DEPTH_PREPASS | Key::DISTANCE_FOG)
        && !key.intersects(
            Key::NORMAL_PREPASS
                | Key::DEFERRED_PREPASS
                | Key::MOTION_VECTOR_PREPASS
                | Key::LIGHTMAPPED
                | Key::LIGHTMAP_BICUBIC_SAMPLING
                | Key::VISIBILITY_RANGE_DITHER
                | Key::MORPH_TARGETS
                | Key::HAS_PREVIOUS_MORPH
                | Key::HAS_PREVIOUS_SKIN,
        )
}

const POSITION: VertexAttribute = VertexAttribute {
    format: VertexFormat::Float32x3,
    offset: 0,
    shader_location: 0,
};
const NORMAL: VertexAttribute = VertexAttribute {
    format: VertexFormat::Float32x3,
    offset: 12,
    shader_location: 1,
};
const UV0: VertexAttribute = VertexAttribute {
    format: VertexFormat::Float32x2,
    offset: 24,
    shader_location: 2,
};

fn is_position_normal_uv0(layout: &MeshVertexBufferLayout) -> bool {
    let buffer = layout.layout();
    layout.get_attribute_compression().is_empty()
        && layout.attribute_ids()
            == [
                Mesh::ATTRIBUTE_POSITION.id,
                Mesh::ATTRIBUTE_NORMAL.id,
                Mesh::ATTRIBUTE_UV_0.id,
            ]
        && buffer.array_stride == 32
        && buffer.step_mode == VertexStepMode::Vertex
        && buffer.attributes == [POSITION, NORMAL, UV0]
}

fn position_normal_layout() -> MeshVertexBufferLayout {
    MeshVertexBufferLayout::new(
        vec![Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_NORMAL.id],
        VertexBufferLayout {
            array_stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![POSITION, NORMAL],
        },
        MeshAttributeCompressionFlags::empty(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use bevy_app::SubApp;
    use bevy_ecs::entity::Entity;
    use bevy_material::{
        key::{ErasedMaterialKey, ErasedMeshPipelineKey},
        MaterialProperties,
    };
    use bevy_mesh::MeshVertexBufferLayoutRef;
    use bevy_render::view::RetainedViewEntity;

    fn position_normal_mesh() -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, Default::default())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; 3])
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[1.0_f32; 3]; 3])
    }

    fn view_key() -> MeshPipelineKey {
        MeshPipelineKey::from_target_format(TextureFormat::Bgra8UnormSrgb)
            | MeshPipelineKey::from_bits_retain(
                MeshPipelineKey::from_primitive_topology_and_strip_index(
                    PrimitiveTopology::TriangleList,
                    None,
                )
                .bits(),
            )
            | MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE
            | MeshPipelineKey::DISTANCE_FOG
            | MeshPipelineKey::DEPTH_PREPASS
    }

    #[test]
    fn source_layout_is_exact_and_target_interns_with_real_mesh_layout() {
        let mut layouts = MeshVertexBufferLayouts::default();
        let companion = layouts.insert(position_normal_layout());
        let mesh = position_normal_mesh();
        assert_eq!(companion, mesh.get_mesh_vertex_buffer_layout(&mut layouts));
        let source_mesh = mesh.with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0_f32; 2]; 3]);
        let source = source_mesh.get_mesh_vertex_buffer_layout(&mut layouts);
        assert!(is_position_normal_uv0(&source.0));
        assert_ne!(companion, source);
        let tangent_mesh =
            source_mesh.with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[0.0_f32; 4]; 3]);
        assert!(!is_position_normal_uv0(
            &tangent_mesh.get_mesh_vertex_buffer_layout(&mut layouts).0
        ));
        assert!(eligible_view(view_key()));
        assert!(!eligible_view(
            view_key() | MeshPipelineKey::MOTION_VECTOR_PREPASS
        ));
        assert!(!eligible_view(
            view_key() | MeshPipelineKey::from_msaa_samples(4)
        ));
    }

    #[test]
    fn request_reuses_exact_key_and_properties_and_render_startup_resets_budget() {
        #[derive(Resource)]
        struct Expected(ErasedMaterialPipelineKey, Arc<MaterialProperties>);

        fn specialize_fixture(
            world: &mut World,
            key: ErasedMaterialPipelineKey,
            layout: &MeshVertexBufferLayoutRef,
            properties: &Arc<MaterialProperties>,
        ) -> Result<CachedRenderPipelineId, bevy_material::specialize::SpecializedMeshPipelineError>
        {
            let expected = world.resource::<Expected>();
            assert_eq!(key, expected.0);
            assert!(Arc::ptr_eq(properties, &expected.1));
            let real_layout = position_normal_mesh().get_mesh_vertex_buffer_layout(
                &mut world.resource_mut::<MeshVertexBufferLayouts>(),
            );
            assert_eq!(*layout, real_layout);
            Ok(CachedRenderPipelineId::new(2))
        }

        let mut app = App::new();
        app.insert_sub_app(RenderApp, SubApp::new());
        prepare_position_normal_pipeline_companion::<crate::StandardMaterial>(&mut app);
        let world = app.sub_app_mut(RenderApp).world_mut();
        world.init_resource::<MeshVertexBufferLayouts>();
        world.run_schedule(RenderStartup);
        let mut state = *world.resource::<PositionNormalPipelineCompanion>();
        let material_type_id = TypeId::of::<crate::StandardMaterial>();
        let mesh_key = view_key();
        let properties = Arc::new(MaterialProperties::default());
        let key = ErasedMaterialPipelineKey {
            type_id: material_type_id,
            mesh_key: ErasedMeshPipelineKey::new(mesh_key),
            material_key: ErasedMaterialKey::new(17_u32),
        };
        world.insert_resource(Expected(key.clone(), properties.clone()));
        let source_mesh = position_normal_mesh()
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0_f32; 2]; 3]);
        let mut item = SpecializationWorkItem {
            visible_entity: Entity::PLACEHOLDER.into(),
            retained_view_entity: RetainedViewEntity::new(Entity::PLACEHOLDER.into(), None, 0),
            mesh_key,
            layout: source_mesh.get_mesh_vertex_buffer_layout(
                &mut world.resource_mut::<MeshVertexBufferLayouts>(),
            ),
            properties,
            material_type_id,
        };
        assert!(state.accepts(&item));
        let source_id = CachedRenderPipelineId::new(1);
        let companion_id = CachedRenderPipelineId::new(2);
        state.request(world, specialize_fixture, key, &item, source_id);
        assert!(!state.accepts(&item));
        state.observe(world, &item, companion_id);
        assert!(state.state == CompanionState::Requested(companion_id));
        item.layout = position_normal_mesh()
            .get_mesh_vertex_buffer_layout(&mut world.resource_mut::<MeshVertexBufferLayouts>());
        state.observe(world, &item, source_id);
        assert!(state.state == CompanionState::Requested(companion_id));
        state.observe(world, &item, companion_id);
        assert!(state.state == CompanionState::Finished);
        world.insert_resource(state);
        world.run_schedule(RenderStartup);
        assert!(
            world.resource::<PositionNormalPipelineCompanion>().state == CompanionState::Waiting
        );
    }
}
