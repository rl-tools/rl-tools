#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_CONTEXT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_CONTEXT_H

#include "../../types.h"
#include "../generic/operations_generic.h"

#include <webgpu/webgpu.h>

#include <vector>
#include <cstdint>

#ifndef RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::webgpu{
    // Bind group 0 layout shared between the host code and the WGSL kernels in device.wgsl.
    // Standard WebGPU has no acceleration structures and no buffer device addresses, so relative
    // to the Vulkan backend the TLAS/overlay-TLAS bindings become BVH storage buffers and the
    // MeshRecord addresses become element offsets into one packed SCENE_GEOMETRY buffer.
    // Exactly 8 storage buffers: browsers tier maxStorageBuffersPerShaderStage down to the spec
    // default of 8 regardless of hardware, so the small per-frame inputs live as sections of the
    // FRAME_INPUTS arena and every output is a section of the OUTPUTS arena, both addressed by
    // element offsets carried in LaunchParams (the SCENE_GEOMETRY pattern). Mesh records live in
    // SCENE_GEOMETRY (shading-only after the packed-triangle intersection path) so the
    // leaf-ordered TRIANGLES stream keeps its own vec4-typed binding. LaunchParams itself is
    // read-only storage, not uniform — see the binding comment in device.wgsl.
    namespace bindings{
        constexpr uint32_t LAUNCH_PARAMS = 0;
        constexpr uint32_t DISPATCH_PARAMS = 1; // dynamic-offset uniform: {shutter_t, overlay_region}
        constexpr uint32_t SCENE_GEOMETRY = 2;  // u32-addressed: mesh data, triangle tables, TLAS leaf permutation, object records, sRGB LUT, mesh records, scene lights, instance classes
        constexpr uint32_t TEXTURE_DATA = 3;    // packed RGBA8 texels, one u32 each, sampled in the shader
        constexpr uint32_t BVH_NODES = 4;       // per-object BLAS slices, the scene TLAS, then the per-region overlay TLASes
        constexpr uint32_t TRIANGLES = 5;       // leaf-ordered packed triangles: 3 x vec4 {vertex.xyz, w: bitcast global id / 0 / 0}
        constexpr uint32_t INSTANCE_DATA = 6;
        constexpr uint32_t FRAME_INPUTS = 7;    // cameras, overlay attachments/meta/primitives, flow deltas, probe directions
        constexpr uint32_t OUTPUTS = 8;         // frame buffer, depth, segmentation, normals, flow, observation, accumulators, collision results
        constexpr uint32_t COUNT = 9;
        constexpr uint32_t STORAGE_COUNT = 8;
    }
    // must match the @workgroup_size of every entry point in device.wgsl
    constexpr uint32_t WORKGROUP_SIZE_X = 8;
    constexpr uint32_t WORKGROUP_SIZE_Y = 4;
    // must match TRAVERSAL_STACK_SIZE in device.wgsl; the ordered traversal pushes at most one
    // entry per tree level, so the host asserts every built tree's depth against this bound
    constexpr uint32_t TRAVERSAL_STACK_SIZE = 48;
    // one 256-byte slot per dynamic-motion-blur sample (+ slot 0 for the shutter-close state);
    // 256 is a multiple of every legal minUniformBufferOffsetAlignment
    constexpr uint32_t DISPATCH_PARAMS_STRIDE = 256;
    constexpr uint32_t ABSENT = 0xFFFFFFFFu;

    using BVHNode = generic::BVHNode<float, uint32_t>;
    static_assert(sizeof(BVHNode) == 32, "BVHNode layout must match the WGSL declaration in device.wgsl");
    static_assert(sizeof(rendering::raytracing::SceneLight) == 60, "SceneLight layout must match load_scene_light in device.wgsl");
    static_assert(sizeof(rendering::raytracing::Camera<float>) == 48, "Camera layout must match load_camera in device.wgsl");
    static_assert(sizeof(rendering::raytracing::CollisionResult) == 8, "CollisionResult must span two OUTPUTS words");

    struct LaunchParams{
        uint32_t fb_width;
        uint32_t fb_height;
        uint32_t cam_width;
        uint32_t cam_height;
        uint32_t grid_cols;
        uint32_t num_cameras;
        uint32_t num_probes;
        uint32_t num_scene_lights;
        float max_depth;
        float max_dist;
        float ambient_color[3];
        float miss_color_0[3];
        float miss_color_1[3];
        uint32_t first_overlay_instance; // global instance ids >= this index the flow-delta table
        uint32_t triangle_mesh_offset;   // u32 elements into SCENE_GEOMETRY
        uint32_t triangle_local_offset;
        uint32_t object_records_offset;  // stride 4 per object: {node_offset, node_count, first_triangle, 0}
        uint32_t tlas_node_offset;       // BVHNode elements into BVH_NODES
        uint32_t tlas_node_count;
        uint32_t tlas_primitive_offset;  // u32 elements into SCENE_GEOMETRY
        uint32_t srgb_lut_offset;        // 256 f32 elements into SCENE_GEOMETRY: sRGB texel -> linear
        uint32_t mesh_records_offset;    // 30 words per record, into SCENE_GEOMETRY
        uint32_t scene_lights_offset;    // 15 words per light, into SCENE_GEOMETRY
        uint32_t instance_classes_offset;
        uint32_t overlay_node_offset;    // BVHNode elements into BVH_NODES
        uint32_t attachments_offset;     // u32 elements into FRAME_INPUTS
        uint32_t overlay_meta_offset;    // per region x overlay: {num_active, num_tlas_nodes}
        uint32_t overlay_primitives_offset;
        uint32_t flow_deltas_offset;
        uint32_t probe_directions_offset;
        uint32_t cameras_offset;         // shutter-close cameras, then the shutter-open set for camera-pair specs
        uint32_t out_frame_buffer;       // u32 elements into OUTPUTS
        uint32_t out_depth;
        uint32_t out_segmentation;
        uint32_t out_normals;
        uint32_t out_flow;
        uint32_t out_observation;
        uint32_t out_rgb_accumulator;
        uint32_t out_depth_accumulator;
        uint32_t out_collision;
    };
    static_assert(sizeof(LaunchParams) == 184, "LaunchParams layout must match the WGSL declaration in device.wgsl");

    struct TextureRef{
        uint32_t offset; // texel index into TEXTURE_DATA, ABSENT = no texture
        uint32_t width;
        uint32_t height;
    };
    static_assert(sizeof(TextureRef) == 12, "TextureRef layout must match the WGSL declaration in device.wgsl");

    struct MeshRecord{
        uint32_t index_offset;     // u32 elements into SCENE_GEOMETRY (i32 data)
        uint32_t vertex_offset;    // f32 data
        uint32_t tex_coord_offset; // ABSENT = no tex coords
        uint32_t normal_offset;    // ABSENT = no normals
        TextureRef texture;
        TextureRef normal_map;
        TextureRef metallic_roughness_map;
        TextureRef emissive_map;
        TextureRef occlusion_map;
        float color[3];
        float metallic;
        float roughness;
        float opacity;
        float emissive[3];
        float alpha_cutoff;
        int32_t alpha_mode;
    };
    static_assert(sizeof(MeshRecord) == 120, "MeshRecord layout must match load_mesh_record in device.wgsl");

    struct InstanceData{
        float object_to_world[12]; // 3x4 row-major [R|t]
        float world_to_object[12];
        uint32_t object;
        uint32_t identity;
        uint32_t padding[2];
    };
    static_assert(sizeof(InstanceData) == 112, "InstanceData layout must match the WGSL declaration in device.wgsl");

    struct DispatchParams{
        float shutter_t;
        uint32_t overlay_region; // 0 = shutter-close state, 1 + s = dynamic-motion-blur sample s
        uint32_t padding[2];
    };
    static_assert(sizeof(DispatchParams) == 16, "DispatchParams layout must match the WGSL declaration in device.wgsl");

    // section offsets (u32 elements) of the SPEC-sized FRAME_INPUTS and OUTPUTS arenas: the
    // single source for buffer sizing, the LaunchParams offset fields, and the host's
    // queueWriteBuffer/clear/copy offsets
    template <typename SPEC>
    struct BufferLayout{
        static constexpr uint32_t OVERLAY_REGIONS = SPEC::ENABLE_OVERLAYS ? (SPEC::ENABLE_DYNAMIC_MOTION_BLUR ? 1 + (uint32_t)SPEC::MOTION_BLUR_SAMPLES : 1) : 0;
        static constexpr uint32_t ATTACHMENTS_WORDS = SPEC::ENABLE_OVERLAYS ? (uint32_t)SPEC::NUM_CAMERAS * (uint32_t)SPEC::MAX_OVERLAYS_PER_CAMERA : 0;
        static constexpr uint32_t OVERLAY_META_WORDS = OVERLAY_REGIONS * (uint32_t)SPEC::NUM_OVERLAYS * 2;
        static constexpr uint32_t OVERLAY_PRIMITIVES_WORDS = OVERLAY_REGIONS * (uint32_t)SPEC::NUM_OVERLAYS * (uint32_t)SPEC::MAX_OVERLAY_INSTANCES;
        static constexpr uint32_t FLOW_DELTAS_WORDS = (SPEC::HAS_FLOW && SPEC::ENABLE_OVERLAYS) ? (uint32_t)SPEC::NUM_OVERLAYS * (uint32_t)SPEC::MAX_OVERLAY_INSTANCES * 12 : 0;
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        static constexpr uint32_t PROBE_DIRECTIONS_WORDS = (uint32_t)SPEC::NUM_PROBES * 3;
        static constexpr uint32_t COLLISION_WORDS = (uint32_t)SPEC::NUM_CAMERAS * (uint32_t)SPEC::NUM_PROBES * 2;
#else
        static constexpr uint32_t PROBE_DIRECTIONS_WORDS = 0;
        static constexpr uint32_t COLLISION_WORDS = 0;
#endif
        static constexpr uint32_t CAMERAS_WORDS = (SPEC::HAS_CAMERA_PAIR ? 2 : 1) * (uint32_t)SPEC::NUM_CAMERAS * 12;
        static constexpr uint32_t CAMERAS_OFFSET = 0;
        static constexpr uint32_t ATTACHMENTS_OFFSET = CAMERAS_OFFSET + CAMERAS_WORDS;
        static constexpr uint32_t OVERLAY_META_OFFSET = ATTACHMENTS_OFFSET + ATTACHMENTS_WORDS;
        static constexpr uint32_t OVERLAY_PRIMITIVES_OFFSET = OVERLAY_META_OFFSET + OVERLAY_META_WORDS;
        static constexpr uint32_t FLOW_DELTAS_OFFSET = OVERLAY_PRIMITIVES_OFFSET + OVERLAY_PRIMITIVES_WORDS;
        static constexpr uint32_t PROBE_DIRECTIONS_OFFSET = FLOW_DELTAS_OFFSET + FLOW_DELTAS_WORDS;
        static constexpr uint32_t FRAME_INPUTS_WORDS = PROBE_DIRECTIONS_OFFSET + PROBE_DIRECTIONS_WORDS;

        static constexpr uint32_t CAM_PIXEL_WORDS = (uint32_t)SPEC::NUM_CAMERAS * (uint32_t)SPEC::CAM_PIXELS;
        static constexpr uint32_t FRAME_BUFFER_WORDS = SPEC::HAS_RGB ? CAM_PIXEL_WORDS : 0;
        static constexpr uint32_t DEPTH_WORDS = SPEC::HAS_DEPTH ? CAM_PIXEL_WORDS : 0;
        static constexpr uint32_t SEGMENTATION_WORDS = SPEC::HAS_SEGMENTATION ? CAM_PIXEL_WORDS : 0;
        static constexpr uint32_t NORMALS_WORDS = SPEC::HAS_NORMALS ? CAM_PIXEL_WORDS * 3 : 0;
        static constexpr uint32_t FLOW_WORDS = SPEC::HAS_FLOW ? CAM_PIXEL_WORDS * 2 : 0;
        static constexpr uint32_t OBSERVATION_WORDS = SPEC::HAS_OBSERVATION ? CAM_PIXEL_WORDS * (uint32_t)SPEC::OBSERVATION_CHANNELS : 0;
        static constexpr uint32_t RGB_ACCUMULATOR_WORDS = (SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_RGB) ? CAM_PIXEL_WORDS * 3 : 0;
        static constexpr uint32_t DEPTH_ACCUMULATOR_WORDS = (SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_DEPTH) ? CAM_PIXEL_WORDS : 0;
        static constexpr uint32_t FRAME_BUFFER_OFFSET = 0;
        static constexpr uint32_t DEPTH_OFFSET = FRAME_BUFFER_OFFSET + FRAME_BUFFER_WORDS;
        static constexpr uint32_t SEGMENTATION_OFFSET = DEPTH_OFFSET + DEPTH_WORDS;
        static constexpr uint32_t NORMALS_OFFSET = SEGMENTATION_OFFSET + SEGMENTATION_WORDS;
        static constexpr uint32_t FLOW_OFFSET = NORMALS_OFFSET + NORMALS_WORDS;
        static constexpr uint32_t OBSERVATION_OFFSET = FLOW_OFFSET + FLOW_WORDS;
        static constexpr uint32_t RGB_ACCUMULATOR_OFFSET = OBSERVATION_OFFSET + OBSERVATION_WORDS;
        static constexpr uint32_t DEPTH_ACCUMULATOR_OFFSET = RGB_ACCUMULATOR_OFFSET + RGB_ACCUMULATOR_WORDS;
        static constexpr uint32_t COLLISION_OFFSET = DEPTH_ACCUMULATOR_OFFSET + DEPTH_ACCUMULATOR_WORDS;
        static constexpr uint32_t OUTPUTS_WORDS = COLLISION_OFFSET + COLLISION_WORDS;
    };

    struct BufferResource{
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
    };

    // staging -> host-tensor readback registered in malloc; lets the in-flight settle run against
    // the Context alone (the renderer memory-domain copy has no SPEC)
    struct ReadbackTarget{
        BufferResource* staging = nullptr;
        void* destination = nullptr;
        size_t bytes = 0;
    };

    struct Context{
        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        uint32_t max_storage_buffers_per_shader_stage = 0;
        uint64_t max_storage_buffer_binding_size = 0;

        WGPUShaderModule module = nullptr;
        WGPUBindGroupLayout bind_group_layout = nullptr;
        WGPUPipelineLayout pipeline_layout = nullptr;
        WGPUComputePipeline rgb_pipeline = nullptr;
        WGPUComputePipeline depth_pipeline = nullptr;
        WGPUComputePipeline collision_pipeline = nullptr;
        WGPUComputePipeline segmentation_pipeline = nullptr;
        WGPUComputePipeline normals_pipeline = nullptr;
        WGPUComputePipeline flow_pipeline = nullptr;
        WGPUComputePipeline resolve_pipeline = nullptr;
        WGPUBindGroup bind_group = nullptr; // recreated per init (scene buffers change)

        BufferResource launch_params;
        BufferResource scene_geometry;
        BufferResource texture_data;
        BufferResource bvh_nodes;
        BufferResource triangles;
        BufferResource instance_data;
        BufferResource frame_inputs;
        BufferResource outputs;
        BufferResource dispatch_params;

        BufferResource staging_frame_buffer;
        BufferResource staging_depth;
        BufferResource staging_segmentation;
        BufferResource staging_normals;
        BufferResource staging_flow;
        BufferResource staging_observation;
        BufferResource staging_collision;
        std::vector<ReadbackTarget> render_readbacks;
        std::vector<ReadbackTarget> probe_readbacks;

        // scene-dependent arena offsets mirrored from LaunchParams for the per-update writes
        uint32_t instance_classes_offset = 0;
        uint32_t overlay_node_offset = 0;

        // host mirrors consumed by the per-update overlay rebuilds (queueWriteBuffer sources)
        std::vector<InstanceData> instance_data_host;
        std::vector<uint32_t> instance_classes_host;
        std::vector<uint32_t> object_classes;        // object -> segmentation_class
        std::vector<uint32_t> object_node_offset;    // object -> BVHNode elements into BVH_NODES
        std::vector<uint32_t> object_node_count;
        std::vector<uint32_t> object_primitive_offset;
        std::vector<float> object_root_bounds_min;   // 3 per object, BLAS root AABB for instance world bounds
        std::vector<float> object_root_bounds_max;
        std::vector<BVHNode> overlay_nodes_host;     // all regions, one 2*capacity slice per (region, overlay)
        std::vector<uint32_t> overlay_primitives_host;
        std::vector<uint32_t> overlay_meta_host;
        std::vector<float> overlay_bounds_min;       // scratch indexed by global instance id
        std::vector<float> overlay_bounds_max;
        std::vector<float> overlay_centroids;
        std::vector<uint32_t> overlay_temp_primitives;
        uint32_t num_scene_instances = 0;
        uint32_t total_instances = 0;

        bool render_in_flight = false;
        bool collision_in_flight = false;
        bool pipelines_built = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
