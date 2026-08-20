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

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::webgpu{
    // Bind group 0 layout shared between the host code and the WGSL kernels in device.wgsl.
    // Standard WebGPU has no acceleration structures and no buffer device addresses, so relative
    // to the Vulkan backend the TLAS/overlay-TLAS bindings become BVH storage buffers and the
    // MeshRecord addresses become element offsets into one packed SCENE_GEOMETRY buffer.
    namespace bindings{
        constexpr uint32_t LAUNCH_PARAMS = 0;
        constexpr uint32_t CAMERAS_CLOSE = 1;
        constexpr uint32_t CAMERAS_OPEN = 2;
        constexpr uint32_t FRAME_BUFFER = 3;
        constexpr uint32_t MESH_RECORDS = 4;
        constexpr uint32_t SCENE_LIGHTS = 5;
        constexpr uint32_t PROBE_DIRECTIONS = 6;
        constexpr uint32_t COLLISION_RESULTS = 7;
        constexpr uint32_t BVH_NODES = 8; // per-object BLAS slices followed by the scene TLAS
        constexpr uint32_t DEPTH_BUFFER = 9;
        constexpr uint32_t SEGMENTATION_BUFFER = 10;
        constexpr uint32_t SCENE_GEOMETRY = 11; // u32-addressed: mesh data, triangle tables, BLAS/TLAS leaf permutations, object records
        constexpr uint32_t INSTANCE_DATA = 12;
        constexpr uint32_t OVERLAY_ATTACHMENTS = 13;
        constexpr uint32_t OVERLAY_META = 14; // per region x overlay: {num_active, num_tlas_nodes}
        constexpr uint32_t OVERLAY_NODES = 15;
        constexpr uint32_t INSTANCE_CLASSES = 16;
        constexpr uint32_t OBSERVATION = 17;
        constexpr uint32_t RGB_ACCUMULATOR = 18;
        constexpr uint32_t DEPTH_ACCUMULATOR = 19;
        constexpr uint32_t NORMALS_BUFFER = 20;
        constexpr uint32_t FLOW_BUFFER = 21;
        constexpr uint32_t FLOW_DELTAS = 22;
        constexpr uint32_t TEXTURE_DATA = 23; // packed RGBA8 texels, one u32 each, sampled in the shader
        constexpr uint32_t OVERLAY_PRIMITIVES = 24;
        constexpr uint32_t DISPATCH_PARAMS = 25; // dynamic-offset uniform: {shutter_t, overlay_region}
        constexpr uint32_t TRIANGLES = 26; // leaf-ordered packed triangles: 3 x vec4 {vertex.xyz, w: bitcast global id / 0 / 0}
        constexpr uint32_t COUNT = 27;
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
    };
    static_assert(sizeof(LaunchParams) == 108, "LaunchParams layout must match the WGSL declaration in device.wgsl");

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
    static_assert(sizeof(MeshRecord) == 120, "MeshRecord layout must match the WGSL declaration in device.wgsl");

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

    struct BufferResource{
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
    };

    struct Context{
        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        uint32_t max_storage_buffers_per_shader_stage = 0;

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
        BufferResource cameras;
        BufferResource cameras_open;
        BufferResource frame_buffer;
        BufferResource depth_buffer;
        BufferResource collision_results;
        BufferResource probe_directions;
        BufferResource mesh_records;
        BufferResource scene_lights;
        BufferResource scene_geometry;
        BufferResource texture_data;
        BufferResource bvh_nodes;
        BufferResource triangles;
        BufferResource segmentation_buffer;
        BufferResource normals_buffer;
        BufferResource flow_buffer;
        BufferResource flow_deltas;
        BufferResource observation;
        BufferResource instance_data;
        BufferResource instance_classes;
        BufferResource overlay_attachments;
        BufferResource overlay_meta;
        BufferResource overlay_nodes;
        BufferResource overlay_primitives;
        BufferResource rgb_accumulator;
        BufferResource depth_accumulator;
        BufferResource dispatch_params;

        BufferResource staging_frame_buffer;
        BufferResource staging_depth;
        BufferResource staging_segmentation;
        BufferResource staging_normals;
        BufferResource staging_flow;
        BufferResource staging_observation;
        BufferResource staging_collision;

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
