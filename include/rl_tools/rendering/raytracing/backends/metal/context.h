#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_METAL_CONTEXT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_METAL_CONTEXT_H

#include "../../types.h"

// Some third-party headers (e.g. tensorboard_logger) inject `using tensorflow::Event;` at global
// scope; without these forward declarations metal-cpp's `class Event*` member declarations would
// bind to that type instead of declaring the MTL classes.
namespace MTL{
    class Event;
    class SharedEvent;
}

#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Metal/Metal.hpp>

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::metal{
    // Buffer binding indices shared between the host encoders and the MSL kernels in device.metal
    namespace bindings{
        constexpr int LAUNCH_PARAMS = 0;
        constexpr int CAMERAS_CLOSE = 1;
        constexpr int CAMERAS_OPEN = 2;
        constexpr int OUTPUT = 3;
        constexpr int MESH_RECORDS = 4;
        constexpr int SCENE_LIGHTS = 5;
        constexpr int PROBE_DIRECTIONS = 6;
        constexpr int COLLISION_RESULTS = 7;
        constexpr int ACCELERATION_STRUCTURE = 8;
        constexpr int INSTANCE_RECORD_BASE = 9;
        constexpr int INSTANCE_DATA = 10;
        constexpr int OVERLAY_STRUCTURES = 11;
        constexpr int OVERLAY_ATTACHMENTS = 12;
        constexpr int INSTANCE_CLASSES = 13;
        constexpr int OBSERVATION = 14;
        constexpr int RGB_ACCUMULATOR = 15;
        constexpr int DEPTH_ACCUMULATOR = 16;
        constexpr int SHUTTER = 17;
        constexpr int DEPTH_OUTPUT = 18;
    }
    namespace function_constants{
        constexpr int SRGB_OUTPUT = 0;
        constexpr int MOTION_BLUR = 1;
        constexpr int MOTION_SAMPLES = 2;
        constexpr int AA_GRID = 3;
        constexpr int CHECKER_BACKGROUND = 4;
        constexpr int LOAD_TEXTURES = 5;
        constexpr int NORMAL_SHADING = 6;
        constexpr int METALLIC_REFLECTIONS = 7;
        constexpr int PBR_SHADING = 8;
        constexpr int PUNCTUAL_LIGHT_SHADOWS = 9;
        constexpr int OVERLAY_COUNT = 10;
        constexpr int SEMANTIC_SEGMENTATION = 11;
        constexpr int HAS_OBSERVATION = 12;
        constexpr int DYNAMIC_MOTION_BLUR = 13;
        constexpr int RESOLVE_RGB = 14;
        constexpr int RESOLVE_DEPTH = 15;
    }

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
        float padding[3];
    };
    static_assert(sizeof(LaunchParams) == 88, "LaunchParams layout must match the MSL declaration in device.metal");

    struct MeshRecord{
        uint64_t index;
        uint64_t vertices;
        uint64_t tex_coord;
        uint64_t normal;
        MTL::ResourceID texture;
        MTL::ResourceID normal_map;
        MTL::ResourceID metallic_roughness_map;
        MTL::ResourceID emissive_map;
        MTL::ResourceID occlusion_map;
        float color[3];
        float metallic;
        float roughness;
        float opacity;
        float emissive[3];
        float alpha_cutoff;
        int32_t alpha_mode;
        int32_t has_texture;
        int32_t has_normal_map;
        int32_t has_metallic_roughness_map;
        int32_t has_emissive_map;
        int32_t has_occlusion_map;
        int32_t padding[2];
    };
    static_assert(sizeof(MTL::ResourceID) == 8, "MTL::ResourceID expected to be 8 bytes");
    static_assert(sizeof(MeshRecord) == 144, "MeshRecord layout must match the MSL declaration in device.metal");

    struct InstanceData{
        float object_to_world[12]; // 3x4 row-major [R|t]
        float world_to_object[12];
        int32_t identity;
        int32_t padding[3];
    };
    static_assert(sizeof(InstanceData) == 112, "InstanceData layout must match the MSL declaration in device.metal");

    struct OverlayStructureEntry{
        MTL::ResourceID structure;
        uint32_t num_active;
        uint32_t padding;
    };
    static_assert(sizeof(OverlayStructureEntry) == 16, "OverlayStructureEntry layout must match the MSL declaration in device.metal");

    struct Context{
        NS::SharedPtr<MTL::Device> device;
        NS::SharedPtr<MTL::CommandQueue> queue;
        NS::SharedPtr<MTL::Library> library;
        NS::SharedPtr<MTL::ComputePipelineState> rgb_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> depth_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> collision_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> segmentation_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> resolve_pipeline;
        NS::SharedPtr<MTL::AccelerationStructure> acceleration_structure; // instance (top-level) AS
        std::vector<NS::SharedPtr<MTL::AccelerationStructure>> object_acceleration_structures;
        NS::SharedPtr<MTL::Buffer> instance_descriptors;
        NS::SharedPtr<MTL::Buffer> instance_record_base;
        NS::SharedPtr<MTL::Buffer> instance_data;
        std::vector<NS::SharedPtr<MTL::AccelerationStructure>> overlay_acceleration_structures;
        std::vector<NS::SharedPtr<MTL::Buffer>> overlay_instance_descriptors;
        std::vector<NS::SharedPtr<MTL::Buffer>> overlay_sample_instance_descriptors; // dynamic motion blur: [sample * NUM_OVERLAYS + overlay], MAX_OVERLAY_INSTANCES descriptors each
        std::vector<NS::SharedPtr<MTL::Buffer>> overlay_scratch_buffers;
        NS::SharedPtr<MTL::Buffer> overlay_structures;
        NS::SharedPtr<MTL::Buffer> overlay_attachments;
        NS::SharedPtr<MTL::Buffer> instance_classes;
        std::vector<uint32_t> object_record_base;
        std::vector<uint32_t> object_classes;
        uint32_t num_scene_instances = 0;
        std::vector<NS::SharedPtr<MTL::Buffer>> mesh_buffers;
        std::vector<NS::SharedPtr<MTL::Texture>> mesh_textures;
        NS::SharedPtr<MTL::Texture> dummy_texture;
        NS::SharedPtr<MTL::Buffer> mesh_records;
        NS::SharedPtr<MTL::Buffer> scene_lights;
        NS::SharedPtr<MTL::Buffer> cameras;
        NS::SharedPtr<MTL::Buffer> cameras_open;
        NS::SharedPtr<MTL::Buffer> frame_buffer;
        NS::SharedPtr<MTL::Buffer> depth_buffer;
        NS::SharedPtr<MTL::Buffer> rgb_accumulator;
        NS::SharedPtr<MTL::Buffer> depth_accumulator;
        NS::SharedPtr<MTL::Buffer> segmentation_buffer;
        NS::SharedPtr<MTL::Buffer> observation;
        NS::SharedPtr<MTL::Buffer> collision_results;
        NS::SharedPtr<MTL::Buffer> probe_directions;
        NS::SharedPtr<MTL::Buffer> launch_params;
        NS::SharedPtr<MTL::CommandBuffer> in_flight;
        NS::SharedPtr<MTL::CommandBuffer> in_flight_collision;
        NS::SharedPtr<MTL::CommandBuffer> in_flight_update;
        bool pipelines_built = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
