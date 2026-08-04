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

    struct Context{
        NS::SharedPtr<MTL::Device> device;
        NS::SharedPtr<MTL::CommandQueue> queue;
        NS::SharedPtr<MTL::Library> library;
        NS::SharedPtr<MTL::ComputePipelineState> rgb_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> depth_pipeline;
        NS::SharedPtr<MTL::ComputePipelineState> collision_pipeline;
        NS::SharedPtr<MTL::AccelerationStructure> acceleration_structure;
        std::vector<NS::SharedPtr<MTL::Buffer>> mesh_buffers;
        std::vector<NS::SharedPtr<MTL::Texture>> mesh_textures;
        NS::SharedPtr<MTL::Texture> dummy_texture;
        NS::SharedPtr<MTL::Buffer> mesh_records;
        NS::SharedPtr<MTL::Buffer> scene_lights;
        NS::SharedPtr<MTL::Buffer> cameras;
        NS::SharedPtr<MTL::Buffer> cameras_open;
        NS::SharedPtr<MTL::Buffer> frame_buffer;
        NS::SharedPtr<MTL::Buffer> depth_buffer;
        NS::SharedPtr<MTL::Buffer> collision_results;
        NS::SharedPtr<MTL::Buffer> probe_directions;
        NS::SharedPtr<MTL::Buffer> launch_params;
        NS::SharedPtr<MTL::CommandBuffer> in_flight;
        NS::SharedPtr<MTL::CommandBuffer> in_flight_collision;
        bool pipelines_built = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
