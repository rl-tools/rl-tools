#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_VULKAN_CONTEXT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_VULKAN_CONTEXT_H

#include "../../types.h"

#include <vulkan/vulkan.h>

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::vulkan{
    // Descriptor bindings shared between the host code and the GLSL kernels in device.comp
    namespace bindings{
        constexpr uint32_t LAUNCH_PARAMS = 0;
        constexpr uint32_t CAMERAS_CLOSE = 1;
        constexpr uint32_t CAMERAS_OPEN = 2;
        constexpr uint32_t FRAME_BUFFER = 3;
        constexpr uint32_t MESH_RECORDS = 4;
        constexpr uint32_t SCENE_LIGHTS = 5;
        constexpr uint32_t PROBE_DIRECTIONS = 6;
        constexpr uint32_t COLLISION_RESULTS = 7;
        constexpr uint32_t ACCELERATION_STRUCTURE = 8;
        constexpr uint32_t DEPTH_BUFFER = 9;
        constexpr uint32_t TEXTURES = 10;
        constexpr uint32_t COUNT = 11;
    }
    namespace specialization_constants{
        constexpr uint32_t SRGB_OUTPUT = 0;
        constexpr uint32_t MOTION_BLUR = 1;
        constexpr uint32_t MOTION_SAMPLES = 2;
        constexpr uint32_t AA_GRID = 3;
        constexpr uint32_t CHECKER_BACKGROUND = 4;
        constexpr uint32_t LOAD_TEXTURES = 5;
        constexpr uint32_t NORMAL_SHADING = 6;
        constexpr uint32_t METALLIC_REFLECTIONS = 7;
        constexpr uint32_t PBR_SHADING = 8;
        constexpr uint32_t PUNCTUAL_LIGHT_SHADOWS = 9;
        constexpr uint32_t COUNT = 10;
    }
    constexpr uint32_t WORKGROUP_SIZE = 8;
    constexpr uint32_t MAX_TEXTURE_DESCRIPTORS = 4096;

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
    static_assert(sizeof(LaunchParams) == 88, "LaunchParams layout must match the GLSL declaration in device.comp");

    struct MeshRecord{
        uint64_t index;
        uint64_t vertices;
        uint64_t tex_coord;
        uint64_t normal;
        uint32_t texture;
        uint32_t normal_map;
        uint32_t metallic_roughness_map;
        uint32_t emissive_map;
        uint32_t occlusion_map;
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
        int32_t has_tex_coord;
        int32_t has_normals;
        int32_t padding[1];
    };
    static_assert(sizeof(MeshRecord) == 128, "MeshRecord layout must match the GLSL declaration in device.comp");

    struct BufferResource{
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };

    struct ImageResource{
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    struct Context{
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queue_family_index = 0;
        VkDeviceSize min_scratch_alignment = 256;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool command_pool = VK_NULL_HANDLE;

        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
        PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;

        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkShaderModule module_rgb = VK_NULL_HANDLE;
        VkShaderModule module_depth = VK_NULL_HANDLE;
        VkShaderModule module_collision = VK_NULL_HANDLE;
        VkPipeline rgb_pipeline = VK_NULL_HANDLE;
        VkPipeline depth_pipeline = VK_NULL_HANDLE;
        VkPipeline collision_pipeline = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;

        BufferResource launch_params;
        BufferResource cameras;
        BufferResource cameras_open;
        BufferResource frame_buffer;
        BufferResource depth_buffer;
        BufferResource collision_results;
        BufferResource probe_directions;
        BufferResource mesh_records;
        BufferResource scene_lights;
        BufferResource mesh_data;
        BufferResource blas_buffer;
        BufferResource tlas_buffer;
        BufferResource instance_buffer;
        BufferResource dummy;

        std::vector<ImageResource> mesh_textures; // index 0 = dummy 1x1 white
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;

        VkCommandBuffer cb_rgb = VK_NULL_HANDLE;
        VkCommandBuffer cb_depth = VK_NULL_HANDLE;
        VkCommandBuffer cb_collision = VK_NULL_HANDLE;
        VkFence fence_render = VK_NULL_HANDLE;
        VkFence fence_collision = VK_NULL_HANDLE;
        bool render_in_flight = false;
        bool collision_in_flight = false;
        bool pipelines_built = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
