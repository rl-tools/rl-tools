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
        constexpr uint32_t SEGMENTATION_BUFFER = 10;
        constexpr uint32_t INSTANCE_RECORD_BASE = 11;
        constexpr uint32_t INSTANCE_DATA = 12;
        constexpr uint32_t OVERLAY_ATTACHMENTS = 13;
        constexpr uint32_t OVERLAY_NUM_ACTIVE = 14;
        constexpr uint32_t OVERLAY_TLAS = 15;
        constexpr uint32_t INSTANCE_CLASSES = 16;
        constexpr uint32_t OBSERVATION = 17;
        constexpr uint32_t RGB_ACCUMULATOR = 18;
        constexpr uint32_t DEPTH_ACCUMULATOR = 19;
        constexpr uint32_t TEXTURES = 20; // variable-descriptor-count binding must have the largest binding number in the set
        constexpr uint32_t COUNT = 21;
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
        constexpr uint32_t OVERLAY_COUNT = 10;
        constexpr uint32_t SEMANTIC_SEGMENTATION = 11;
        constexpr uint32_t HAS_OBSERVATION = 12;
        constexpr uint32_t DYNAMIC_MOTION_BLUR = 13;
        constexpr uint32_t RESOLVE_RGB = 14;
        constexpr uint32_t RESOLVE_DEPTH = 15;
        constexpr uint32_t COUNT = 16;
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

    struct InstanceData{
        float object_to_world[12]; // 3x4 row-major [R|t]
        float world_to_object[12];
        uint32_t identity;
        uint32_t padding[3];
    };
    static_assert(sizeof(InstanceData) == 112, "InstanceData layout must match the GLSL std430 declaration in device.comp");

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
        VkShaderModule module_segmentation = VK_NULL_HANDLE;
        VkShaderModule module_resolve = VK_NULL_HANDLE;
        VkPipeline rgb_pipeline = VK_NULL_HANDLE;
        VkPipeline depth_pipeline = VK_NULL_HANDLE;
        VkPipeline collision_pipeline = VK_NULL_HANDLE;
        VkPipeline segmentation_pipeline = VK_NULL_HANDLE;
        VkPipeline resolve_pipeline = VK_NULL_HANDLE;
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
        BufferResource tlas_buffer;
        BufferResource instance_buffer;
        BufferResource segmentation_buffer;
        BufferResource observation;
        BufferResource instance_data;
        BufferResource instance_record_base;
        BufferResource instance_classes;
        BufferResource overlay_attachments;
        BufferResource overlay_num_active;
        BufferResource overlay_scratch;
        BufferResource rgb_accumulator;
        BufferResource depth_accumulator;
        BufferResource dummy;

        std::vector<ImageResource> mesh_textures; // index 0 = dummy 1x1 white
        std::vector<VkAccelerationStructureKHR> blas_list; // one per object
        std::vector<BufferResource> blas_buffers;
        std::vector<VkDeviceAddress> blas_addresses;
        std::vector<uint32_t> object_record_base; // object -> first index into mesh_records
        std::vector<uint32_t> object_classes; // object -> segmentation_class, consumed by update() for overlay slots
        uint32_t num_scene_instances = 0;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        std::vector<VkAccelerationStructureKHR> overlay_tlas; // one per overlay, rebuilt in place by update()
        std::vector<BufferResource> overlay_tlas_buffers;
        std::vector<BufferResource> overlay_instance_buffers;
        std::vector<BufferResource> overlay_sample_instance_buffers; // dynamic motion blur: one per motion sample, NUM_OVERLAYS * MAX_OVERLAY_INSTANCES descriptors each
        VkDeviceSize overlay_scratch_stride = 0;

        VkCommandBuffer cb_rgb = VK_NULL_HANDLE;
        VkCommandBuffer cb_depth = VK_NULL_HANDLE;
        VkCommandBuffer cb_collision = VK_NULL_HANDLE;
        VkCommandBuffer cb_segmentation = VK_NULL_HANDLE;
        VkCommandBuffer cb_dynamic = VK_NULL_HANDLE; // dynamic motion blur frame, re-recorded per render_launch
        VkFence fence_render = VK_NULL_HANDLE;
        VkFence fence_collision = VK_NULL_HANDLE;
        VkFence fence_update = VK_NULL_HANDLE;
        VkCommandBuffer update_command_buffer = VK_NULL_HANDLE;
        bool render_in_flight = false;
        bool collision_in_flight = false;
        bool pipelines_built = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
