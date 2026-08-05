#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_VULKAN_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_VULKAN_OPERATIONS_CPU_H

#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "context.h"
#include "device_source.h"

#include <vector>
#include <cstring>
#include <cstdlib>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::backends::vulkan{
        template <typename SPEC>
        Context& context(rendering::raytracing::Renderer<SPEC>& renderer){
            return *(Context*)renderer.backend.context;
        }

        template <typename DEVICE>
        void check(DEVICE& device, VkResult result, const char* message){
            if(result != VK_SUCCESS){
                RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR(message << " (VkResult " << (int)result << ")");
                utils::assert_exit(device, false, message);
            }
        }

        template <typename DEVICE>
        uint32_t find_memory_type(DEVICE& device, Context& ctx, uint32_t type_bits, VkMemoryPropertyFlags properties){
            VkPhysicalDeviceMemoryProperties memory_properties;
            vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &memory_properties);
            for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++){
                if((type_bits & (1u << i)) != 0 && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties){
                    return i;
                }
            }
            if((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0){
                return find_memory_type(device, ctx, type_bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            utils::assert_exit(device, false, "Vulkan: no suitable memory type");
            return 0;
        }

        template <typename DEVICE>
        BufferResource create_buffer(DEVICE& device, Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool map){
            BufferResource resource;
            resource.size = size;
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(device, vkCreateBuffer(ctx.device, &buffer_info, nullptr, &resource.buffer), "Vulkan: buffer creation failed");
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(ctx.device, resource.buffer, &requirements);
            VkMemoryAllocateFlagsInfo flags_info{};
            flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocate_info{};
            allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            if((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0){
                allocate_info.pNext = &flags_info;
            }
            allocate_info.allocationSize = requirements.size;
            allocate_info.memoryTypeIndex = find_memory_type(device, ctx, requirements.memoryTypeBits, properties);
            VkResult allocation_result = vkAllocateMemory(ctx.device, &allocate_info, nullptr, &resource.memory);
            // BAR-preferred allocations fall back to plain host memory when the host-visible VRAM heap is exhausted (non-ReBAR systems)
            if(allocation_result != VK_SUCCESS && (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0){
                allocate_info.memoryTypeIndex = find_memory_type(device, ctx, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                allocation_result = vkAllocateMemory(ctx.device, &allocate_info, nullptr, &resource.memory);
            }
            check(device, allocation_result, "Vulkan: buffer memory allocation failed");
            check(device, vkBindBufferMemory(ctx.device, resource.buffer, resource.memory, 0), "Vulkan: buffer memory bind failed");
            if(map){
                check(device, vkMapMemory(ctx.device, resource.memory, 0, VK_WHOLE_SIZE, 0, &resource.mapped), "Vulkan: buffer memory map failed");
            }
            return resource;
        }

        inline void destroy_buffer(Context& ctx, BufferResource& resource){
            if(resource.buffer != VK_NULL_HANDLE){
                vkDestroyBuffer(ctx.device, resource.buffer, nullptr);
                resource.buffer = VK_NULL_HANDLE;
            }
            if(resource.memory != VK_NULL_HANDLE){
                vkFreeMemory(ctx.device, resource.memory, nullptr);
                resource.memory = VK_NULL_HANDLE;
            }
            resource.mapped = nullptr;
            resource.size = 0;
        }

        inline VkDeviceAddress buffer_address(Context& ctx, const BufferResource& resource){
            VkBufferDeviceAddressInfo address_info{};
            address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            address_info.buffer = resource.buffer;
            return vkGetBufferDeviceAddress(ctx.device, &address_info);
        }

        template <typename DEVICE>
        VkCommandBuffer one_shot_begin(DEVICE& device, Context& ctx){
            VkCommandBufferAllocateInfo allocate_info{};
            allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocate_info.commandPool = ctx.command_pool;
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;
            VkCommandBuffer command_buffer;
            check(device, vkAllocateCommandBuffers(ctx.device, &allocate_info, &command_buffer), "Vulkan: one-shot command buffer allocation failed");
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(device, vkBeginCommandBuffer(command_buffer, &begin_info), "Vulkan: one-shot command buffer begin failed");
            return command_buffer;
        }

        template <typename DEVICE>
        void one_shot_end(DEVICE& device, Context& ctx, VkCommandBuffer command_buffer){
            check(device, vkEndCommandBuffer(command_buffer), "Vulkan: one-shot command buffer end failed");
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            check(device, vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE), "Vulkan: one-shot submit failed");
            check(device, vkQueueWaitIdle(ctx.queue), "Vulkan: one-shot wait failed");
            vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &command_buffer);
        }

        template <typename DEVICE>
        void upload_buffer(DEVICE& device, Context& ctx, BufferResource& destination, const void* data, VkDeviceSize size){
            BufferResource staging = create_buffer(device, ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
            std::memcpy(staging.mapped, data, size);
            VkCommandBuffer command_buffer = one_shot_begin(device, ctx);
            VkBufferCopy region{};
            region.size = size;
            vkCmdCopyBuffer(command_buffer, staging.buffer, destination.buffer, 1, &region);
            one_shot_end(device, ctx, command_buffer);
            destroy_buffer(ctx, staging);
        }

        template <typename DEVICE>
        ImageResource create_texture_image(DEVICE& device, Context& ctx, int width, int height, bool srgb){
            ImageResource resource;
            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            image_info.extent = {(uint32_t)width, (uint32_t)height, 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            check(device, vkCreateImage(ctx.device, &image_info, nullptr, &resource.image), "Vulkan: image creation failed");
            VkMemoryRequirements requirements;
            vkGetImageMemoryRequirements(ctx.device, resource.image, &requirements);
            VkMemoryAllocateInfo allocate_info{};
            allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocate_info.allocationSize = requirements.size;
            allocate_info.memoryTypeIndex = find_memory_type(device, ctx, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(device, vkAllocateMemory(ctx.device, &allocate_info, nullptr, &resource.memory), "Vulkan: image memory allocation failed");
            check(device, vkBindImageMemory(ctx.device, resource.image, resource.memory, 0), "Vulkan: image memory bind failed");
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = resource.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = image_info.format;
            view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(device, vkCreateImageView(ctx.device, &view_info, nullptr, &resource.view), "Vulkan: image view creation failed");
            return resource;
        }

        inline void record_texture_upload(VkCommandBuffer command_buffer, ImageResource& image, BufferResource& staging, int width, int height){
            VkImageMemoryBarrier to_transfer{};
            to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_transfer.srcAccessMask = 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = image.image;
            to_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
            vkCmdCopyBufferToImage(command_buffer, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            VkImageMemoryBarrier to_sampled = to_transfer;
            to_sampled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            to_sampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_sampled);
        }

        inline void destroy_image(Context& ctx, ImageResource& resource){
            if(resource.view != VK_NULL_HANDLE){
                vkDestroyImageView(ctx.device, resource.view, nullptr);
                resource.view = VK_NULL_HANDLE;
            }
            if(resource.image != VK_NULL_HANDLE){
                vkDestroyImage(ctx.device, resource.image, nullptr);
                resource.image = VK_NULL_HANDLE;
            }
            if(resource.memory != VK_NULL_HANDLE){
                vkFreeMemory(ctx.device, resource.memory, nullptr);
                resource.memory = VK_NULL_HANDLE;
            }
        }

        template <typename DEVICE>
        void wait_render_in_flight(DEVICE& device, Context& ctx){
            if(ctx.render_in_flight){
                check(device, vkWaitForFences(ctx.device, 1, &ctx.fence_render, VK_TRUE, UINT64_MAX), "Vulkan: render fence wait failed");
                check(device, vkResetFences(ctx.device, 1, &ctx.fence_render), "Vulkan: render fence reset failed");
                ctx.render_in_flight = false;
            }
        }

        template <typename DEVICE>
        void wait_collision_in_flight(DEVICE& device, Context& ctx){
            if(ctx.collision_in_flight){
                check(device, vkWaitForFences(ctx.device, 1, &ctx.fence_collision, VK_TRUE, UINT64_MAX), "Vulkan: collision fence wait failed");
                check(device, vkResetFences(ctx.device, 1, &ctx.fence_collision), "Vulkan: collision fence reset failed");
                ctx.collision_in_flight = false;
            }
        }

        template <typename DEVICE>
        void wait_in_flight(DEVICE& device, Context& ctx){
            wait_render_in_flight(device, ctx);
            wait_collision_in_flight(device, ctx);
        }

        inline void destroy_scene_resources(Context& ctx){
            if(ctx.descriptor_pool != VK_NULL_HANDLE){
                vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, nullptr);
                ctx.descriptor_pool = VK_NULL_HANDLE;
                ctx.descriptor_set = VK_NULL_HANDLE;
            }
            if(ctx.tlas != VK_NULL_HANDLE){
                ctx.vkDestroyAccelerationStructureKHR(ctx.device, ctx.tlas, nullptr);
                ctx.tlas = VK_NULL_HANDLE;
            }
            if(ctx.blas != VK_NULL_HANDLE){
                ctx.vkDestroyAccelerationStructureKHR(ctx.device, ctx.blas, nullptr);
                ctx.blas = VK_NULL_HANDLE;
            }
            destroy_buffer(ctx, ctx.tlas_buffer);
            destroy_buffer(ctx, ctx.blas_buffer);
            destroy_buffer(ctx, ctx.instance_buffer);
            destroy_buffer(ctx, ctx.mesh_data);
            destroy_buffer(ctx, ctx.mesh_records);
            destroy_buffer(ctx, ctx.scene_lights);
            for(auto& texture : ctx.mesh_textures){
                destroy_image(ctx, texture);
            }
            ctx.mesh_textures.clear();
        }

        template <typename DEVICE>
        void submit_render(DEVICE& device, Context& ctx, VkCommandBuffer first, VkCommandBuffer second = VK_NULL_HANDLE){
            wait_render_in_flight(device, ctx);
            VkCommandBuffer command_buffers[2] = {first, second};
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = second != VK_NULL_HANDLE ? 2 : 1;
            submit_info.pCommandBuffers = command_buffers;
            check(device, vkQueueSubmit(ctx.queue, 1, &submit_info, ctx.fence_render), "Vulkan: render submit failed");
            ctx.render_in_flight = true;
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The Vulkan raytracing backend requires T = float");

        malloc(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            malloc(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            malloc(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            malloc(device, renderer.depth_buffer);
        }
        malloc(device, renderer.collision_results);

        auto* ctx = new vk::Context{};

        VkApplicationInfo application_info{};
        application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application_info.pApplicationName = "rl_tools";
        application_info.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &application_info;
        vk::check(device, vkCreateInstance(&instance_info, nullptr, &ctx->instance), "Vulkan: instance creation failed");

        uint32_t physical_device_count = 0;
        vkEnumeratePhysicalDevices(ctx->instance, &physical_device_count, nullptr);
        utils::assert_exit(device, physical_device_count > 0, "Vulkan: no devices available");
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        vkEnumeratePhysicalDevices(ctx->instance, &physical_device_count, physical_devices.data());

        const char* required_extensions[] = {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
        };
        const char* device_index_env = std::getenv("RL_TOOLS_VULKAN_DEVICE_INDEX");
        int64_t requested_device_index = device_index_env != nullptr ? std::atoll(device_index_env) : -1;
        int64_t selected = -1;
        bool selected_discrete = false;
        for(uint32_t physical_device_i = 0; physical_device_i < physical_device_count; physical_device_i++){
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(physical_devices[physical_device_i], &properties);
            if(properties.apiVersion < VK_API_VERSION_1_2){
                continue;
            }
            uint32_t extension_count = 0;
            vkEnumerateDeviceExtensionProperties(physical_devices[physical_device_i], nullptr, &extension_count, nullptr);
            std::vector<VkExtensionProperties> extensions(extension_count);
            vkEnumerateDeviceExtensionProperties(physical_devices[physical_device_i], nullptr, &extension_count, extensions.data());
            bool extensions_present = true;
            for(const char* required : required_extensions){
                bool found = false;
                for(const auto& extension : extensions){
                    if(std::strcmp(extension.extensionName, required) == 0){
                        found = true;
                        break;
                    }
                }
                extensions_present = extensions_present && found;
            }
            if(!extensions_present){
                continue;
            }
            VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features{};
            acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
            ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            ray_query_features.pNext = &acceleration_structure_features;
            VkPhysicalDeviceVulkan12Features features_1_2{};
            features_1_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features_1_2.pNext = &ray_query_features;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &features_1_2;
            vkGetPhysicalDeviceFeatures2(physical_devices[physical_device_i], &features);
            bool features_present = acceleration_structure_features.accelerationStructure && ray_query_features.rayQuery
                && features_1_2.bufferDeviceAddress && features_1_2.runtimeDescriptorArray
                && features_1_2.shaderSampledImageArrayNonUniformIndexing
                && features_1_2.descriptorBindingPartiallyBound && features_1_2.descriptorBindingVariableDescriptorCount;
            if(!features_present){
                continue;
            }
            if(requested_device_index >= 0){
                if((int64_t)physical_device_i == requested_device_index){
                    selected = physical_device_i;
                    break;
                }
                continue;
            }
            bool discrete = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if(selected < 0 || (discrete && !selected_discrete)){
                selected = physical_device_i;
                selected_discrete = discrete;
            }
        }
        utils::assert_exit(device, selected >= 0, "Vulkan: no device with ray query support found");
        ctx->physical_device = physical_devices[selected];
        {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties{};
            acceleration_structure_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = &acceleration_structure_properties;
            vkGetPhysicalDeviceProperties2(ctx->physical_device, &properties);
            ctx->min_scratch_alignment = acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment;
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Vulkan device: " << properties.properties.deviceName);
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, queue_families.data());
        bool queue_family_found = false;
        for(uint32_t queue_family_i = 0; queue_family_i < queue_family_count; queue_family_i++){
            if((queue_families[queue_family_i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0){
                ctx->queue_family_index = queue_family_i;
                queue_family_found = true;
                break;
            }
        }
        utils::assert_exit(device, queue_family_found, "Vulkan: no compute queue family");

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = ctx->queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features{};
        acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        acceleration_structure_features.accelerationStructure = VK_TRUE;
        VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
        ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        ray_query_features.pNext = &acceleration_structure_features;
        ray_query_features.rayQuery = VK_TRUE;
        VkPhysicalDeviceVulkan12Features features_1_2{};
        features_1_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features_1_2.pNext = &ray_query_features;
        features_1_2.bufferDeviceAddress = VK_TRUE;
        features_1_2.descriptorIndexing = VK_TRUE;
        features_1_2.runtimeDescriptorArray = VK_TRUE;
        features_1_2.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        features_1_2.descriptorBindingPartiallyBound = VK_TRUE;
        features_1_2.descriptorBindingVariableDescriptorCount = VK_TRUE;
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &features_1_2;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = 3;
        device_info.ppEnabledExtensionNames = required_extensions;
        vk::check(device, vkCreateDevice(ctx->physical_device, &device_info, nullptr, &ctx->device), "Vulkan: device creation failed");
        vkGetDeviceQueue(ctx->device, ctx->queue_family_index, 0, &ctx->queue);

        ctx->vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(ctx->device, "vkCreateAccelerationStructureKHR");
        ctx->vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(ctx->device, "vkDestroyAccelerationStructureKHR");
        ctx->vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureBuildSizesKHR");
        ctx->vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(ctx->device, "vkCmdBuildAccelerationStructuresKHR");
        ctx->vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureDeviceAddressKHR");
        utils::assert_exit(device, ctx->vkCmdBuildAccelerationStructuresKHR != nullptr, "Vulkan: acceleration structure entry points unavailable");

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = ctx->queue_family_index;
        vk::check(device, vkCreateCommandPool(ctx->device, &pool_info, nullptr, &ctx->command_pool), "Vulkan: command pool creation failed");

        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = ctx->command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        vk::check(device, vkAllocateCommandBuffers(ctx->device, &command_buffer_info, &ctx->cb_rgb), "Vulkan: command buffer allocation failed");
        vk::check(device, vkAllocateCommandBuffers(ctx->device, &command_buffer_info, &ctx->cb_depth), "Vulkan: command buffer allocation failed");
        vk::check(device, vkAllocateCommandBuffers(ctx->device, &command_buffer_info, &ctx->cb_collision), "Vulkan: command buffer allocation failed");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk::check(device, vkCreateFence(ctx->device, &fence_info, nullptr, &ctx->fence_render), "Vulkan: fence creation failed");
        vk::check(device, vkCreateFence(ctx->device, &fence_info, nullptr, &ctx->fence_collision), "Vulkan: fence creation failed");

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        vk::check(device, vkCreateSampler(ctx->device, &sampler_info, nullptr, &ctx->sampler), "Vulkan: sampler creation failed");

        {
            VkDescriptorSetLayoutBinding layout_bindings[vk::bindings::COUNT]{};
            VkDescriptorBindingFlags binding_flags[vk::bindings::COUNT]{};
            for(uint32_t binding_i = 0; binding_i < vk::bindings::COUNT; binding_i++){
                layout_bindings[binding_i].binding = binding_i;
                layout_bindings[binding_i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                layout_bindings[binding_i].descriptorCount = 1;
                layout_bindings[binding_i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            layout_bindings[vk::bindings::ACCELERATION_STRUCTURE].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            layout_bindings[vk::bindings::TEXTURES].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            layout_bindings[vk::bindings::TEXTURES].descriptorCount = vk::MAX_TEXTURE_DESCRIPTORS;
            binding_flags[vk::bindings::TEXTURES] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
            binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            binding_flags_info.bindingCount = vk::bindings::COUNT;
            binding_flags_info.pBindingFlags = binding_flags;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.pNext = &binding_flags_info;
            layout_info.bindingCount = vk::bindings::COUNT;
            layout_info.pBindings = layout_bindings;
            vk::check(device, vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &ctx->descriptor_set_layout), "Vulkan: descriptor set layout creation failed");
            VkPipelineLayoutCreateInfo pipeline_layout_info{};
            pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &ctx->descriptor_set_layout;
            vk::check(device, vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, nullptr, &ctx->pipeline_layout), "Vulkan: pipeline layout creation failed");
        }

        auto make_module = [&](const uint32_t* words, size_t size_bytes) -> VkShaderModule {
            VkShaderModuleCreateInfo module_info{};
            module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            module_info.codeSize = size_bytes;
            module_info.pCode = words;
            VkShaderModule module;
            vk::check(device, vkCreateShaderModule(ctx->device, &module_info, nullptr, &module), "Vulkan: shader module creation failed");
            return module;
        };
        size_t spirv_size = 0;
        const uint32_t* spirv = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            spirv = rendering::raytracing::backends::vulkan::device_spirv_rgb(spirv_size);
            ctx->module_rgb = make_module(spirv, spirv_size);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            spirv = rendering::raytracing::backends::vulkan::device_spirv_depth(spirv_size);
            ctx->module_depth = make_module(spirv, spirv_size);
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        spirv = rendering::raytracing::backends::vulkan::device_spirv_collision(spirv_size);
        ctx->module_collision = make_module(spirv, spirv_size);
#endif

        // mapped like Metal's StorageModeShared, but preferring host-visible VRAM (BAR/ReBAR) so per-frame shader access stays at device speed
        constexpr VkMemoryPropertyFlags HOST_MEMORY = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr VkBufferUsageFlags STORAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        constexpr typename SPEC::TI cam_pixels = SPEC::CAM_PIXELS;
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        ctx->launch_params = vk::create_buffer(device, *ctx, sizeof(vk::LaunchParams), STORAGE, HOST_MEMORY, true);
        ctx->cameras = vk::create_buffer(device, *ctx, camera_bytes, STORAGE, HOST_MEMORY, true);
        renderer.backend.cameras_buffer = (void*)ctx->cameras.buffer;
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            ctx->cameras_open = vk::create_buffer(device, *ctx, camera_bytes, STORAGE, HOST_MEMORY, true);
            renderer.backend.cameras_open_buffer = (void*)ctx->cameras_open.buffer;
        }
        if constexpr (SPEC::HAS_RGB) {
            ctx->frame_buffer = vk::create_buffer(device, *ctx, (size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(uint32_t), STORAGE, HOST_MEMORY, true);
            renderer.backend.frame_buffer_handle = (void*)ctx->frame_buffer.buffer;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            ctx->depth_buffer = vk::create_buffer(device, *ctx, (size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(float), STORAGE, HOST_MEMORY, true);
            renderer.backend.depth_buffer_handle = (void*)ctx->depth_buffer.buffer;
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        ctx->collision_results = vk::create_buffer(device, *ctx, (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult), STORAGE, HOST_MEMORY, true);
        renderer.backend.collision_results_buffer = (void*)ctx->collision_results.buffer;
        ctx->probe_directions = vk::create_buffer(device, *ctx, (size_t)SPEC::NUM_PROBES * 3 * sizeof(float), STORAGE, HOST_MEMORY, true);
#endif
        ctx->dummy = vk::create_buffer(device, *ctx, 64, STORAGE, HOST_MEMORY, false);

        renderer.backend.context = ctx;
        renderer.backend.module = (void*)(SPEC::HAS_RGB ? ctx->module_rgb : ctx->module_depth);
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const rendering::raytracing::Scene& scene){
        namespace vk = rendering::raytracing::backends::vulkan;
        using TI = typename SPEC::TI;
        auto& ctx = vk::context(renderer);

        rendering::raytracing::detail::compute_scene_bounds(renderer, scene);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << scene.meshes.size() << " geometries ...");

        // scene-dependent resources from a previous init are released here; the pipelines are
        // spec-dependent and survive re-init
        vk::check(device, vkDeviceWaitIdle(ctx.device), "Vulkan: device wait idle failed");
        ctx.render_in_flight = false;
        ctx.collision_in_flight = false;
        vkResetFences(ctx.device, 1, &ctx.fence_render);
        vkResetFences(ctx.device, 1, &ctx.fence_collision);
        vk::destroy_scene_resources(ctx);

        constexpr VkMemoryPropertyFlags HOST_MEMORY = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr VkMemoryPropertyFlags STAGING_MEMORY = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        const size_t num_meshes = scene.meshes.size();
        std::vector<size_t> vertex_offsets(num_meshes), index_offsets(num_meshes), tex_coord_offsets(num_meshes), normal_offsets(num_meshes);
        size_t mesh_data_size = 0;
        auto append_section = [&](size_t bytes) -> size_t {
            size_t offset = mesh_data_size;
            mesh_data_size += (bytes + 15) & ~(size_t)15;
            return offset;
        };
        for(size_t mesh_i = 0; mesh_i < num_meshes; mesh_i++){
            const auto& mesh = scene.meshes[mesh_i];
            vertex_offsets[mesh_i] = append_section(mesh.vertices.size() * sizeof(float));
            index_offsets[mesh_i] = append_section(mesh.indices.size() * sizeof(int));
            tex_coord_offsets[mesh_i] = mesh.tex_coords.empty() ? 0 : append_section(mesh.tex_coords.size() * sizeof(float));
            normal_offsets[mesh_i] = mesh.normals.empty() ? 0 : append_section(mesh.normals.size() * sizeof(float));
        }
        std::vector<uint8_t> mesh_data_host(mesh_data_size);
        for(size_t mesh_i = 0; mesh_i < num_meshes; mesh_i++){
            const auto& mesh = scene.meshes[mesh_i];
            std::memcpy(mesh_data_host.data() + vertex_offsets[mesh_i], mesh.vertices.data(), mesh.vertices.size() * sizeof(float));
            std::memcpy(mesh_data_host.data() + index_offsets[mesh_i], mesh.indices.data(), mesh.indices.size() * sizeof(int));
            if(!mesh.tex_coords.empty()){
                std::memcpy(mesh_data_host.data() + tex_coord_offsets[mesh_i], mesh.tex_coords.data(), mesh.tex_coords.size() * sizeof(float));
            }
            if(!mesh.normals.empty()){
                std::memcpy(mesh_data_host.data() + normal_offsets[mesh_i], mesh.normals.data(), mesh.normals.size() * sizeof(float));
            }
        }
        ctx.mesh_data = vk::create_buffer(device, ctx, mesh_data_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
        vk::upload_buffer(device, ctx, ctx.mesh_data, mesh_data_host.data(), mesh_data_size);
        const VkDeviceAddress mesh_data_address = vk::buffer_address(ctx, ctx.mesh_data);

        std::vector<vk::MeshRecord> mesh_records(num_meshes);
        std::vector<vk::BufferResource> texture_stagings;
        ctx.mesh_textures.clear();
        {
            const uint8_t dummy_pixel[4] = {255, 255, 255, 255};
            ctx.mesh_textures.push_back(vk::create_texture_image(device, ctx, 1, 1, false));
            vk::BufferResource staging = vk::create_buffer(device, ctx, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, STAGING_MEMORY, true);
            std::memcpy(staging.mapped, dummy_pixel, 4);
            texture_stagings.push_back(staging);
        }
        std::vector<int> texture_widths = {1}, texture_heights = {1};
        auto add_texture = [&](const rendering::raytracing::Texture& texture, bool srgb) -> uint32_t {
            uint32_t texture_index = (uint32_t)ctx.mesh_textures.size();
            ctx.mesh_textures.push_back(vk::create_texture_image(device, ctx, texture.width, texture.height, srgb));
            vk::BufferResource staging = vk::create_buffer(device, ctx, texture.pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, STAGING_MEMORY, true);
            std::memcpy(staging.mapped, texture.pixels.data(), texture.pixels.size());
            texture_stagings.push_back(staging);
            texture_widths.push_back(texture.width);
            texture_heights.push_back(texture.height);
            return texture_index;
        };
        for(size_t mesh_i = 0; mesh_i < num_meshes; mesh_i++){
            const auto& mesh = scene.meshes[mesh_i];
            vk::MeshRecord& record = mesh_records[mesh_i];
            record = {};
            record.vertices = mesh_data_address + vertex_offsets[mesh_i];
            record.index = mesh_data_address + index_offsets[mesh_i];
            record.has_tex_coord = mesh.tex_coords.empty() ? 0 : 1;
            record.tex_coord = record.has_tex_coord ? mesh_data_address + tex_coord_offsets[mesh_i] : 0;
            record.has_normals = mesh.normals.empty() ? 0 : 1;
            record.normal = record.has_normals ? mesh_data_address + normal_offsets[mesh_i] : 0;
            record.color[0] = mesh.color[0];
            record.color[1] = mesh.color[1];
            record.color[2] = mesh.color[2];
            record.metallic = mesh.metallic;
            record.roughness = mesh.roughness;
            record.opacity = mesh.opacity;
            record.emissive[0] = mesh.emissive[0];
            record.emissive[1] = mesh.emissive[1];
            record.emissive[2] = mesh.emissive[2];
            record.alpha_cutoff = mesh.alpha_cutoff;
            record.alpha_mode = mesh.alpha_mode;
            if(mesh.texture.present()){
                record.texture = add_texture(mesh.texture, true);
                record.has_texture = 1;
            }
            if(mesh.normal_map.present()){
                record.normal_map = add_texture(mesh.normal_map, false);
                record.has_normal_map = 1;
            }
            if(mesh.metallic_roughness_map.present()){
                record.metallic_roughness_map = add_texture(mesh.metallic_roughness_map, false);
                record.has_metallic_roughness_map = 1;
            }
            if(mesh.emissive_map.present()){
                record.emissive_map = add_texture(mesh.emissive_map, true);
                record.has_emissive_map = 1;
            }
            if(mesh.occlusion_map.present()){
                record.occlusion_map = add_texture(mesh.occlusion_map, false);
                record.has_occlusion_map = 1;
            }
        }
        {
            VkCommandBuffer command_buffer = vk::one_shot_begin(device, ctx);
            for(size_t texture_i = 0; texture_i < ctx.mesh_textures.size(); texture_i++){
                vk::record_texture_upload(command_buffer, ctx.mesh_textures[texture_i], texture_stagings[texture_i], texture_widths[texture_i], texture_heights[texture_i]);
            }
            vk::one_shot_end(device, ctx, command_buffer);
            for(auto& staging : texture_stagings){
                vk::destroy_buffer(ctx, staging);
            }
        }

        ctx.mesh_records = vk::create_buffer(device, ctx, mesh_records.size() * sizeof(vk::MeshRecord), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_MEMORY, true);
        std::memcpy(ctx.mesh_records.mapped, mesh_records.data(), mesh_records.size() * sizeof(vk::MeshRecord));

        const auto scene_lights = rendering::raytracing::detail::effective_scene_lights<SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING>(scene);
        {
            const size_t scene_lights_bytes = scene_lights.empty() ? sizeof(rendering::raytracing::SceneLight) : scene_lights.size() * sizeof(rendering::raytracing::SceneLight);
            ctx.scene_lights = vk::create_buffer(device, ctx, scene_lights_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST_MEMORY, true);
            if(!scene_lights.empty()){
                std::memcpy(ctx.scene_lights.mapped, scene_lights.data(), scene_lights.size() * sizeof(rendering::raytracing::SceneLight));
            }
        }

        auto* params = (vk::LaunchParams*)ctx.launch_params.mapped;
        *params = {};
        params->fb_width = SPEC::FB_WIDTH;
        params->fb_height = SPEC::FB_HEIGHT;
        params->cam_width = SPEC::CAM_WIDTH;
        params->cam_height = SPEC::CAM_HEIGHT;
        params->grid_cols = SPEC::GRID_COLS;
        params->num_cameras = SPEC::NUM_CAMERAS;
        params->num_probes = SPEC::NUM_PROBES;
        params->num_scene_lights = (uint32_t)scene_lights.size();
        params->max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        params->max_dist = renderer.camera_radius * 2.0f;
        params->ambient_color[0] = 0.10f;
        params->ambient_color[1] = 0.10f;
        params->ambient_color[2] = 0.10f;
        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            params->miss_color_0[0] = 0.f; params->miss_color_0[1] = 0.f; params->miss_color_0[2] = 0.f;
            params->miss_color_1[0] = 0.f; params->miss_color_1[1] = 0.f; params->miss_color_1[2] = 0.f;
        } else {
            params->miss_color_0[0] = .8f; params->miss_color_0[1] = 0.f; params->miss_color_0[2] = 0.f;
            params->miss_color_1[0] = .8f; params->miss_color_1[1] = .8f; params->miss_color_1[2] = .8f;
        }

        {
            std::vector<VkAccelerationStructureGeometryKHR> geometries(num_meshes);
            std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(num_meshes);
            std::vector<uint32_t> primitive_counts(num_meshes);
            for(size_t mesh_i = 0; mesh_i < num_meshes; mesh_i++){
                const auto& mesh = scene.meshes[mesh_i];
                VkAccelerationStructureGeometryKHR& geometry = geometries[mesh_i];
                geometry = {};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
                geometry.geometry.triangles = {};
                geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                geometry.geometry.triangles.vertexData.deviceAddress = mesh_data_address + vertex_offsets[mesh_i];
                geometry.geometry.triangles.vertexStride = 3 * sizeof(float);
                geometry.geometry.triangles.maxVertex = (uint32_t)(mesh.vertices.size() / 3) - 1;
                geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
                geometry.geometry.triangles.indexData.deviceAddress = mesh_data_address + index_offsets[mesh_i];
                ranges[mesh_i] = {};
                ranges[mesh_i].primitiveCount = (uint32_t)(mesh.indices.size() / 3);
                primitive_counts[mesh_i] = ranges[mesh_i].primitiveCount;
            }
            VkAccelerationStructureBuildGeometryInfoKHR blas_build_info{};
            blas_build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            blas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blas_build_info.geometryCount = (uint32_t)geometries.size();
            blas_build_info.pGeometries = geometries.data();
            VkAccelerationStructureBuildSizesInfoKHR blas_sizes{};
            blas_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx.vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_build_info, primitive_counts.data(), &blas_sizes);

            ctx.blas_buffer = vk::create_buffer(device, ctx, blas_sizes.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
            VkAccelerationStructureCreateInfoKHR blas_info{};
            blas_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            blas_info.buffer = ctx.blas_buffer.buffer;
            blas_info.size = blas_sizes.accelerationStructureSize;
            blas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            vk::check(device, ctx.vkCreateAccelerationStructureKHR(ctx.device, &blas_info, nullptr, &ctx.blas), "Vulkan: BLAS creation failed");

            VkAccelerationStructureDeviceAddressInfoKHR blas_address_info{};
            blas_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            blas_address_info.accelerationStructure = ctx.blas;
            const VkDeviceAddress blas_address = ctx.vkGetAccelerationStructureDeviceAddressKHR(ctx.device, &blas_address_info);

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform.matrix[0][0] = 1.0f;
            instance.transform.matrix[1][1] = 1.0f;
            instance.transform.matrix[2][2] = 1.0f;
            instance.mask = 0xFF;
            instance.accelerationStructureReference = blas_address;
            ctx.instance_buffer = vk::create_buffer(device, ctx, sizeof(instance),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                HOST_MEMORY, true);
            std::memcpy(ctx.instance_buffer.mapped, &instance, sizeof(instance));

            VkAccelerationStructureGeometryKHR tlas_geometry{};
            tlas_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlas_geometry.geometry.instances = {};
            tlas_geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            tlas_geometry.geometry.instances.data.deviceAddress = vk::buffer_address(ctx, ctx.instance_buffer);
            VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info{};
            tlas_build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            tlas_build_info.geometryCount = 1;
            tlas_build_info.pGeometries = &tlas_geometry;
            uint32_t tlas_primitive_count = 1;
            VkAccelerationStructureBuildSizesInfoKHR tlas_sizes{};
            tlas_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx.vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_build_info, &tlas_primitive_count, &tlas_sizes);

            ctx.tlas_buffer = vk::create_buffer(device, ctx, tlas_sizes.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
            VkAccelerationStructureCreateInfoKHR tlas_info{};
            tlas_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            tlas_info.buffer = ctx.tlas_buffer.buffer;
            tlas_info.size = tlas_sizes.accelerationStructureSize;
            tlas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            vk::check(device, ctx.vkCreateAccelerationStructureKHR(ctx.device, &tlas_info, nullptr, &ctx.tlas), "Vulkan: TLAS creation failed");

            const VkDeviceSize scratch_alignment = ctx.min_scratch_alignment;
            VkDeviceSize scratch_size = (blas_sizes.buildScratchSize > tlas_sizes.buildScratchSize ? blas_sizes.buildScratchSize : tlas_sizes.buildScratchSize) + scratch_alignment;
            vk::BufferResource scratch = vk::create_buffer(device, ctx, scratch_size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
            VkDeviceAddress scratch_address = vk::buffer_address(ctx, scratch);
            scratch_address = (scratch_address + scratch_alignment - 1) & ~(VkDeviceAddress)(scratch_alignment - 1);

            blas_build_info.dstAccelerationStructure = ctx.blas;
            blas_build_info.scratchData.deviceAddress = scratch_address;
            tlas_build_info.dstAccelerationStructure = ctx.tlas;
            tlas_build_info.scratchData.deviceAddress = scratch_address;

            VkCommandBuffer command_buffer = vk::one_shot_begin(device, ctx);
            const VkAccelerationStructureBuildRangeInfoKHR* blas_ranges = ranges.data();
            ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &blas_build_info, &blas_ranges);
            VkMemoryBarrier build_barrier{};
            build_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            build_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            build_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &build_barrier, 0, nullptr, 0, nullptr);
            VkAccelerationStructureBuildRangeInfoKHR tlas_range{};
            tlas_range.primitiveCount = 1;
            const VkAccelerationStructureBuildRangeInfoKHR* tlas_ranges = &tlas_range;
            ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &tlas_build_info, &tlas_ranges);
            vk::one_shot_end(device, ctx, command_buffer);
            vk::destroy_buffer(ctx, scratch);
        }
        renderer.backend.world = (void*)ctx.tlas;

        if(!ctx.pipelines_built){
            struct SpecializationData{
                VkBool32 srgb_output;
                VkBool32 motion_blur;
                int32_t motion_samples;
                int32_t aa_grid;
                VkBool32 checker_background;
                VkBool32 load_textures;
                VkBool32 normal_shading;
                VkBool32 metallic_reflections;
                VkBool32 pbr_shading;
                VkBool32 punctual_light_shadows;
            };
            static_assert(sizeof(SpecializationData) == vk::specialization_constants::COUNT * 4);
            SpecializationData specialization_data{};
            specialization_data.srgb_output = SPEC::SHADING::SRGB_OUTPUT ? VK_TRUE : VK_FALSE;
            specialization_data.motion_blur = SPEC::ENABLE_MOTION_BLUR ? VK_TRUE : VK_FALSE;
            specialization_data.motion_samples = SPEC::ENABLE_MOTION_BLUR ? (int32_t)SPEC::MOTION_BLUR_SAMPLES : 1;
            specialization_data.aa_grid = SPEC::ENABLE_ANTI_ALIASING ? (int32_t)SPEC::ANTI_ALIASING_GRID_SIZE : 1;
            specialization_data.checker_background = SPEC::SHADING::CHECKER_BACKGROUND ? VK_TRUE : VK_FALSE;
            specialization_data.load_textures = SPEC::SHADING::LOAD_TEXTURES ? VK_TRUE : VK_FALSE;
            specialization_data.normal_shading = SPEC::SHADING::NORMAL_SHADING ? VK_TRUE : VK_FALSE;
            specialization_data.metallic_reflections = SPEC::SHADING::METALLIC_REFLECTIONS ? VK_TRUE : VK_FALSE;
            specialization_data.pbr_shading = SPEC::SHADING::PBR_SHADING ? VK_TRUE : VK_FALSE;
            specialization_data.punctual_light_shadows = SPEC::SHADING::PUNCTUAL_LIGHT_SHADOWS ? VK_TRUE : VK_FALSE;
            VkSpecializationMapEntry map_entries[vk::specialization_constants::COUNT];
            for(uint32_t constant_i = 0; constant_i < vk::specialization_constants::COUNT; constant_i++){
                map_entries[constant_i] = {constant_i, constant_i * 4, 4};
            }
            VkSpecializationInfo specialization_info{};
            specialization_info.mapEntryCount = vk::specialization_constants::COUNT;
            specialization_info.pMapEntries = map_entries;
            specialization_info.dataSize = sizeof(specialization_data);
            specialization_info.pData = &specialization_data;

            auto make_pipeline = [&](VkShaderModule module) -> VkPipeline {
                VkComputePipelineCreateInfo pipeline_info{};
                pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pipeline_info.stage = {};
                pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pipeline_info.stage.module = module;
                pipeline_info.stage.pName = "main";
                pipeline_info.stage.pSpecializationInfo = &specialization_info;
                pipeline_info.layout = ctx.pipeline_layout;
                VkPipeline pipeline;
                vk::check(device, vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "Vulkan: compute pipeline creation failed");
                return pipeline;
            };
            if constexpr (SPEC::HAS_RGB) {
                ctx.rgb_pipeline = make_pipeline(ctx.module_rgb);
                renderer.backend.ray_gen = (void*)ctx.rgb_pipeline;
            }
            if constexpr (SPEC::HAS_DEPTH) {
                ctx.depth_pipeline = make_pipeline(ctx.module_depth);
                renderer.backend.depth_ray_gen = (void*)ctx.depth_pipeline;
            }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
            ctx.collision_pipeline = make_pipeline(ctx.module_collision);
            renderer.backend.collision_ray_gen = (void*)ctx.collision_pipeline;
#endif
            ctx.pipelines_built = true;
        }

        {
            const uint32_t texture_count = (uint32_t)ctx.mesh_textures.size();
            utils::assert_exit(device, texture_count <= vk::MAX_TEXTURE_DESCRIPTORS, "Vulkan: scene exceeds MAX_TEXTURE_DESCRIPTORS");
            VkDescriptorPoolSize pool_sizes[3];
            pool_sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vk::bindings::COUNT};
            pool_sizes[1] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
            pool_sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texture_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = 1;
            pool_info.poolSizeCount = 3;
            pool_info.pPoolSizes = pool_sizes;
            vk::check(device, vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &ctx.descriptor_pool), "Vulkan: descriptor pool creation failed");

            VkDescriptorSetVariableDescriptorCountAllocateInfo variable_count_info{};
            variable_count_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
            variable_count_info.descriptorSetCount = 1;
            variable_count_info.pDescriptorCounts = &texture_count;
            VkDescriptorSetAllocateInfo set_info{};
            set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_info.pNext = &variable_count_info;
            set_info.descriptorPool = ctx.descriptor_pool;
            set_info.descriptorSetCount = 1;
            set_info.pSetLayouts = &ctx.descriptor_set_layout;
            vk::check(device, vkAllocateDescriptorSets(ctx.device, &set_info, &ctx.descriptor_set), "Vulkan: descriptor set allocation failed");

            auto buffer_or_dummy = [&](const vk::BufferResource& resource) -> VkDescriptorBufferInfo {
                const vk::BufferResource& source = resource.buffer != VK_NULL_HANDLE ? resource : ctx.dummy;
                return {source.buffer, 0, VK_WHOLE_SIZE};
            };
            VkDescriptorBufferInfo buffer_infos[vk::bindings::COUNT];
            buffer_infos[vk::bindings::LAUNCH_PARAMS] = buffer_or_dummy(ctx.launch_params);
            buffer_infos[vk::bindings::CAMERAS_CLOSE] = buffer_or_dummy(ctx.cameras);
            buffer_infos[vk::bindings::CAMERAS_OPEN] = ctx.cameras_open.buffer != VK_NULL_HANDLE ? buffer_or_dummy(ctx.cameras_open) : buffer_or_dummy(ctx.cameras);
            buffer_infos[vk::bindings::FRAME_BUFFER] = buffer_or_dummy(ctx.frame_buffer);
            buffer_infos[vk::bindings::MESH_RECORDS] = buffer_or_dummy(ctx.mesh_records);
            buffer_infos[vk::bindings::SCENE_LIGHTS] = buffer_or_dummy(ctx.scene_lights);
            buffer_infos[vk::bindings::PROBE_DIRECTIONS] = buffer_or_dummy(ctx.probe_directions);
            buffer_infos[vk::bindings::COLLISION_RESULTS] = buffer_or_dummy(ctx.collision_results);
            buffer_infos[vk::bindings::DEPTH_BUFFER] = buffer_or_dummy(ctx.depth_buffer);

            std::vector<VkWriteDescriptorSet> writes;
            for(uint32_t binding_i = 0; binding_i < vk::bindings::COUNT; binding_i++){
                if(binding_i == vk::bindings::ACCELERATION_STRUCTURE || binding_i == vk::bindings::TEXTURES){
                    continue;
                }
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = ctx.descriptor_set;
                write.dstBinding = binding_i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer_infos[binding_i];
                writes.push_back(write);
            }
            VkWriteDescriptorSetAccelerationStructureKHR tlas_write_info{};
            tlas_write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            tlas_write_info.accelerationStructureCount = 1;
            tlas_write_info.pAccelerationStructures = &ctx.tlas;
            VkWriteDescriptorSet tlas_write{};
            tlas_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            tlas_write.pNext = &tlas_write_info;
            tlas_write.dstSet = ctx.descriptor_set;
            tlas_write.dstBinding = vk::bindings::ACCELERATION_STRUCTURE;
            tlas_write.descriptorCount = 1;
            tlas_write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            writes.push_back(tlas_write);
            std::vector<VkDescriptorImageInfo> image_infos(texture_count);
            for(uint32_t texture_i = 0; texture_i < texture_count; texture_i++){
                image_infos[texture_i] = {ctx.sampler, ctx.mesh_textures[texture_i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
            VkWriteDescriptorSet textures_write{};
            textures_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            textures_write.dstSet = ctx.descriptor_set;
            textures_write.dstBinding = vk::bindings::TEXTURES;
            textures_write.descriptorCount = texture_count;
            textures_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            textures_write.pImageInfo = image_infos.data();
            writes.push_back(textures_write);
            vkUpdateDescriptorSets(ctx.device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
        }

        {
            auto record = [&](VkCommandBuffer command_buffer, VkPipeline pipeline, uint32_t groups_x, uint32_t groups_y){
                vk::check(device, vkResetCommandBuffer(command_buffer, 0), "Vulkan: command buffer reset failed");
                VkCommandBufferBeginInfo begin_info{};
                begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vk::check(device, vkBeginCommandBuffer(command_buffer, &begin_info), "Vulkan: command buffer begin failed");
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout, 0, 1, &ctx.descriptor_set, 0, nullptr);
                vkCmdDispatch(command_buffer, groups_x, groups_y, 1);
                VkMemoryBarrier to_host{};
                to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr, 0, nullptr);
                vk::check(device, vkEndCommandBuffer(command_buffer), "Vulkan: command buffer end failed");
            };
            constexpr uint32_t fb_groups_x = (SPEC::FB_WIDTH + vk::WORKGROUP_SIZE - 1) / vk::WORKGROUP_SIZE;
            constexpr uint32_t fb_groups_y = (SPEC::FB_HEIGHT + vk::WORKGROUP_SIZE - 1) / vk::WORKGROUP_SIZE;
            if constexpr (SPEC::HAS_RGB) {
                record(ctx.cb_rgb, ctx.rgb_pipeline, fb_groups_x, fb_groups_y);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                record(ctx.cb_depth, ctx.depth_pipeline, fb_groups_x, fb_groups_y);
            }
            if(ctx.collision_pipeline != VK_NULL_HANDLE){
                record(ctx.cb_collision, ctx.collision_pipeline, (SPEC::NUM_CAMERAS + vk::WORKGROUP_SIZE - 1) / vk::WORKGROUP_SIZE, (SPEC::NUM_PROBES + vk::WORKGROUP_SIZE - 1) / vk::WORKGROUP_SIZE);
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        namespace vk = rendering::raytracing::backends::vulkan;
        rendering::raytracing::detail::generate_camera_poses(device, renderer, center, radius, up, fov);

        auto& ctx = vk::context(renderer);
        vk::wait_in_flight(device, ctx);
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        std::memcpy(ctx.cameras.mapped, data(renderer.cameras), camera_bytes);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            std::memcpy(ctx.cameras_open.mapped, data(renderer.cameras), camera_bytes);
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_in_flight(device, ctx);

        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        std::memcpy(ctx.cameras.mapped, data(cameras), camera_bytes);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            std::memcpy(ctx.cameras_open.mapped, data(cameras), camera_bytes);
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras(device, renderer, cameras);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_OPEN_SPEC, typename CAMERAS_CLOSE_SPEC>
    void set_motion_blur_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_OPEN_SPEC>& cameras_open, const Tensor<CAMERAS_CLOSE_SPEC>& cameras_close){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "set_motion_blur_cameras requires a motion-blur renderer specification");
        static_assert(utils::typing::is_same_v<typename CAMERAS_OPEN_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(utils::typing::is_same_v<typename CAMERAS_CLOSE_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_OPEN_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<0>(typename CAMERAS_CLOSE_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_in_flight(device, ctx);

        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        std::memcpy(ctx.cameras_open.mapped, data(cameras_open), camera_bytes);
        std::memcpy(ctx.cameras.mapped, data(cameras_close), camera_bytes);
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        std::memcpy(ctx.probe_directions.mapped, dirs.data(), dirs.size() * sizeof(float));
        renderer.backend.probe_dirs_buffer = (void*)ctx.probe_directions.buffer;
#endif
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        if constexpr (SPEC::HAS_RGB && SPEC::HAS_DEPTH) {
            vk::submit_render(device, ctx, ctx.cb_rgb, ctx.cb_depth);
        }
        else if constexpr (SPEC::HAS_RGB) {
            vk::submit_render(device, ctx, ctx.cb_rgb);
        }
        else if constexpr (SPEC::HAS_DEPTH) {
            vk::submit_render(device, ctx, ctx.cb_depth);
        }
        if(renderer.backend.collision_ray_gen != nullptr){
            vk::wait_collision_in_flight(device, ctx);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &ctx.cb_collision;
            vk::check(device, vkQueueSubmit(ctx.queue, 1, &submit_info, ctx.fence_collision), "Vulkan: collision submit failed");
            ctx.collision_in_flight = true;
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        if(renderer.backend.collision_ray_gen != nullptr){
            vk::wait_collision_in_flight(device, ctx);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &ctx.cb_collision;
            vk::check(device, vkQueueSubmit(ctx.queue, 1, &submit_info, ctx.fence_collision), "Vulkan: collision submit failed");
            ctx.collision_in_flight = true;
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_collision_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_collision_only_launch(device, renderer);
        render_collision_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::submit_render(device, ctx, ctx.cb_rgb);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_render_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_only_launch(device, renderer);
        render_rgb_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::submit_render(device, ctx, ctx.cb_depth);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_render_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_depth_only_launch(device, renderer);
        render_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::submit_render(device, ctx, ctx.cb_rgb, ctx.cb_depth);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_render_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_depth_only_launch(device, renderer);
        render_rgb_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void render_rgb_only_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras_async(device, renderer, cameras);
        render_rgb_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename FB_SPEC>
    void read_frame_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<FB_SPEC>& out_pixels){
        static_assert(SPEC::HAS_RGB, "read_frame_buffer requires an RGB-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename FB_SPEC::T, uint32_t>);
        static_assert(get<0>(typename FB_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace vk = rendering::raytracing::backends::vulkan;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_pixels), vk::context(renderer).frame_buffer.mapped, expected * sizeof(uint32_t));
    }

    template <typename DEVICE, typename SPEC, typename DEPTH_SPEC>
    void read_depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<DEPTH_SPEC>& out_depth){
        static_assert(SPEC::HAS_DEPTH, "read_depth_buffer requires a depth-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename DEPTH_SPEC::T, float>);
        static_assert(get<0>(typename DEPTH_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);
        namespace vk = rendering::raytracing::backends::vulkan;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_depth), vk::context(renderer).depth_buffer.mapped, expected * sizeof(float));
    }

    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        rendering::raytracing::detail::write_grid_png<SPEC>((const uint32_t*)vk::context(renderer).frame_buffer.mapped, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        rendering::raytracing::detail::write_depth_grid_png<SPEC>((const float*)vk::context(renderer).depth_buffer.mapped, renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        rendering::raytracing::detail::write_depth_bin<SPEC>((const float*)vk::context(renderer).depth_buffer.mapped, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        namespace vk = rendering::raytracing::backends::vulkan;
        const auto* probe_results = (const rendering::raytracing::CollisionResult*)vk::context(renderer).collision_results.mapped;
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(probe_results, filename);
#endif
    }

    template <typename DEVICE, typename SPEC, typename COLL_SPEC>
    void read_collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<COLL_SPEC>& out){
        static_assert(utils::typing::is_same_v<typename COLL_SPEC::T, rendering::raytracing::CollisionResult>);
        static_assert(get<0>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_PROBES);
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        namespace vk = rendering::raytracing::backends::vulkan;
        if(renderer.backend.collision_results_buffer != nullptr){
            std::memcpy(data(out), vk::context(renderer).collision_results.mapped,
                        SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult));
        }
#endif
    }

    template <typename DEVICE, typename SPEC>
    const rendering::raytracing::CollisionResult* read_collision_results_raw(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        return nullptr;
#else
        namespace vk = rendering::raytracing::backends::vulkan;
        if(renderer.backend.collision_results_buffer == nullptr){
            return nullptr;
        }
        return (const rendering::raytracing::CollisionResult*)vk::context(renderer).collision_results.mapped;
#endif
    }

    template <typename DEVICE, typename SPEC>
    uint32_t* get_framebuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "get_framebuffer_device_ptr requires an RGB-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        return (uint32_t*)vk::context(renderer).frame_buffer.mapped;
    }

    template <typename DEVICE, typename SPEC>
    float* get_depthbuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "get_depthbuffer_device_ptr requires a depth-capable renderer specification");
        namespace vk = rendering::raytracing::backends::vulkan;
        return (float*)vk::context(renderer).depth_buffer.mapped;
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        auto& ctx = vk::context(renderer);
        vk::wait_in_flight(device, ctx);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace vk = rendering::raytracing::backends::vulkan;
        if(renderer.backend.context != nullptr){
            auto& ctx = vk::context(renderer);
            vkDeviceWaitIdle(ctx.device);
            vk::destroy_scene_resources(ctx);
            vk::destroy_buffer(ctx, ctx.launch_params);
            vk::destroy_buffer(ctx, ctx.cameras);
            vk::destroy_buffer(ctx, ctx.cameras_open);
            vk::destroy_buffer(ctx, ctx.frame_buffer);
            vk::destroy_buffer(ctx, ctx.depth_buffer);
            vk::destroy_buffer(ctx, ctx.collision_results);
            vk::destroy_buffer(ctx, ctx.probe_directions);
            vk::destroy_buffer(ctx, ctx.dummy);
            if(ctx.rgb_pipeline != VK_NULL_HANDLE){ vkDestroyPipeline(ctx.device, ctx.rgb_pipeline, nullptr); }
            if(ctx.depth_pipeline != VK_NULL_HANDLE){ vkDestroyPipeline(ctx.device, ctx.depth_pipeline, nullptr); }
            if(ctx.collision_pipeline != VK_NULL_HANDLE){ vkDestroyPipeline(ctx.device, ctx.collision_pipeline, nullptr); }
            if(ctx.module_rgb != VK_NULL_HANDLE){ vkDestroyShaderModule(ctx.device, ctx.module_rgb, nullptr); }
            if(ctx.module_depth != VK_NULL_HANDLE){ vkDestroyShaderModule(ctx.device, ctx.module_depth, nullptr); }
            if(ctx.module_collision != VK_NULL_HANDLE){ vkDestroyShaderModule(ctx.device, ctx.module_collision, nullptr); }
            vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, nullptr);
            vkDestroySampler(ctx.device, ctx.sampler, nullptr);
            vkDestroyFence(ctx.device, ctx.fence_render, nullptr);
            vkDestroyFence(ctx.device, ctx.fence_collision, nullptr);
            vkDestroyCommandPool(ctx.device, ctx.command_pool, nullptr);
            vkDestroyDevice(ctx.device, nullptr);
            vkDestroyInstance(ctx.instance, nullptr);
            delete &ctx;
            renderer.backend.context = nullptr;
        }
        free(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            free(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            free(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            free(device, renderer.depth_buffer);
        }
        free(device, renderer.collision_results);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
