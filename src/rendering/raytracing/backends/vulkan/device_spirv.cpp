#include <stdint.h>
#include <stddef.h>

#include "device_spirv_rgb.h"
#include "device_spirv_depth.h"
#include "device_spirv_collision.h"
#include "device_spirv_segmentation.h"
#include "device_spirv_normals.h"
#include "device_spirv_resolve.h"

extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_rgb(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_rgb_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_rgb_words;
}
extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_depth(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_depth_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_depth_words;
}
extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_collision(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_collision_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_collision_words;
}
extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_segmentation(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_segmentation_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_segmentation_words;
}
extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_normals(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_normals_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_normals_words;
}
extern "C" const uint32_t* rl_tools_rendering_raytracing_vulkan_device_spirv_resolve(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_rendering_raytracing_vulkan_device_spirv_resolve_words);
    return rl_tools_rendering_raytracing_vulkan_device_spirv_resolve_words;
}
