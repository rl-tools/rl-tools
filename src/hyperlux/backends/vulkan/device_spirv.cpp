#include <stdint.h>
#include <stddef.h>

#include "device_spirv_rgb.h"
#include "device_spirv_auxiliary.h"
#include "device_spirv_collision.h"
#include "device_spirv_flow_forward.h"
#include "device_spirv_resolve.h"

extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_rgb(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_hyperlux_vulkan_device_spirv_rgb_words);
    return rl_tools_hyperlux_vulkan_device_spirv_rgb_words;
}
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_auxiliary(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_hyperlux_vulkan_device_spirv_auxiliary_words);
    return rl_tools_hyperlux_vulkan_device_spirv_auxiliary_words;
}
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_collision(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_hyperlux_vulkan_device_spirv_collision_words);
    return rl_tools_hyperlux_vulkan_device_spirv_collision_words;
}
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_flow_forward(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_hyperlux_vulkan_device_spirv_flow_forward_words);
    return rl_tools_hyperlux_vulkan_device_spirv_flow_forward_words;
}
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_resolve(size_t* size_bytes){
    *size_bytes = sizeof(rl_tools_hyperlux_vulkan_device_spirv_resolve_words);
    return rl_tools_hyperlux_vulkan_device_spirv_resolve_words;
}
