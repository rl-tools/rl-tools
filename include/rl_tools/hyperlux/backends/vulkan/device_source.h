#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_BACKENDS_VULKAN_DEVICE_SOURCE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_BACKENDS_VULKAN_DEVICE_SOURCE_H

#include <cstdint>
#include <cstddef>

extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_rgb(size_t* size_bytes);
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_auxiliary(size_t* size_bytes);
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_collision(size_t* size_bytes);
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_flow_forward(size_t* size_bytes);
extern "C" const uint32_t* rl_tools_hyperlux_vulkan_device_spirv_resolve(size_t* size_bytes);

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::backends::vulkan{
    inline const uint32_t* device_spirv_rgb(size_t& size_bytes){
        return rl_tools_hyperlux_vulkan_device_spirv_rgb(&size_bytes);
    }
    inline const uint32_t* device_spirv_auxiliary(size_t& size_bytes){
        return rl_tools_hyperlux_vulkan_device_spirv_auxiliary(&size_bytes);
    }
    inline const uint32_t* device_spirv_collision(size_t& size_bytes){
        return rl_tools_hyperlux_vulkan_device_spirv_collision(&size_bytes);
    }
    inline const uint32_t* device_spirv_flow_forward(size_t& size_bytes){
        return rl_tools_hyperlux_vulkan_device_spirv_flow_forward(&size_bytes);
    }
    inline const uint32_t* device_spirv_resolve(size_t& size_bytes){
        return rl_tools_hyperlux_vulkan_device_spirv_resolve(&size_bytes);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
