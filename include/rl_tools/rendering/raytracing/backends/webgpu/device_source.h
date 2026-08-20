#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_DEVICE_SOURCE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_DEVICE_SOURCE_H

extern "C" const char rl_tools_rendering_raytracing_webgpu_device_source[];

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::webgpu{
    inline const char* device_source(){
        return rl_tools_rendering_raytracing_webgpu_device_source;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
