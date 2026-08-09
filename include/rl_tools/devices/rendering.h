#include "../version.h"
#include "../rl_tools.h"
#include "../utils/generic/typing.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_RENDERING_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_RENDERING_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices::rendering {
    struct None {};
    struct Generic {};
    struct Optix {};
    struct Metal {};
    struct Vulkan {};

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
    using Default = Metal;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    using Default = Optix;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
    using Default = Vulkan;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)
    using Default = Generic;
#else
    using Default = None;
#endif

    template <typename SPEC, typename = void>
    struct component {
        using TYPE = Default;
    };

    template <typename SPEC>
    struct component<SPEC, utils::typing::void_t<typename SPEC::RENDERING>> {
        using TYPE = typename SPEC::RENDERING;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
