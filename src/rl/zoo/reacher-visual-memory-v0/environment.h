#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_REACHER_VISUAL_MEMORY_V0_ENVIRONMENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_REACHER_VISUAL_MEMORY_V0_ENVIRONMENT_H

#include <rl_tools/rl/environments/reacher/operations_cpu.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::reacher_visual_memory_v0{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        struct REACHER_PARAMETERS: rlt::rl::environments::reacher::DefaultParameters<T>{
            static constexpr auto IMAGE_HEIGHT = 16;
            static constexpr auto IMAGE_WIDTH = 16;
        };
        using ENVIRONMENT_SPEC = rlt::rl::environments::reacher::Specification<T, TI, REACHER_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::ReacherVisualMemory<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
