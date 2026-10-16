#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_H

#include "../../../../numeric_types/policy.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::controllers{
    template<typename T_TYPE_POLICY, typename T_TI, T_TI T_INSTANCES, bool T_DYNAMIC_ALLOCATION = true>
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr TI INSTANCES = T_INSTANCES;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        static_assert(INSTANCES > 0);
    };
    template<typename T>
    struct Bounds{
        T lower;
        T upper;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
