#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_CURRICULUM_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_CURRICULUM_CONFIG_H

#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::loop::steps::curriculum{
    struct DefaultTag{};
    template<typename T_TI, T_TI T_INTERVAL=0>
    struct Parameters{
        static constexpr T_TI INTERVAL = T_INTERVAL; // 0 = use evaluation interval
    };
    template<typename T_NEXT, typename T_PARAMETERS = Parameters<typename T_NEXT::TI>, typename T_TAG = DefaultTag>
    struct Config: T_NEXT {
        using NEXT = T_NEXT;
        using CURRICULUM_PARAMETERS = T_PARAMETERS;
        using CURRICULUM_TAG = T_TAG;
        template <typename CONFIG>
        using State = State<CONFIG>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
