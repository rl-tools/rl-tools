#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EVALUATION_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EVALUATION_CONFIG_H

#include "../../../../rl/components/off_policy_runner/operations_generic.h"

#include "../../../../rl/utils/evaluation.h"

#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::loop::steps::evaluation{
    template <typename T, typename TI, typename NEXT>
    struct DefaultParameters{
        static constexpr bool DETERMINISTIC_EVALUATION = true;
        static constexpr TI EVALUATION_INTERVAL = 1000;
        static constexpr TI NUM_EVALUATION_EPISODES = 10;
        static constexpr TI N_EVALUATIONS = NEXT::PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
    };
    template<typename T_NEXT, typename T_PARAMETERS = DefaultParameters<typename T_NEXT::T, typename T_NEXT::TI, T_NEXT>>
    struct DefaultConfig: T_NEXT {
        using NEXT = T_NEXT;
        using PARAMETERS = T_PARAMETERS;
        using T = typename NEXT::T;
        using TI = typename NEXT::TI;
        static_assert(PARAMETERS::N_EVALUATIONS > 0 && PARAMETERS::N_EVALUATIONS < 1000000);
        template <typename CONFIG>
        using State = TrainingState<CONFIG>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif




