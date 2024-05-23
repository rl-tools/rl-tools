#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EVALUATION_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EVALUATION_STATE_H

#include "../../../../rl/utils/evaluation/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::loop::steps::evaluation{
    template<typename T_CONFIG, typename T_NEXT = typename T_CONFIG::NEXT::template State<typename T_CONFIG::NEXT>>
    struct State: T_NEXT {
        using CONFIG = T_CONFIG;
        using NEXT = T_NEXT;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using RESULT_SPEC = rl::utils::evaluation::Specification<T, TI, typename CONFIG::ENVIRONMENT_EVALUATION, CONFIG::EVALUATION_PARAMETERS::NUM_EVALUATION_EPISODES, CONFIG::EVALUATION_PARAMETERS::EPISODE_STEP_LIMIT, false>;
        rl::utils::evaluation::Result<RESULT_SPEC> evaluation_results[CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS];
        typename CONFIG::RNG rng_eval;
        typename NEXT::CONFIG::ENVIRONMENT_EVALUATION env_eval;
        typename CONFIG::UI ui;
    };
}
#endif




