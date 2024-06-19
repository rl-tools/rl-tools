#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_WRAPPER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_WRAPPER_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environment_wrappers {
    template <typename T_ENVIRONMENT>
    struct Wrapper{
        using ENVIRONMENT = T_ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
        static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
        using Parameters = typename ENVIRONMENT::Parameters;
        using State = typename ENVIRONMENT::State;

        ENVIRONMENT env;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "scale_observations/wrapper.h"

#endif

