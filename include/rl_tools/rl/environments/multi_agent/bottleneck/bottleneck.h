#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_BOTTLENECK_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_BOTTLENECK_H

#include "../../../../math/operations_generic.h"
#include "../environments.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::multi_agent::bottleneck {
    template <typename T_T, typename T_TI>
    struct DefaultParameters {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI N_AGENTS = 2;
        static constexpr TI LIDAR_RESOLUTION = 3;
        static constexpr T LIDAR_FOV = math::PI<T> * 20/180; // in radians (0 to PI)
    };
    template <typename T_PARAMETERS>
    struct Observation{
        using PARAMETERS = T_PARAMETERS;
        using T = typename PARAMETERS::T;
        using TI = typename PARAMETERS::TI;
        static constexpr TI DIM = 4 + PARAMETERS::LIDAR_RESOLUTION;
    };
    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T, T_TI>, typename T_OBSERVATION = Observation<T_PARAMETERS>>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using OBSERVATION = T_OBSERVATION;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename T, typename TI>
    struct AgentState {
        T position[2];
        T velocity[2];
    };

    template <typename SPEC>
    struct State{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        AgentState<T, TI> agent_states[SPEC::PARAMETERS::N_AGENTS];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::multi_agent{
    template <typename T_SPEC>
    struct Bottleneck: Environment{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = pendulum::State<T, TI>;
        using PARAMETERS = typename SPEC::PARAMETERS;
        using OBSERVATION = typename SPEC::OBSERVATION;
        static constexpr TI N_AGENTS = PARAMETERS::N_AGENTS;
        static constexpr TI OBSERVATION_DIM = OBSERVATION::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = 2; // x and y acceleration
        static constexpr TI EPISODE_STEP_LIMIT = 200;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END







#endif
