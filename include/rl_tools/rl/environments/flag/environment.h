#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_FLAG_ENVIRONMENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_FLAG_ENVIRONMENT_H

#include "../../../math/operations_generic.h"
#include "../environments.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::flag{
    template <typename T>
    struct DefaultParameters {
        constexpr static T MAX_ACCELERATION = 50;
        constexpr static T MAX_VELOCITY = 5;
        constexpr static T FLAG_DISTANCE_THRESHOLD = 2;
        constexpr static T DT = 0.10;
        constexpr static T BOARD_SIZE = 10;
        T flag_position[2];
    };
    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename TI>
    struct Observation{
        static constexpr TI DIM = 4 + 3;
    };
    template <typename TI>
    struct ObservationPrivileged{
        static constexpr TI DIM = 4 + 3 + 2;
    };

    template <typename T, typename TI>
    struct State{
        enum class StateMachine{
            INITIAL = 0,
            FLAG_VISITED = 1,
            ORIGIN_VISITED = 2,
            FLAG_VISITED_AGAIN = 3
        };
        static constexpr TI DIM = 5;
        T position[2];
        T velocity[2];
        StateMachine state_machine;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    template <typename T_SPEC>
    struct Flag: Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = flag::State<T, TI>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = flag::Observation<TI>;
        using ObservationPrivileged = Observation; //flag::ObservationPrivileged<TI>;
        static constexpr TI N_AGENTS = 1; // single agent
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 200;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END







#endif
