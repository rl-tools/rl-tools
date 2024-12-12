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
        constexpr static T FLAG_DISTANCE_THRESHOLD = 1;
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

    template <typename TI>
    struct ObservationMemory{
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
    template <typename T, typename TI>
    struct StateMemory: State<T, TI>{
        bool first_step;
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
        static constexpr TI EPISODE_STEP_LIMIT = 400;
    };
    template <typename T_SPEC>
    struct FlagMemory: Flag<T_SPEC>{
        // The Flag environment was meant to test memory but it is also a quite hard exploration problem. This modified environment tests the memory better by revealing the flag position at the first state. The agent also must only reach the flag once.
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = flag::StateMemory<T, TI>;
        using Observation = flag::ObservationMemory<TI>;
        using ObservationPrivileged = Observation;
        static constexpr TI EPISODE_STEP_LIMIT = 400;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END







#endif
