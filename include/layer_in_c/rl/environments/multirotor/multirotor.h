#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H

#include <layer_in_c/devices.h>

namespace layer_in_c::rl::environments::multirotor {
    constexpr index_t STATE_DIM = 13;
    constexpr index_t ACTION_DIM = 4;

    template <typename T, auto N>
    class Parameters {
    public:
        struct Dynamics{
            T rotor_positions[N][3];
            T rotor_thrust_directions[N][3];
            T rotor_torque_directions[N][3];
            T thrust_constants[3];
            T torque_constant;
            T mass;
            T gravity[3];
            T J[3][3];
            T J_inv[3][3];
        };
        struct ActionLimit{
            T min;
            T max;
        };
        struct Reward{
            T position;
            T orientation;
            T linear_velocity;
            T angular_velocity;
            T action_baseline;
            T action;
        };
        struct Initialization{
            T max_position;
            T max_linear_velocity;
            T max_angular_velocity;
        };
        Dynamics dynamics;
        ActionLimit action_limit;
        Initialization init;
        Reward reward;
        T dt;
    };
    struct StaticParameters{};
    template <typename T_T, typename T_STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using STATIC_PARAMETERS = T_STATIC_PARAMETERS;
    };

    template <typename T>
    struct State{
        static constexpr index_t DIM = STATE_DIM;
        T state[DIM];
    };

}

namespace layer_in_c::rl::environments{
    template <typename DEVICE, typename SPEC>
    struct Multirotor{
        static constexpr bool REQUIRES_OBSERVATION = false;
        static constexpr index_t OBSERVATION_DIM = multirotor::STATE_DIM;
        static constexpr index_t ACTION_DIM = 4;
        using State = multirotor::State<typename SPEC::T>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        multirotor::Parameters<typename SPEC::T, 4> parameters;
    };
}

#include "default_parameters.h"

#endif
