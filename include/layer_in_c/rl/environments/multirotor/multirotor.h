#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H

#include <layer_in_c/dependencies.h>
#include <layer_in_c/devices.h>
#include <layer_in_c/utils/generic/math.h>

namespace layer_in_c::rl::environments::multirotor {
    constexpr int STATE_DIM = 13;
    constexpr int ACTION_DIM = 4;

    template <typename T, int N>
    class Parameters {
    public:
        class Reward{
            T position;
            T orientation;
            T linear_velocity;
            T angular_velocity;
            T actions_baseline;
            T actions;
        };
        class ActionLimit{
            T min;
            T max;
        };
        class Dynamics{
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
        Reward reward;
        Dynamics dynamics;
        ActionLimit action_limit;
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
        T state[STATE_DIM];
    };

}

namespace layer_in_c::rl::environments{
    template <typename DEVICE, typename SPEC>
    struct Multirotor{
        static constexpr size_t OBSERVATION_DIM = multirotor::STATE_DIM;
        static constexpr size_t ACTION_DIM = 4;
        using State = multirotor::State<typename SPEC::T>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
    };
}

#endif
