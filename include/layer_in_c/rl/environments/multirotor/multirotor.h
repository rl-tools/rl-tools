#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H

namespace layer_in_c::rl::environments::multirotor {
    template <typename T, typename TI, TI N>
    class Parameters {
    public:
        struct Dynamics{
            struct ActionLimit{
                T min;
                T max;
            };
            T rotor_positions[N][3];
            T rotor_thrust_directions[N][3];
            T rotor_torque_directions[N][3];
            T thrust_constants[3];
            T torque_constant;
            T mass;
            T gravity[3];
            T J[3][3];
            T J_inv[3][3];
            ActionLimit action_limit;
        };
        struct Integration{
            T dt;
        };
        struct MDP{
            struct Initialization{
                T max_position;
                T max_linear_velocity;
                T max_angular_velocity;
            };
            struct Reward{
                T position;
                T orientation;
                T linear_velocity;
                T angular_velocity;
                T action_baseline;
                T action;
            };
            Initialization init;
            Reward reward;
        };
        Dynamics dynamics;
        Integration integration;
        MDP mdp;
    };
    struct StaticParameters{};
    template <typename T_T, typename T_TI, typename T_STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using STATIC_PARAMETERS = T_STATIC_PARAMETERS;
    };

    template <typename T, typename TI, TI STATE_DIM>
    struct State{
        static constexpr TI DIM = STATE_DIM;
        T state[DIM];
    };

}

namespace layer_in_c::rl::environments{
    template <typename SPEC>
    struct Multirotor{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI STATE_DIM = 13;
        static constexpr TI ACTION_DIM = 4;

        static constexpr bool REQUIRES_OBSERVATION = false;
        static constexpr TI OBSERVATION_DIM = STATE_DIM;
        using State = multirotor::State<T, TI, STATE_DIM>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        multirotor::Parameters<T, TI, ACTION_DIM> parameters;
    };
}

#endif
