#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_MULTIROTOR_H

namespace backprop_tools::rl::environments::multirotor {
    template <typename T, typename TI, TI N, typename T_REWARD_FUNCTION>
    struct Parameters {
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
            using REWARD_FUNCTION = T_REWARD_FUNCTION;
            struct Initialization{
                T guidance;
                T max_position;
                T max_angle;
                T max_linear_velocity;
                T max_angular_velocity;
            };
            struct Termination{
                bool enabled = false;
                T position_threshold;
                T linear_velocity_threshold;
                T angular_velocity_threshold;
            };
            Initialization init;
            Termination termination;
            REWARD_FUNCTION reward;
        };
        Dynamics dynamics;
        Integration integration;
        MDP mdp;
    };
    struct StaticParameters{};
    template <typename T_T, typename T_TI, typename T_PARAMETERS, typename T_STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
        using STATIC_PARAMETERS = T_STATIC_PARAMETERS;
    };

    template <typename T, typename TI, TI STATE_DIM>
    struct State{
        static constexpr TI DIM = STATE_DIM;
        T state[DIM];
    };

}

namespace backprop_tools::rl::environments{
    template <typename SPEC>
    struct Multirotor{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using REWARD_FUNCTION = typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION;
        static constexpr TI STATE_DIM = 13;
        static constexpr TI ACTION_DIM = 4;

        static constexpr bool REQUIRES_OBSERVATION = false;
        static constexpr TI OBSERVATION_DIM = STATE_DIM;
        using State = multirotor::State<T, TI, STATE_DIM>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        typename SPEC::PARAMETERS parameters;
    };
}

#endif
