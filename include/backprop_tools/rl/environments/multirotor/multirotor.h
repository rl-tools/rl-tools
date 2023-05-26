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
            T rpm_time_constant;
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


    enum class StateType{
        Normal,
        RPM
    };
    enum class ObservationType{
        Normal,
        DoubleQuaternion,
        RotationMatrix
    };

    struct StaticParametersDefault{
        static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
        static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
        static constexpr StateType STATE_TYPE = StateType::Normal;
        static constexpr ObservationType OBSERVATION_TYPE = ObservationType::Normal;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS, typename T_STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
        using STATIC_PARAMETERS = T_STATIC_PARAMETERS;
    };


    template <typename T, typename TI>
    struct StateNormal{
        static constexpr TI DIM = 13;
        T state[DIM];
    };
    template <typename T, typename TI>
    struct StateRPM{
        static constexpr TI DIM = 13 + 4;
        T state[DIM];
    };
}

namespace backprop_tools::rl::environments{
    template <typename SPEC>
    struct Multirotor{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using REWARD_FUNCTION = typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION;
//        static constexpr TI STATE_DIM = 13;
        static constexpr TI ACTION_DIM = 4;

        static constexpr multirotor::StateType STATE_TYPE = SPEC::STATIC_PARAMETERS::STATE_TYPE;
        static constexpr multirotor::ObservationType OBSERVATION_TYPE = SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE;

        static constexpr TI OBSERVATION_DIM = OBSERVATION_TYPE == multirotor::ObservationType::Normal ? 13 : (OBSERVATION_TYPE == multirotor::ObservationType::DoubleQuaternion ? (13 + 4) : (13 - 4 + 9));
        using State = utils::typing::conditional_t<
                STATE_TYPE == multirotor::StateType::Normal,
                multirotor::StateNormal<T, TI>, multirotor::StateRPM<T, TI>>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        typename SPEC::PARAMETERS parameters;
    };
}

#endif
