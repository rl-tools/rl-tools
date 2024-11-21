#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_MULTIROTOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_MULTIROTOR_H

#include "../../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template <typename T_T, typename T_TI, T_TI T_N, typename T_REWARD_FUNCTION, typename REGISTRY, REGISTRY T_MODEL>
    struct ParametersBaseSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI N = T_N;
        using REWARD_FUNCTION = T_REWARD_FUNCTION;
        static constexpr REGISTRY MODEL = T_MODEL;
    };
    template <typename T_SPEC>
    struct ParametersBase{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        static constexpr typename SPEC::TI N = SPEC::N;

        struct Dynamics{
            struct ActionLimit{
                T min;
                T max;
            };
            T rotor_positions[N][3];
            T rotor_thrust_directions[N][3];
            T rotor_torque_directions[N][3];
            T rotor_thrust_coefficients[3];
            T rotor_torque_constant;
            T mass;
            T gravity[3];
            T J[3][3];
            T J_inv[3][3];
            T motor_time_constant;
            T hovering_throttle_relative; // relative to the action limits [0, 1]
            ActionLimit action_limit;
        };
        struct Integration{
            T dt;
        };
        struct MDP{
            using REWARD_FUNCTION = typename SPEC::REWARD_FUNCTION;
            struct Initialization{
                T guidance;
                T max_position;
                T max_angle;
                T max_linear_velocity;
                T max_angular_velocity;
                bool relative_rpm; //(specification from -1 to 1)
                T min_rpm; // -1 for default limit when relative_rpm is true, -1 if relative_rpm is false
                T max_rpm; //  1 for default limit when relative_rpm is true, -1 if relative_rpm is false
            };
            struct Termination{
                bool enabled = false;
                T position_threshold;
                T linear_velocity_threshold;
                T angular_velocity_threshold;
                T position_integral_threshold;
                T orientation_integral_threshold;
            };
            struct ObservationNoise{
                T position;
                T orientation;
                T linear_velocity;
                T angular_velocity;
                T imu_acceleration;
            };
            struct ActionNoise{
                T normalized_rpm; // std of additive gaussian noise onto the normalized action (-1, 1)
            };
            Initialization init;
            REWARD_FUNCTION reward;
            ObservationNoise observation_noise;
            ActionNoise action_noise;
            Termination termination;
        };
        struct DomainRandomization{
            T thrust_to_weight_min;
            T thrust_to_weight_max;
            T thrust_to_weight_by_torque_to_inertia_min;
            T thrust_to_weight_by_torque_to_inertia_max;
            T mass_min;
            T mass_max;
            T mass_size_deviation; // percentage variation around the nominal value derived from the mass scale and the sampled thrust to weight ratio
            T motor_time_constant;
            T rotor_torque_constant;
        };
        Dynamics dynamics;
        Integration integration;
        MDP mdp;
        DomainRandomization domain_randomization;
    };
    template <typename T, typename TI, typename T_NEXT_COMPONENT>
    struct ParametersDisturbances: T_NEXT_COMPONENT{
        struct Disturbances{
            struct UnivariateGaussian{
                T mean;
                T std;
            };
            UnivariateGaussian random_force;
            UnivariateGaussian random_torque;
        };
        Disturbances disturbances;
    };

    template <typename T, typename TI, typename T_NEXT_COMPONENT>
    struct ParametersDomainRandomization: T_NEXT_COMPONENT{
        struct DomainRandomization{
            struct UnivariateGaussian{
                T mean;
                T std;
            };
            UnivariateGaussian J_factor;
            UnivariateGaussian mass_factor;
        };
        DomainRandomization domain_randomization;
    };


//    enum class LatentStateType{
//        Empty,
//        RandomForce
//    };
//    enum class StateType{
//        Base,
//        BaseRotors,
//        BaseRotorsHistory,
//    };
//    enum class ObservationType{
//        Normal,
//        DoubleQuaternion,
//        RotationMatrix
//    };
    namespace observation{
        template <typename T_TI>
        struct LastComponent{
            static constexpr T_TI DIM = 0;
        };
        template <typename T_TI>
        struct NONE{
            static constexpr T_TI DIM = 0;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PoseIntegralSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct PoseIntegral{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 2;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecificationPrivileged: PositionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct Position{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecificationPrivileged: OrientationQuaternionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationQuaternion{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecificationPrivileged: OrientationRotationMatrixSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationRotationMatrix{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 9;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecificationPrivileged: LinearVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct LinearVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecificationPrivileged: AngularVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct AngularVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct IMUAccelerometerSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct IMUAccelerometerSpecificationPrivileged: IMUAccelerometerSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct IMUAccelerometer{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct MagnetometerSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct MagnetometerSpecificationPrivileged: IMUAccelerometerSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct Magnetometer{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 2;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RotorSpeedsSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RotorSpeeds{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ActionHistorySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        };
        template <typename SPEC>
        struct ActionHistory{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = SPEC::HISTORY_LENGTH;
            static constexpr TI ACTION_DIM = 4;
            static constexpr TI CURRENT_DIM = ACTION_DIM * HISTORY_LENGTH;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RandomForceSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RandomForce{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 6;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
    }


    template <typename T_T, typename T_TI>
    struct StateBase{
        using T = T_T;
        using TI = T_TI;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI DIM = 13;
        T position[3];
        T orientation[4];
        T linear_velocity[3];
        T angular_velocity[3];
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StateLinearAcceleration: T_NEXT_COMPONENT{
        using T = T_T;
        T linear_acceleration[3]; // this is just to save computation when simulating IMU measurements. Wihtout this we would need to recalculate the acceleration in the observation operation. This is not part of the minimal state in the sense that the transition dynamics are independent of the acceleration given the other parts of the state and the action
    };

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StatePoseErrorIntegral: T_NEXT_COMPONENT{
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
        using T = T_T;
        using TI = T_TI;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI DIM = 2;
        T position_integral;
        T orientation_integral;
    };
    template <typename T_T, typename T_TI, bool T_CLOSED_FORM, typename T_NEXT_COMPONENT>
    struct StateRotors: T_NEXT_COMPONENT{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
        static constexpr bool CLOSED_FORM = T_CLOSED_FORM;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI PARENT_DIM = NEXT_COMPONENT::DIM;
        static constexpr TI DIM = PARENT_DIM + 4;
        T rpm[4];
    };
    template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, bool T_CLOSED_FORM, typename T_NEXT_COMPONENT>
    struct StateRotorsHistory: StateRotors<T_T, T_TI, T_CLOSED_FORM, T_NEXT_COMPONENT>{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = StateRotors<T, TI, T_CLOSED_FORM, T_NEXT_COMPONENT>;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        static constexpr TI PARENT_DIM = StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>::DIM;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI DIM = PARENT_DIM + HISTORY_LENGTH * ACTION_DIM;
        TI current_step;
        T action_history[HISTORY_LENGTH][4];
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StateRandomForce: T_NEXT_COMPONENT{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 6 + NEXT_COMPONENT::DIM;
        T force[3];
        T torque[3];
    };

    template <typename T, typename TI, typename T_PARAMETERS, typename T_PARAMETER_VALUES>
    struct StaticParametersDefault{
//        static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//        static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//        static constexpr LatentStateType LATENT_STATE_TYPE = LatentStateType::Empty;
//        static constexpr StateType STATE_TYPE = StateType::Base;
//        static constexpr ObservationType OBSERVATION_TYPE = ObservationType::Normal;
        using STATE_TYPE = StateBase<T, TI>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                                 observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                 observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                 observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::NONE<TI>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = T_PARAMETERS;
        static constexpr auto PARAMETER_VALUES = T_PARAMETER_VALUES::VALUES;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using STATIC_PARAMETERS = T_PARAMETERS;
        using STATE_TYPE = typename STATIC_PARAMETERS::STATE_TYPE;
        using OBSERVATION_TYPE = typename STATIC_PARAMETERS::OBSERVATION_TYPE;
        using OBSERVATION_TYPE_PRIVILEGED = typename STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
        using PARAMETERS = typename STATIC_PARAMETERS::PARAMETERS;
        static constexpr auto PARAMETER_VALUES = STATIC_PARAMETERS::PARAMETER_VALUES;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    template <typename T_SPEC>
    struct Multirotor{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
//        using PARAMETERS = typename SPEC::PARAMETERS;
        using Parameters = typename SPEC::PARAMETERS;
//        using REWARD_FUNCTION = typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION;
//        static constexpr TI STATE_DIM = 13;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI EPISODE_STEP_LIMIT = 500;

        static constexpr TI ACTION_HISTORY_LENGTH = SPEC::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH;

//        static constexpr multirotor::LatentStateType LATENT_STATE_TYPE = SPEC::STATIC_PARAMETERS::LATENT_STATE_TYPE;
//        static constexpr multirotor::StateType STATE_TYPE = SPEC::STATIC_PARAMETERS::STATE_TYPE;
        using State = typename SPEC::STATE_TYPE;
        using Observation = typename SPEC::OBSERVATION_TYPE;
        using ObservationPrivileged = typename SPEC::OBSERVATION_TYPE_PRIVILEGED;

//        using LatentState = utils::typing::conditional_t<
//            LATENT_STATE_TYPE == multirotor::LatentStateType::Empty,
//            multirotor::StateLatentEmpty<T, TI>,
//            multirotor::StateLatentRandomForce<T, TI>
//        >;
//        using State = utils::typing::conditional_t<
//            STATE_TYPE == multirotor::StateType::Base,
//            multirotor::StateBase<T, TI, LatentState>,
//            utils::typing::conditional_t<
//                STATE_TYPE == multirotor::StateType::BaseRotors,
//                multirotor::StateBaseRotors<T, TI, LatentState>,
//                multirotor::StateBaseRotorsHistory<T, TI, SPEC::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH, LatentState>
//            >
//        >;

//        static constexpr TI OBSERVATION_DIM_BASE = 3 + 3 + 3;
//        static constexpr TI OBSERVATION_DIM_ORIENTATION = OBSERVATION_TYPE == multirotor::ObservationType::Normal ? 4 : (OBSERVATION_TYPE == multirotor::ObservationType::DoubleQuaternion ? (2*4) : (9));
//        static constexpr TI OBSERVATION_DIM_ACTION_HISTORY = (STATE_TYPE == multirotor::StateType::BaseRotorsHistory) * ACTION_DIM * ACTION_HISTORY_LENGTH;
//        static constexpr TI OBSERVATION_DIM = OBSERVATION_DIM_BASE + OBSERVATION_DIM_ORIENTATION + OBSERVATION_DIM_ACTION_HISTORY;
//        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = STATE_TYPE == multirotor::StateType::Base || LATENT_STATE_TYPE == multirotor::LatentStateType::RandomForce;
//        static constexpr TI OBSERVATION_DIM_PRIVILEGED_LATENT_STATE = (LATENT_STATE_TYPE == multirotor::LatentStateType::RandomForce ? LatentState::DIM : 0);
//        static constexpr TI OBSERVATION_DIM_PRIVILEGED = PRIVILEGED_OBSERVATION_AVAILABLE ? (
//                OBSERVATION_DIM_BASE + OBSERVATION_DIM_ORIENTATION
//                + (STATE_TYPE == multirotor::StateType::BaseRotors || STATE_TYPE == multirotor::StateType::BaseRotorsHistory ? ACTION_DIM : 0)
//                + OBSERVATION_DIM_PRIVILEGED_LATENT_STATE
//        ) : 0;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = !rl_tools::utils::typing::is_same_v<typename SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED, l2f::observation::NONE<TI>>;
//        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        Parameters parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "parameters/registry.h"

#endif