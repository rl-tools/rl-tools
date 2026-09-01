#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_PRESETS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_PRESETS_H

#include "world.h"
#include "../l2f/parameters/registry.h"
#include "../l2f/parameters/init/default.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::presets {

    namespace x500_fpv {
        template <typename T_T, typename T_TI>
        struct DynamicsStaticParameters {
            using T = T_T;
            using TI = T_TI;
            using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
            static constexpr TI EPISODE_STEP_LIMIT = 500;
            using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
            using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
            static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
            using STATE_PLAIN = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
            using STATE_TYPE = l2f::StateRenderRotorPhase<l2f::StateSpecification<T, TI, STATE_PLAIN>>;
            using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                    l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                    l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                    l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::x500, PARAMETERS_SPEC>;
            static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
            static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
            static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
            static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
            static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
            static constexpr T STATE_LIMIT_POSITION_X = 100000;
            static constexpr T STATE_LIMIT_POSITION_Y = 100000;
            static constexpr T STATE_LIMIT_POSITION_Z = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
        };
    }

    // reference FPV configuration on the x500 platform: spinning-prop render state
    // (StateRenderRotorPhase) and the drone entity attached to its own cameras — set
    // World::drone_asset_path to an x500 body/prop_* assembly before init. Derive-and-shadow
    // for further customization, like every hyperdrone Specification
    template <typename T_T, typename T_TI>
    struct X500FPV: Specification<T_T, T_TI, x500_fpv::DynamicsStaticParameters<T_T, T_TI>> {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool SELF_VISIBLE = true;
    };

    namespace x500_fpv_imu {
        // x500 FPV variant stepping at IMU rate: the integration dt is the IMU sampling period
        // and StateIMU produces one noised accelerometer/gyroscope sample per step
        template <typename T_T, typename T_TI>
        struct DynamicsStaticParameters {
            using T = T_T;
            using TI = T_TI;
            using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
            static constexpr TI IMU_RATE = 200; // Hz
            static constexpr TI EPISODE_LENGTH_SECONDS = 10;
            static constexpr TI EPISODE_STEP_LIMIT = EPISODE_LENGTH_SECONDS * IMU_RATE;
            using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
            using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersIMU<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>>>;
            static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
            using STATE_IMU = l2f::StateIMU<T, TI, STATE_BASE>;
            using STATE_PLAIN = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_IMU>>>>>>;
            using STATE_TYPE = l2f::StateRenderRotorPhase<l2f::StateSpecification<T, TI, STATE_PLAIN>>;
            using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                    l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                    l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                    l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::x500, PARAMETERS_SPEC>;
            static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)1 / (T)IMU_RATE};
            // rest start: level attitude, zero velocities at t=0 — the benchmark protocol
            // evaluates the pose relative to the initial pose and lets the estimator assume
            // zero tilt and zero velocity instead of receiving any ground truth
            static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
                    1,   // guidance (always guided: origin, level, at rest)
                    0,   // max_position
                    0,   // max_angle
                    0,   // max_linear_velocity
                    0,   // max_angular_velocity
                    true,// relative_rpm
                    0,   // min_rpm
                    0,   // max_rpm
            };
            // MEMS-class per-sample white noise at IMU_RATE (noise density * sqrt(rate))
            static constexpr T GYRO_NOISE_STD = 0.07;           // rad/s
            static constexpr T ACCELEROMETER_NOISE_STD = 0.28;  // m/s^2
            static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
            static constexpr typename PARAMETERS_TYPE::IMU imu = {
                    {{ // accelerometer error
                            {0, ACCELEROMETER_NOISE_STD},     // noise: mean, std [m/s^2]
                            {(T)0.2, (T)0, (T)0.001}          // bias: turn-on [m/s^2], tau <= 0: random walk, increment density [m/s^2/sqrt(s)]
                    }},
                    {{ // gyro error
                            {0, GYRO_NOISE_STD},              // noise: mean, std [rad/s]
                            {(T)0.02, (T)0, (T)0.0006}        // bias: turn-on [rad/s], tau <= 0: random walk, increment density [rad/s/sqrt(s)]
                    }}
            };
            static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
            static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{{dynamics, integration, mdp}, imu}, disturbances};
            static constexpr T STATE_LIMIT_POSITION_X = 100000;
            static constexpr T STATE_LIMIT_POSITION_Y = 100000;
            static constexpr T STATE_LIMIT_POSITION_Z = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
        };
    }

    template <typename T_T, typename T_TI>
    struct X500FPVIMU: Specification<T_T, T_TI, x500_fpv_imu::DynamicsStaticParameters<T_T, T_TI>> {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool SELF_VISIBLE = true;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
