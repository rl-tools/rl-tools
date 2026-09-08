#pragma once

namespace rl_tools::rl::environments::l2f_visual::training{
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<float>;
    using TI = typename ::DEVICE::index_t;
    using RNG = rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024>>;
    using RNG_GPU = typename ::DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

    namespace l2f = rlt::rl::environments::l2f;
    namespace obs = l2f::observation;

    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 10000;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    struct DOMAIN_RANDOMIZATION_OPTIONS {
        static constexpr bool THRUST_TO_WEIGHT = true;
        static constexpr bool MASS = false;
        static constexpr bool TORQUE_TO_INERTIA = false;
        static constexpr bool MASS_SIZE_DEVIATION = false;
        static constexpr bool ROTOR_TORQUE_CONSTANT = false;
        static constexpr bool DISTURBANCE_FORCE = false;
        static constexpr bool ROTOR_TIME_CONSTANT = false;
    };
    using PARAMETERS_BASE = l2f::ParametersBase<PARAMETERS_SPEC>;
    using PARAMETERS_IMU = l2f::ParametersIMU<l2f::ParametersSpecification<T, TI, PARAMETERS_BASE>>;
    using PARAMETERS_DISTURBANCES = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, PARAMETERS_IMU>>;
    using PARAMETERS_TYPE = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, PARAMETERS_DISTURBANCES>>;

    static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie_openmv;

    static constexpr REWARD_FUNCTION reward_function = {
        false,
        0.10,
        1.00,
        -100.00,
        10.00,
        0.00,
        1.00,
        0.05,
        1.50,
        0.00,
        0.00,
        {0.00, 0.00, 0.00, 0.00},
        {1.50, 1.50, 1.50, 1.50},
        0.00
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
        1.0, 0.0, 0.3, 1.0, 1.0, true, -1, +1,
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
        true, 1.0, 0, 10, 35, 10000, 50000,
    };
    static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
    static constexpr typename PARAMETERS_TYPE::Integration integration = {
        static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
    };
    static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };

    static constexpr T MAX_THRUST_PER_ROTOR = dynamics.rotor_thrust_coefficients[0][0]
                                            + dynamics.rotor_thrust_coefficients[0][1]
                                            + dynamics.rotor_thrust_coefficients[0][2];
    static constexpr T MAX_THRUST = static_cast<T>(PARAMETERS_SPEC::N) * MAX_THRUST_PER_ROTOR;

    static constexpr T ROTOR_ARM = dynamics.rotor_positions[0][0] < 0 ? -dynamics.rotor_positions[0][0] : dynamics.rotor_positions[0][0];

    static constexpr T DISTURBANCE_FRACTION = static_cast<T>(0.0);
    static constexpr T DISTURBANCE_FORCE_STD = DISTURBANCE_FRACTION * MAX_THRUST;
    static constexpr T DISTURBANCE_TORQUE_STD = DISTURBANCE_FRACTION * MAX_THRUST * ROTOR_ARM;
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
        {0, DISTURBANCE_FORCE_STD},
        {0, DISTURBANCE_TORQUE_STD}
    };
    static constexpr typename PARAMETERS_TYPE::IMU imu = {
        {{
            {static_cast<T>(0), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}
        }},
        {{
            {static_cast<T>(0), static_cast<T>(0)},
            {static_cast<T>(0.02), static_cast<T>(0), static_cast<T>(0)}
        }}
    };
    static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
        1.7, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    static constexpr PARAMETERS_TYPE nominal_parameters = { {{{dynamics, integration, mdp}, imu}, disturbances}, domain_randomization };

    static constexpr TI ACTION_HISTORY_LENGTH = 8;

    struct STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = training::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_BASE_LA = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
        using STATE_BASE_LAH = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE_LA>>;
        using STATE_BASE_GB = l2f::StateIMU<T, TI, STATE_BASE_LAH>;
        using STATE_BASE_MAHONY = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_BASE_GB>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE_MAHONY>>>>>>;
        using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
                obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
                obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
                obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
                obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI,
                obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI,
                obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI,
                obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto PARAMETER_VALUES = nominal_parameters;
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

    using ACTOR_STATE_OBS = obs::OrientationWorldZ<obs::OrientationWorldZSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>;
    static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM;

    static constexpr TI N_TOTAL_SCENES = 25;
    static constexpr TI N_ACTIVE_SCENES = 2;
    static constexpr TI N_ENVIRONMENTS_PER_SCENE = 16;
    static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI GRADIENT_ACCUMULATION_ROLLOUTS = 1;
    static constexpr TI CAM_WIDTH = 80;
    static constexpr TI CAM_HEIGHT = 50;
    static constexpr TI NUM_PROBES = 64;
    static constexpr T CAMERA_FOV = 79.6;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(5.0);
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X = static_cast<T>(0.01);
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y = static_cast<T>(0.01);
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z = static_cast<T>(0.01);
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
    using RENDER_SHADING = rlt::rendering::raytracing::High;
    static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
    static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
    static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
    static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
    static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
    static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);
    static_assert(RENDER_SHUTTER_FRACTION_MIN >= static_cast<T>(0) && RENDER_SHUTTER_FRACTION_MIN <= RENDER_SHUTTER_FRACTION_MAX && RENDER_SHUTTER_FRACTION_MAX <= static_cast<T>(1), "Invalid l2f_visual training shutter fraction range");

    static constexpr TI FRAME_STACK_N = 10;
    static constexpr TI FRAME_STACK_STRIDE = 10;
    static constexpr TI ROLLOUT_STEPS_PER_ENV = 320;
    static constexpr TI SCENE_SET_MIN_STEPS = 2048;
    static constexpr TI ROLLOUTS_PER_SCENE_SET = (SCENE_SET_MIN_STEPS + ROLLOUT_STEPS_PER_ENV - 1) / ROLLOUT_STEPS_PER_ENV;

    static constexpr TI INDOOR_POSITION_DIM = 3;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
    static constexpr T OBSERVATION_NOISE_STD = 0.0;

    static constexpr TI COMBINED_IMG_C_LOGICAL = 3 * (FRAME_STACK_N + 1);
    static constexpr TI COMBINED_IMG_C = (COMBINED_IMG_C_LOGICAL + 7) & ~((TI)7);
    static constexpr TI COMBINED_OBS_DIM = CAM_HEIGHT * CAM_WIDTH * COMBINED_IMG_C;
    static constexpr TI EXTRACK_SAVE_INTERVAL_PPO_STEPS = 640;
    static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE = (EXTRACK_SAVE_INTERVAL_PPO_STEPS + ROLLOUTS_PER_SCENE_SET - 1) / ROLLOUTS_PER_SCENE_SET;
    static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS = 2 * EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE;
    static constexpr TI VIDEO_SAVE_INTERVAL_SCENE_SETS = EXTRACK_SAVE_INTERVAL_SCENE_SETS;
    static constexpr TI CHECKPOINT_CADENCE_SCENE_SETS = EXTRACK_SAVE_INTERVAL_SCENE_SETS;
    static constexpr TI REWARD_COMPONENT_LOG_INTERVAL_PPO_STEPS = 20;
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
    static constexpr bool EXPORT_CHECKPOINT_TAR = true;
#else
    static constexpr bool EXPORT_CHECKPOINT_TAR = false;
#endif
    static constexpr bool EXPORT_CHECKPOINT_CODE = false;
    static constexpr TI N_EXAMPLES = 512;
    static constexpr TI REDUCED_BATCH_SIZE = 2;
    static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
    static constexpr TI TRAJECTORY_NUM_ENVS = 10;
    static constexpr TI TRAJECTORY_MAX_EPISODES = 10;

    struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr T ALPHA = 1e-4;
    };

    template<typename ENVIRONMENT>
    struct LoopParameters: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
        static constexpr TI BATCH_SIZE = 1024;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI ACTOR_CNN_CHANNEL_MULTIPLIER = 2;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
        static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
        static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ROLLOUT_STEPS_PER_ENV;
        static constexpr TI N_ENVIRONMENTS = training::N_ENVIRONMENTS;
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMOKE
        static constexpr TI TOTAL_STEP_LIMIT = 2 * ROLLOUTS_PER_SCENE_SET * N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV - 1;
#else
        static constexpr TI TOTAL_STEP_LIMIT = 1000000000;
#endif
        static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV) + 1;
        static constexpr TI EPISODE_STEP_LIMIT = training::EPISODE_STEP_LIMIT;
        static constexpr TI COMBINED_IMG_C = training::COMBINED_IMG_C;
        static constexpr TI STATE_OBS_DIM = training::STATE_OBS_DIM;
        using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        static constexpr bool NORMALIZE_OBSERVATIONS = false;
        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.005;
            static constexpr TI N_EPOCHS = 1;
            static constexpr T GAMMA = 0.99;
            static constexpr T LAMBDA = 0.95;
            static constexpr T EPSILON_CLIP = 0.2;
            static constexpr T INITIAL_ACTION_STD = 0.5;
        };
    };

    static constexpr T EPISODE_END_REASON_NONE = 0;
    static constexpr T EPISODE_END_REASON_TERMINATED = 1;
    static constexpr T EPISODE_END_REASON_TIME_LIMIT = 2;
    static constexpr T EPISODE_END_REASON_SCENE_BOUNDARY = 3;

}
