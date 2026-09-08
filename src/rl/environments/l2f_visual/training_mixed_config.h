#pragma once

namespace rl_tools::rl::environments::l2f_visual::training_mixed{
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
    using LOGGER = rlt::devices::logging::CPU;
#endif
    using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
    using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
    using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;

    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<float>;
    using TI = typename DEVICE::index_t;
    using RNG = rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024>>;
    using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
    static constexpr bool MIXED_TRAINING = true;
#ifdef RL_TOOLS_L2F_VISUAL_EVALUATION
    static constexpr bool EVALUATION_BUILD = true;
#else
    static constexpr bool EVALUATION_BUILD = false;
#endif

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
    using PARAMETERS_RANDOMIZED = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, PARAMETERS_DISTURBANCES>>;

    using PARAMETERS_TYPE = l2f::ParametersCTBRController<l2f::ParametersSpecification<T, TI, PARAMETERS_RANDOMIZED>>;

    static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie_openmv;

    static constexpr REWARD_FUNCTION reward_function = {
        false, // bool non_negative;
        0.10,  // scale;
        1.00,  // constant;
        -100.00, // termination_penalty;
        10.00, // position;
        0.00,  // position_clip;
        1.00,  // orientation;
        0.05,  // linear_velocity;
        1.50,  // angular_velocity;
        0.00,  // linear_acceleration;
        0.00,  // angular_acceleration;
        {0.00, 0.00, 0.00, 0.00}, // action[ACTION_DIM]
        {1.50, 1.50, 1.50, 1.50}, // d_action[ACTION_DIM]
        0.00   // position_error_integral
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Initialization initialization = {
        MIXED_TRAINING ? 0.2f : 1.0f, MIXED_TRAINING ? 0.1f : 0.0f, 0.02, 0.05, 0.01, true, static_cast<T>(2 * l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>.hovering_throttle_relative - 1), static_cast<T>(2 * l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>.hovering_throttle_relative - 1),
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
        true, 1.0, 0, 10, 35, 10000, 50000,
    };
    static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
    static constexpr typename PARAMETERS_TYPE::Integration integration = {
        static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
    };
    static constexpr typename PARAMETERS_TYPE::MDP mdp = { initialization, reward_function, {}, {}, termination };
    static constexpr T MAX_THRUST_PER_ROTOR = dynamics.rotor_thrust_coefficients[0][0]
                                            + dynamics.rotor_thrust_coefficients[0][1]
                                            + dynamics.rotor_thrust_coefficients[0][2];
    static constexpr T MAX_THRUST = static_cast<T>(PARAMETERS_SPEC::N) * MAX_THRUST_PER_ROTOR;
    static constexpr T ROTOR_ARM = dynamics.rotor_positions[0][0] < 0 ? -dynamics.rotor_positions[0][0] : dynamics.rotor_positions[0][0];
    static constexpr T DISTURBANCE_FRACTION = static_cast<T>(0.0);
    static constexpr T DISTURBANCE_FORCE_STD = DISTURBANCE_FRACTION * MAX_THRUST;
    static constexpr T DISTURBANCE_TORQUE_STD = DISTURBANCE_FRACTION * MAX_THRUST * ROTOR_ARM;
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
        {0, DISTURBANCE_FORCE_STD},  // random_force
        {0, DISTURBANCE_TORQUE_STD}  // random_torque (z axis is internally scaled by 1/100 in sample_initial_state)
    };
    static constexpr typename PARAMETERS_TYPE::IMU imu = {
        {{ // accelerometer error
            {static_cast<T>(0), static_cast<T>(0)}, // noise: mean, std
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)} // bias: init_max, tau, sigma
        }},
        {{ // gyro error
            {static_cast<T>(0), static_cast<T>(0)}, // noise: mean, std
            {static_cast<T>(0.02), static_cast<T>(0), static_cast<T>(0)} // bias: init_max, tau (<=0: random walk), increment density
        }}
    };
    static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
        1.7, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    static constexpr typename PARAMETERS_TYPE::CTBRController ctbr_controller = {
        0, 1, {4, 4, 2}, {216, 216, 96}, {5.76, 5.76, 1.6}, {0.003, 0.003, 0.0012}
    };
    static constexpr PARAMETERS_TYPE nominal_parameters = { { {{{dynamics, integration, mdp}, imu}, disturbances}, domain_randomization }, ctbr_controller };

    static constexpr TI ACTION_HISTORY_LENGTH = 8;

    struct STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::CTBR;
        static constexpr TI N_SUBSTEPS = 10;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = training_mixed::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_BASE_LA = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
        using STATE_BASE_LAH = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE_LA>>;
        using STATE_BASE_GB = l2f::StateIMU<T, TI, STATE_BASE_LAH>;
        using STATE_BASE_MAHONY = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_BASE_GB>>;
        using STATE_BASE_CTBR = l2f::StateCTBRController<l2f::StateSpecification<T, TI, STATE_BASE_MAHONY>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE_CTBR>>>>>>;
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

    using ACTOR_ACCEL_OBS = obs::LinearAccelerationBodyFrameHistory<obs::LinearAccelerationBodyFrameHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>;
    using ACTOR_STATE_OBS = obs::OrientationWorldZ<obs::OrientationWorldZSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, ACTOR_ACCEL_OBS>>>>>>;
    static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM + 3;
    using INERTIAL_SPEC = rlt::rl::environments::l2f_visual::inertial_velocity::Specification<TYPE_POLICY, TI>;
    using INERTIAL_STATE = rlt::rl::environments::l2f_visual::inertial_velocity::State<INERTIAL_SPEC>;

    static constexpr TI N_TOTAL_SCENES = 25;
    static constexpr TI N_ACTIVE_SCENES = EVALUATION_BUILD ? 2 : 8;
    static constexpr TI N_ENVIRONMENTS_PER_SCENE = 16;
    static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
namespace mixed_imitation = rlt::rl::algorithms::ppo::mixed_imitation;
    struct MIXED_PARAMETERS: mixed_imitation::DefaultParameters<TYPE_POLICY, TI>{
        static constexpr TI IMITATION_ENVIRONMENTS_PER_SCENE = 8;
        static constexpr TI TEACHER_FORCING_UPDATES = 2000;
        static constexpr T IMITATION_WEIGHT = 1;
        static constexpr T OUTPUT_GRADIENT_NORM_RATIO = 50;
        static constexpr T MAX_IMITATION_WEIGHT = 100000000;
        static constexpr TI TEACHER_EPISODE_STEPS = 200;
        static constexpr T TEACHER_ACTION_NOISE_STD = 0.01;
        static constexpr TI TEACHER_PURE_UPDATES = 500;
        using BALANCING_MODE = rlt::Mode<mixed_imitation::FixedNormRatio<>>;
    };
    static_assert(!MIXED_TRAINING || (MIXED_PARAMETERS::IMITATION_ENVIRONMENTS_PER_SCENE > 0 && MIXED_PARAMETERS::IMITATION_ENVIRONMENTS_PER_SCENE < N_ENVIRONMENTS_PER_SCENE));
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
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = MIXED_TRAINING ? static_cast<T>(0) : static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
    using RENDER_SHADING = rlt::rendering::raytracing::High;
    static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
    static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
    static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
    static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
    static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
    static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);
    static_assert(RENDER_SHUTTER_FRACTION_MIN >= static_cast<T>(0) && RENDER_SHUTTER_FRACTION_MIN <= RENDER_SHUTTER_FRACTION_MAX && RENDER_SHUTTER_FRACTION_MAX <= static_cast<T>(1), "Invalid l2f_visual training shutter fraction range");

    using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, RENDER_SHADING, RENDER_ENABLE_MOTION_BLUR, RENDER_MOTION_BLUR_SAMPLES, RENDER_ENABLE_ANTI_ALIASING, RENDER_ANTI_ALIASING_GRID_SIZE>;
    static_assert(VISUAL_SPEC::RENDERER_SPEC::SHADING::PBR_SHADING, "l2f visual training must use the High renderer profile");
    using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
    using CAMERA_DATA = rlt::rendering::raytracing::Camera<T>;
    using LEVEL_CAMERA_SPEC = rlt::rl::environments::l2f_visual::level_camera::Specification<TYPE_POLICY, TI, CAM_WIDTH, CAM_HEIGHT>;
    using LEVEL_CAMERA = rlt::rl::environments::l2f_visual::level_camera::Camera<LEVEL_CAMERA_SPEC>;
    static constexpr bool RENDER_MOTION_BLUR_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_MOTION_BLUR;
    static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_ANTI_ALIASING;

    static constexpr TI ENV_GRID_SIDE = 4;
    static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
    static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
    static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "ENV_GRID_SIDE^2 must equal N_ENVIRONMENTS_PER_SCENE for the mosaic layout");

    static constexpr TI FRAME_STACK_N = 1;
    static constexpr TI FRAME_STACK_STRIDE = 2;
    static constexpr TI ROLLOUT_STEPS_PER_ENV = EVALUATION_BUILD ? 16 : 4;
    static constexpr TI SCENE_SET_MIN_STEPS = 2048;
    static constexpr TI ROLLOUTS_PER_SCENE_SET = (SCENE_SET_MIN_STEPS + ROLLOUT_STEPS_PER_ENV - 1) / ROLLOUT_STEPS_PER_ENV;
    static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + ROLLOUT_STEPS_PER_ENV;
    static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS * FRAME_STACK_N;
    static constexpr TI COMBINED_IMG_C_LOGICAL = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
    static constexpr TI COMBINED_IMG_C = (COMBINED_IMG_C_LOGICAL + 7) & ~((TI)7);
    static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * 2 * COMBINED_IMG_C;
    static constexpr TI INDOOR_POSITION_DIM = 3;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = MIXED_TRAINING ? 0 : 0.25;
    static constexpr T OBSERVATION_NOISE_STD = 0.0;

    static constexpr TI EXTRACK_SAVE_INTERVAL_PPO_STEPS = 640;
    static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE = (EXTRACK_SAVE_INTERVAL_PPO_STEPS + ROLLOUTS_PER_SCENE_SET - 1) / ROLLOUTS_PER_SCENE_SET;
    static constexpr TI EXTRACK_SAVE_INTERVAL_SCENE_SETS = 2 * EXTRACK_SAVE_INTERVAL_SCENE_SETS_BASE;
    static constexpr TI VIDEO_SAVE_INTERVAL_SCENE_SETS = EXTRACK_SAVE_INTERVAL_SCENE_SETS;
    static constexpr TI CHECKPOINT_CADENCE_SCENE_SETS = MIXED_TRAINING ? 20 : EXTRACK_SAVE_INTERVAL_SCENE_SETS;
    static constexpr TI REWARD_COMPONENT_LOG_INTERVAL_PPO_STEPS = 20;
    static constexpr bool EXPORT_CHECKPOINT_TAR = false;
    static constexpr bool EXPORT_CHECKPOINT_CODE = false;
    static constexpr TI N_EXAMPLES = MIXED_TRAINING ? 2 : 512;
    static constexpr TI REDUCED_BATCH_SIZE = 2;
    static constexpr T TRAINING_TIME_LIMIT_SECONDS = MIXED_TRAINING ? 600 : 0;
    static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
    static constexpr TI TRAJECTORY_NUM_ENVS = 10;
    static constexpr TI TRAJECTORY_MAX_EPISODES = 10;

    struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr T ALPHA = MIXED_TRAINING ? 3e-4 : 1e-4;
    };

    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
        static constexpr TI BATCH_SIZE = MIXED_TRAINING ? 512 : 1024;
        static constexpr TI ACTOR_HIDDEN_DIM = 4;
        static constexpr TI ACTOR_CNN_CHANNEL_MULTIPLIER = 2;
        static constexpr bool ACTOR_CNN_KAIMING_INIT = MIXED_TRAINING;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
        static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
        static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ROLLOUT_STEPS_PER_ENV;
        static constexpr TI N_ENVIRONMENTS = training_mixed::N_ENVIRONMENTS;
        static constexpr TI TOTAL_STEP_LIMIT = 1000000000;
#ifdef RL_TOOLS_L2F_VISUAL_TRAINING_SMALL
        static constexpr TI STEP_LIMIT = 600;
#else
        static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV) + 1;
#endif
        static constexpr TI EPISODE_STEP_LIMIT = training_mixed::EPISODE_STEP_LIMIT;
        using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        static constexpr bool NORMALIZE_OBSERVATIONS = false;
        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.005;
            static constexpr TI N_EPOCHS = 1;
            static constexpr T GAMMA = 0.99;
            static constexpr T LAMBDA = 0.95;
            static constexpr T EPSILON_CLIP = 0.2;
            static constexpr T INITIAL_ACTION_STD = MIXED_TRAINING ? 0.05 : 0.5;
        };
    };

    template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION = true>
    struct ConfigApproximators{
        static constexpr T_TI STEPS = 1;
        static constexpr T_TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;

        static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
        static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH * 2;

        template <typename CAPABILITY>
        struct Actor{
            struct CNN_INITIALIZER_PARAMETERS: rlt::nn::layers::conv2d::KaimingUniformSpecification<T_TYPE_POLICY, T_TI>{
                static constexpr bool INIT_LEGACY = !PARAMETERS::ACTOR_CNN_KAIMING_INIT;
            };
            using CNN_INITIALIZER = rlt::nn::layers::conv2d::KaimingUniform<CNN_INITIALIZER_PARAMETERS>;
            using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
            using STATE_INPUT_SHAPE = rlt::tensor::Shape<T_TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

            using GREY_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 2, 2, 2, 2, 2, 0, 0, rlt::nn::activation_functions::IDENTITY>;
            using GREY = rlt::nn::layers::conv2d::BindConfiguration<GREY_CONFIG>;
            using COST_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 100, 5, 5, 1, 1, 0, 0, rlt::nn::activation_functions::GELU>;
            using COST = rlt::nn::layers::conv2d::BindConfiguration<COST_CONFIG>;
            using POOL_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 50, 2, 2, 2, 2, 0, 0, rlt::nn::activation_functions::IDENTITY>;
            using POOL = rlt::nn::layers::conv2d::BindConfiguration<POOL_CONFIG>;
            using NORMALIZE_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 250, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::SIGMOID>;
            using NORMALIZE = rlt::nn::layers::conv2d::BindConfiguration<NORMALIZE_CONFIG>;
            using DIFFERENCE_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 50, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::IDENTITY>;
            using DIFFERENCE = rlt::nn::layers::conv2d::BindConfiguration<DIFFERENCE_CONFIG>;
            using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>>;
            using PATCH_CONTEXT_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 64, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::GELU, rlt::nn::layers::conv2d::Normalization::BATCH_NORM>;
            using PATCH_CONTEXT = rlt::nn::layers::conv2d::BindConfiguration<PATCH_CONTEXT_CONFIG>;
            using IMAGE_CONTEXT_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 64, 10, 18, 10, 20, 0, 0, rlt::nn::activation_functions::GELU, rlt::nn::layers::conv2d::Normalization::BATCH_NORM>;
            using IMAGE_CONTEXT = rlt::nn::layers::conv2d::BindConfiguration<IMAGE_CONTEXT_CONFIG>;
            using IMAGE_OUTPUT_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 4, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::IDENTITY>;
            using IMAGE_OUTPUT = rlt::nn::layers::conv2d::BindConfiguration<IMAGE_OUTPUT_CONFIG>;
            using PAIR_RESPONSES = rlt::nn_models::sequential::Module<GREY, COST, POOL, NORMALIZE, DIFFERENCE, PATCH_CONTEXT, IMAGE_CONTEXT, IMAGE_OUTPUT>;
            using PAIR_RESPONSE_TYPE = typename PAIR_RESPONSES::template Layer<CAPABILITY, IMAGE_INPUT_SHAPE>;
            static_assert(rlt::get<2>(typename PAIR_RESPONSE_TYPE::OUTPUT_SHAPE{}) == 1 && rlt::get<3>(typename PAIR_RESPONSE_TYPE::OUTPUT_SHAPE{}) == 2);
            using PAIR_DIFFERENCE = rlt::nn::layers::dense::BindConfiguration<rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, 4, rlt::nn::activation_functions::IDENTITY>>;
            using IMAGE_BRANCH = rlt::nn_models::sequential::Module<GREY, COST, POOL, NORMALIZE, DIFFERENCE, PATCH_CONTEXT, IMAGE_CONTEXT, IMAGE_OUTPUT, OUTPUT_FLATTEN, PAIR_DIFFERENCE>;

            using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
            using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
            using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::activation_functions::IDENTITY>;
            using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
            using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

            using MLP_HEAD_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::activation_functions::IDENTITY, rlt::nn::activation_functions::IDENTITY>;
            using MLP_HEAD = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_HEAD_CONFIG>;

            using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
            using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
            using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, MLP_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
        };
        template <typename CAPABILITY>
        struct Critic{
            using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
            using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
            using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
            using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
            using CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
            using TYPE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;
            using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<TYPE>>;
            using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
        };
        using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
        using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
        using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
        using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
        using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, T_DYNAMIC_ALLOCATION>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
        using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
    };

    using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximators>;

    using PPO_SPEC = typename LOOP_CORE_CONFIG::PPO_SPEC;
    using PPO_TYPE = typename LOOP_CORE_CONFIG::PPO_TYPE;
    using PPO_BUFFERS_TYPE = typename LOOP_CORE_CONFIG::PPO_BUFFERS_TYPE;
    using ON_POLICY_RUNNER_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_SPEC;
    using ON_POLICY_RUNNER_DATASET_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_SPEC;
    using ON_POLICY_RUNNER_DATASET_TYPE = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_TYPE;
    using ACTOR_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::ACTOR_OPTIMIZER;
    using CRITIC_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::CRITIC_OPTIMIZER;
    using ACTOR_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_BUFFERS;
    using CRITIC_BUFFERS = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS;
    using ACTOR_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE;

    using CAPABILITY_ROLLOUT = rlt::nn::capability::Forward<true>;
    using ROLLOUT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
    using ROLLOUT_ACTOR_BUFFERS = typename ROLLOUT_ACTOR_TYPE::template Buffer<true>;
    using CHECKPOINT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
    static constexpr TI STEPS_PER_ENV = LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;
    static constexpr TI BATCH_SIZE = LOOP_CORE_PARAMETERS::BATCH_SIZE;
    static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr TI OBS_PRIV_DIM = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr TI STEPS_TOTAL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
    static constexpr TI STEPS_TOTAL_ALL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL;
    static constexpr TI N_EPOCHS = LOOP_CORE_PARAMETERS::PPO_PARAMETERS::N_EPOCHS;
    static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
    using MIXED_SPEC = mixed_imitation::Specification<TYPE_POLICY, TI, BATCH_SIZE, N_ENVIRONMENTS_PER_SCENE, MIXED_TRAINING ? MIXED_PARAMETERS::IMITATION_ENVIRONMENTS_PER_SCENE : 0, MIXED_PARAMETERS>;
    static constexpr TI RL_BATCH_SIZE = MIXED_SPEC::RL_BATCH_SIZE;
    static constexpr TI RL_STEPS_TOTAL = N_BATCHES * RL_BATCH_SIZE;
    static constexpr TI RL_UPDATE_SAMPLES = RL_STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
    static constexpr TI UPDATE_SAMPLES = STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
    static constexpr T GRADIENT_SCALE = static_cast<T>(BATCH_SIZE) / static_cast<T>(UPDATE_SAMPLES);
    static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
    static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH * 2;
    static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

    static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
    static_assert(STEPS_TOTAL % BATCH_SIZE == 0, "Every rollout sample must be consumed exactly once");
    static_assert(N_EPOCHS == 1, "On-policy accumulation does not reuse rollout samples");
    static_assert(GRADIENT_ACCUMULATION_ROLLOUTS == 1 || ROLLOUTS_PER_SCENE_SET == 1, "Multiple accumulation groups require resets at each rollout boundary");
    static_assert(N_ENVIRONMENTS * GRADIENT_ACCUMULATION_ROLLOUTS <= RNG_GPU::NUM_RNGS, "Each logical environment needs an independent RNG stream");
    static_assert(N_EXAMPLES <= BATCH_SIZE, "N_EXAMPLES must fit the reusable combined-observation batch buffer");
    static_assert(N_EXAMPLES <= STEPS_TOTAL, "N_EXAMPLES must fit one PPO rollout dataset");
    static_assert(!MIXED_TRAINING || BATCH_SIZE % N_ENVIRONMENTS == 0, "Every mixed minibatch must cover all scenes with the same IL/RL split");

}
