#pragma once

namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::yaw{
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
    using LOGGER = rlt::devices::logging::CPU;
#endif
    using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
    using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
    using DEVICE_COMPUTE_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
    using DEVICE_COMPUTE = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_COMPUTE_SPEC>;
    using DEVICE_GPU = DEVICE_COMPUTE;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<float, rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>, rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>, rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>, rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
    using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
    using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
    using TEACHER_TYPE_POLICY = rlt::numeric_types::Policy<float>;
    using TI = typename DEVICE::index_t;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
    using RNG_COMPUTE = typename DEVICE_COMPUTE::SPEC::RANDOM::ENGINE<>;
    using RNG_GPU = RNG_COMPUTE;

    namespace l2f = rlt::rl::environments::l2f;
    namespace obs = l2f::observation;
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr T INIT_ORIENTATION_CURRICULUM_START_DEG = 0;
    static constexpr T INIT_ORIENTATION_CURRICULUM_FULL_DEG = 90;
    static constexpr T INIT_ORIENTATION_CURRICULUM_STEP_DEG = 1;
    static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO = 0.25;
    static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG = 2.5;
    static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG = 5;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    struct DOMAIN_RANDOMIZATION_OPTIONS{
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
        false, 0.10, 1.00, -100.00, 10.00, 0.00, 1.00, 0.05, 1.50, 0.00, 0.00,
        {0.00, 0.00, 0.00, 0.00}, {1.50, 1.50, 1.50, 1.50}, 0.00
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Initialization initialization = {
        0.0, 0.0, INIT_ORIENTATION_CURRICULUM_FULL_DEG / static_cast<T>(180) * rlt::math::PI<T>, 0.0, 1.0, true, -1, +1,
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
        true, 1.0, 0, 10, 35, 10000, 50000,
    };
    static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
    static constexpr typename PARAMETERS_TYPE::Integration integration = {static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)};
    static constexpr typename PARAMETERS_TYPE::MDP mdp = {initialization, reward_function, {}, {}, termination};
    static constexpr typename PARAMETERS_TYPE::IMU imu = {
        {{{0, 0}, {0, 0, 0}}},
        {{{0, 0}, {0.02, 0, 0}}}
    };
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
    static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {1.7, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr PARAMETERS_TYPE nominal_parameters = {{{{dynamics, integration, mdp}, imu}, disturbances}, domain_randomization};
    static constexpr TI ACTION_HISTORY_LENGTH = 8;

    struct STATIC_PARAMETERS{
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr bool CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = yaw::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_BASE_LA = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
        using STATE_BASE_LAH = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE_LA>>;
        using STATE_BASE_IMU = l2f::StateIMU<T, TI, STATE_BASE_LAH>;
        using STATE_BASE_MAHONY = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_BASE_IMU>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE_MAHONY>>>>>>;
        using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI, obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI, obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI, obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI, obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
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

#ifdef RL_TOOLS_HYPERDRONE_YAW_SMOKE
    static constexpr bool SMOKE = true;
    static constexpr TI N_TRAIN_SCENES = 2;
    static constexpr TI N_VALIDATION_SCENES = 1;
    static constexpr TI NUM_EPOCHS = 3;
    static constexpr TI VALIDATION_CADENCE = 1;
    static constexpr TI CHECKPOINT_CADENCE = 1;
#else
    static constexpr bool SMOKE = false;
    static constexpr TI N_TRAIN_SCENES = 25;
    static constexpr TI N_VALIDATION_SCENES = 5;
    static constexpr TI NUM_EPOCHS = 1000000;
    static constexpr TI VALIDATION_CADENCE = 10;
    static constexpr TI CHECKPOINT_CADENCE = 1000;
#endif
    static constexpr TI N_TOTAL_SCENES = N_TRAIN_SCENES + N_VALIDATION_SCENES;
    using SCENE_DATASET = rlt::rendering::datasets::procthor::Train200;
    static constexpr TI N_ACTIVE_SCENES = 2;
    static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
    static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI STEPS_PER_ENV = 50;
    static constexpr TI BATCH_SIZE = 512;
    static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
    static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
    static constexpr TI TRAINED_ROWS_PER_PASS = N_BATCHES * BATCH_SIZE;
    static constexpr TI N_TRAIN_PASSES = 4;
    static constexpr TI VALIDATION_BATCH_SIZE = N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI N_VALIDATION_YAW_BINS = 11;
    static constexpr TI N_EXAMPLES = 512;
    static constexpr TI REDUCED_BATCH_SIZE = 2;
    static_assert(N_BATCHES == 12 && N_TRAIN_PASSES * N_BATCHES == 48);
    static_assert(N_ACTIVE_SCENES <= N_TRAIN_SCENES);
    static_assert(N_ENVIRONMENTS <= RNG_COMPUTE::NUM_RNGS);
    static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES && N_EXAMPLES <= STEPS_TOTAL);

    static constexpr TI CAM_WIDTH = 80;
    static constexpr TI CAM_HEIGHT = 50;
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI NUM_PROBES = 64;
    static constexpr T CAMERA_FOV = 79.6;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = 5;
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0.01;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = static_cast<T>(5) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = static_cast<T>(10) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
    using RENDER_SHADING = rlt::rendering::raytracing::High;
    static constexpr bool RENDER_ENABLE_MOTION_BLUR = true;
    static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 2;
    static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
    static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
    static constexpr T RENDER_SHUTTER_FRACTION_MIN = 0.25;
    static constexpr T RENDER_SHUTTER_FRACTION_MAX = 1;
    static constexpr TI FRAME_STACK_N = 1;
    static constexpr TI FRAME_STACK_STRIDE = 10;
    static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;

    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, STATIC_PARAMETERS>{
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = N_ENVIRONMENTS_PER_SCENE;
        static constexpr TI CAM_WIDTH = yaw::CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = yaw::CAM_HEIGHT;
        static constexpr TI NUM_PROBES = yaw::NUM_PROBES;
        using SHADING = RENDER_SHADING;
        static constexpr bool ENABLE_MOTION_BLUR = RENDER_ENABLE_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = RENDER_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = RENDER_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = RENDER_ANTI_ALIASING_GRID_SIZE;
        static constexpr TI HISTORY_LENGTH = FRAME_STACK_HISTORY_LENGTH;
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = yaw::CAMERA_FOV_RANDOMIZATION_RANGE;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = yaw::CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = yaw::CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = yaw::BRIGHTNESS_RANDOMIZATION_RANGE;
        static constexpr T SHUTTER_FRACTION_MIN = RENDER_SHUTTER_FRACTION_MIN;
        static constexpr T SHUTTER_FRACTION_MAX = RENDER_SHUTTER_FRACTION_MAX;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD>{
        static constexpr TI IMAGE_STACK_N = FRAME_STACK_N;
        static constexpr TI IMAGE_STACK_STRIDE = FRAME_STACK_STRIDE;
        static constexpr TI PAD_CHANNELS_TO = 8;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = yaw::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = yaw::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
        // The initial render and final unused next observation must stay inside this epoch's bank.
        static constexpr TI TARGET_BANK_REFRESH_STEPS = STEPS_PER_ENV + 2;
    };
    using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    using HYPERDRONE = rlt::rl::environments::HyperDrone<TASK_WORLD, N_ACTIVE_SCENES>;
#ifdef RL_TOOLS_RENDERING_RAYTRACING_ENABLE_SPY
    using SPY = rlt::rl::environments::hyperdrone::Spy<rlt::rl::environments::hyperdrone::spy::Config<HYPERDRONE>>;
#endif
    using ENVIRONMENT = TASK_WORLD;
    static constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
    static constexpr TI IMG_H = CAM_HEIGHT;
    static constexpr TI IMG_W = CAM_WIDTH;
    static constexpr TI IMG_C = BASE_WORLD::IMAGE_CHANNELS;
    static constexpr TI COMBINED_IMG_C = TASK_WORLD::OBSERVATION_CHANNELS;
    static constexpr TI COMBINED_OBS_DIM = TASK_WORLD::OBSERVATION_DIM;
    static constexpr bool RENDER_MOTION_BLUR_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_MOTION_BLUR;
    static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_ANTI_ALIASING;
    static_assert(COMBINED_IMG_C == 8 && COMBINED_OBS_DIM == 32000);
    static_assert(BASE_WORLD::RENDERER_SPEC::SHADING::PBR_SHADING);

    static constexpr TI RAPTOR_HIDDEN_DIM = 16;
    using RAPTOR_OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI, obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI, obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
    static constexpr TI RAPTOR_OBS_DIM = RAPTOR_OBSERVATION_TYPE::DIM;
    using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
    using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
    using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
    using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using RAPTOR_DENSE2 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE2_CONFIG>;
    using RAPTOR_MODULE = rlt::nn_models::sequential::Module<RAPTOR_DENSE1, RAPTOR_GRU, RAPTOR_DENSE2>;
    using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, RAPTOR_OBS_DIM>;
    using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
    using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, RAPTOR_MODULE, RAPTOR_INPUT_SHAPE>;
    static_assert(RAPTOR_OBS_DIM == 22);

    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr TI TARGET_DIM = 2;
    static constexpr T YAW_R2_EPS = 1e-8;
    using ADAM_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;

    template<typename CAPABILITY, typename T_TYPE_POLICY = TYPE_POLICY>
    struct StudentActor{
        using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
        template<TI CHANNELS>
        using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, CHANNELS, 3, 3, 2, 2, 1, 1, ACTOR_ACTIVATION_FUNCTION>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG<32>>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG<64>>;
        using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG<128>>;
        using CONV4 = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG<256>>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, TI>>;
        using DENSE_HIDDEN = rlt::nn::layers::dense::BindConfiguration<rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>>;
        using IMAGE_BRANCH = rlt::nn_models::sequential::Module<CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, DENSE_HIDDEN>;
        using DENSE_OUTPUT = rlt::nn::layers::dense::BindConfiguration<rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, TARGET_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>>;
        using HEAD = rlt::nn_models::sequential::Module<DENSE_HIDDEN, DENSE_HIDDEN, DENSE_OUTPUT>;
        using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD, rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>>;
    };
    using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
    using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
    using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
    using CAPABILITY_FORWARD_CPU = rlt::nn::capability::Forward<true>;
    using CPU_STUDENT_TYPE = typename StudentActor<CAPABILITY_FORWARD_CPU, TEACHER_TYPE_POLICY>::MODEL;
    using CPU_STUDENT_INIT_TYPE = typename StudentActor<CAPABILITY_ADAM, TEACHER_TYPE_POLICY>::MODEL;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    using RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, HYPERDRONE, typename RAPTOR_MODEL::State<true>, typename TASK_WORLD::Observation, RAPTOR_OBSERVATION_TYPE, T_ACTIVATION, T, EPISODE_STEP_LIMIT, false, true, false>;
    using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
    using RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<RUNNER_SPEC>;
    using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
}
