#pragma once

namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::imitation{
#ifdef RL_TOOLS_HYPERDRONE_IMITATION_STATE_ESTIMATION
    static constexpr bool STATE_ESTIMATION_MODE = true;
#else
    static constexpr bool STATE_ESTIMATION_MODE = false;
#endif

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
    using LOGGER = rlt::devices::logging::CPU;
#endif
    using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
    using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
    using T = float;
#ifdef __CUDACC__
    using DEVICE_COMPUTE_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
    using DEVICE_COMPUTE = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_COMPUTE_SPEC>;
    using TYPE_POLICY = rlt::numeric_types::Policy<float,
        rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>,
        rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>,
        rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>,
        rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
    static constexpr const char* COMPUTE_DEVICE_NAME = "cuda";
#else
    using DEVICE_COMPUTE = DEVICE;
    using TYPE_POLICY = rlt::numeric_types::Policy<float>;
    static constexpr const char* COMPUTE_DEVICE_NAME = "cpu";
#endif
    using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
    using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
    using TEACHER_TYPE_POLICY = rlt::numeric_types::Policy<float>;
    using TI = typename DEVICE::index_t;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
    using RNG_COMPUTE = typename DEVICE_COMPUTE::SPEC::RANDOM::ENGINE<>;

    namespace l2f = rlt::rl::environments::l2f;
    namespace obs = l2f::observation;

    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    struct DOMAIN_RANDOMIZATION_OPTIONS {
        static constexpr bool THRUST_TO_WEIGHT = false;
        static constexpr bool MASS = false;
        static constexpr bool TORQUE_TO_INERTIA = false;
        static constexpr bool MASS_SIZE_DEVIATION = false;
        static constexpr bool ROTOR_TORQUE_CONSTANT = false;
        static constexpr bool DISTURBANCE_FORCE = false;
        static constexpr bool ROTOR_TIME_CONSTANT = false;
    };
    using PARAMETERS_TYPE = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>>>;

    static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie_openmv;

    static constexpr REWARD_FUNCTION reward_function = {
        false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        {0.00, 0.00, 0.00, 0.00}, {0.00, 0.00, 0.00, 0.00}, 0.00
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
    static constexpr T DISTURBANCE_FORCE_STD = 0;
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, DISTURBANCE_FORCE_STD}, {0, 0} };
    static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    static constexpr PARAMETERS_TYPE nominal_parameters = { {{dynamics, integration, mdp}, disturbances}, domain_randomization };

    static constexpr TI ACTION_HISTORY_LENGTH = 8;

    struct STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = imitation::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE>>>>>>>>;
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

    static constexpr TI N_RANKS = RL_TOOLS_HYPERDRONE_IMITATION_N_RANKS;
    static_assert(N_RANKS >= 1, "At least one imitation rank is required");
#ifndef __CUDACC__
    static_assert(N_RANKS == 1, "Multiple imitation ranks require CUDA compute");
#endif
#ifdef RL_TOOLS_HYPERDRONE_IMITATION_SMALL
    static constexpr TI N_TOTAL_SCENES = 2;
    static constexpr TI N_ENVIRONMENTS_PER_SCENE = 16;
    static constexpr TI STEPS_PER_ENV = 100;
    static constexpr TI N_TRAIN_PASSES = 1;
    static constexpr TI N_EXAMPLES = 64;
    static constexpr TI ENV_GRID_SIDE = 4;
    static constexpr TI NUM_EPOCHS = 3;
    static constexpr TI CHECKPOINT_CADENCE = 2;
#else
    static constexpr TI N_TOTAL_SCENES = 25;
    static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
    static constexpr TI STEPS_PER_ENV = 500;
    static constexpr TI N_TRAIN_PASSES = 4;
    static constexpr TI N_EXAMPLES = 512;
    static constexpr TI ENV_GRID_SIDE = 8;
    static constexpr TI NUM_EPOCHS = 1000000;
    static constexpr TI CHECKPOINT_CADENCE = 1000;
#endif
    static constexpr TI N_ACTIVE_SCENES_TOTAL = 2;
    static_assert(N_ACTIVE_SCENES_TOTAL % N_RANKS == 0, "The active Worlds must split evenly over the ranks");
    static_assert(N_TOTAL_SCENES >= N_ACTIVE_SCENES_TOTAL, "Every World needs at least one scene");
    static constexpr TI N_ACTIVE_SCENES = N_ACTIVE_SCENES_TOTAL / N_RANKS;
    static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI N_ENVIRONMENTS_TOTAL = N_ACTIVE_SCENES_TOTAL * N_ENVIRONMENTS_PER_SCENE;
#ifdef __CUDACC__
    static_assert(N_ENVIRONMENTS_TOTAL <= RNG_COMPUTE::NUM_RNGS, "Every global instance needs its own CUDA RNG stream");
#endif
    static constexpr TI CAM_WIDTH = 80;
    static constexpr TI CAM_HEIGHT = 50;
    static constexpr TI NUM_PROBES = 64;
    static constexpr T CAMERA_FOV = 63.8;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(5.0);
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE =
        STATE_ESTIMATION_MODE ? static_cast<T>(0) : static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;

    using RENDER_SHADING = rlt::rendering::raytracing::High;
    static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
    static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
    static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
    static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
    static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
    static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);

    static constexpr TI FRAME_STACK_N = 10;
    static constexpr TI FRAME_STACK_STRIDE = 10;
    static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;

    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, STATIC_PARAMETERS>{
        static constexpr TI INSTANCES_PER_ENVIRONMENT = N_ENVIRONMENTS_PER_SCENE;
        static constexpr TI CAM_WIDTH = imitation::CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = imitation::CAM_HEIGHT;
        static constexpr TI NUM_PROBES = imitation::NUM_PROBES;
        using SHADING = RENDER_SHADING;
        static constexpr bool ENABLE_MOTION_BLUR = RENDER_ENABLE_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = RENDER_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = RENDER_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = RENDER_ANTI_ALIASING_GRID_SIZE;
        static constexpr TI HISTORY_LENGTH = FRAME_STACK_HISTORY_LENGTH;
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = imitation::CAMERA_FOV_RANDOMIZATION_RANGE;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = imitation::BRIGHTNESS_RANDOMIZATION_RANGE;
        static constexpr T SHUTTER_FRACTION_MIN = RENDER_SHUTTER_FRACTION_MIN;
        static constexpr T SHUTTER_FRACTION_MAX = RENDER_SHUTTER_FRACTION_MAX;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    static_assert(BASE_WORLD::RENDERER_SPEC::SHADING::PBR_SHADING, "Hyperdrone imitation must use the High renderer profile");
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD>{
        static constexpr TI IMAGE_STACK_N = FRAME_STACK_N;
        static constexpr TI IMAGE_STACK_STRIDE = FRAME_STACK_STRIDE;
        static constexpr TI PAD_CHANNELS_TO = 8;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = imitation::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = imitation::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
        // teacher-flown episodes span the epoch, whose scene rotation already refreshes the bank
        static constexpr TI TARGET_BANK_REFRESH_STEPS = EPISODE_STEP_LIMIT / 2;
    };
    using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    static constexpr TI NUMBER_OF_ENVIRONMENTS = N_ACTIVE_SCENES;
    using MULTI_ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<TASK_WORLD, NUMBER_OF_ENVIRONMENTS>;
    using ENVIRONMENT = TASK_WORLD;
    static constexpr bool RENDER_MOTION_BLUR_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_MOTION_BLUR;
    static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_ANTI_ALIASING;
    static_assert(TASK_WORLD::INSTANCES == N_ENVIRONMENTS_PER_SCENE);
    static_assert(MULTI_ENVIRONMENT::INSTANCES == N_ENVIRONMENTS);
    static constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
    static constexpr TI IMG_H = BASE_WORLD::Observation::HEIGHT;
    static constexpr TI IMG_W = BASE_WORLD::Observation::WIDTH;
    static constexpr TI IMG_C = BASE_WORLD::Observation::CHANNELS;
    static constexpr TI COMBINED_IMG_C = TASK_WORLD::OBSERVATION_CHANNELS;
    static constexpr TI COMBINED_OBS_DIM = TASK_WORLD::OBSERVATION_DIM;
    static_assert(COMBINED_IMG_C == ((IMG_C * (FRAME_STACK_N + 1) + 7) & ~((TI)7)), "Composed image channels must be padded to a multiple of eight");
    static_assert(COMBINED_OBS_DIM == IMG_H * IMG_W * COMBINED_IMG_C);

    static constexpr TI RAPTOR_HIDDEN_DIM = 16;
    using RAPTOR_OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
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

    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr TI STATE_ESTIMATION_TARGET_DIM = 3 + 3 + 9;
    static constexpr TI STATE_ESTIMATION_NUM_METRICS = 4;
    static constexpr TI TARGET_DIM = STATE_ESTIMATION_MODE ? STATE_ESTIMATION_TARGET_DIM : ACTION_DIM;
    static constexpr TI BATCH_SIZE_TOTAL = 512;
    static_assert(BATCH_SIZE_TOTAL % N_RANKS == 0, "The training batch must split evenly over the ranks");
    static_assert(BATCH_SIZE_TOTAL % N_ENVIRONMENTS_TOTAL == 0, "Training batches must contain whole global simulation steps");
    static constexpr TI BATCH_SIZE = BATCH_SIZE_TOTAL / N_RANKS;
    static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
    static constexpr TI STEPS_TOTAL_ALL_RANKS = STEPS_PER_ENV * N_ENVIRONMENTS_TOTAL;
    static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
    static_assert(N_BATCHES == STEPS_TOTAL_ALL_RANKS / BATCH_SIZE_TOTAL);
    static constexpr T TEACHER_FORCING_FRACTION = 0.0;
    static constexpr T EFFECTIVE_TEACHER_FORCING_FRACTION = STATE_ESTIMATION_MODE ? static_cast<T>(1) : TEACHER_FORCING_FRACTION;
    static constexpr bool EXPORT_CHECKPOINT_TAR = false;
    static constexpr bool EXPORT_CHECKPOINT_CODE = false;
    static constexpr TI REDUCED_BATCH_SIZE = 2;
    static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
    static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
    static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
    static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "N_ENVIRONMENTS_PER_SCENE must be a perfect square for the per-scene video mosaic");
    static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
    static constexpr TI TRAJECTORY_NUM_ENVS = 10;
    static constexpr TI TRAJECTORY_MAX_EPISODES = 10;
    static_assert(TRAJECTORY_NUM_ENVS <= N_ENVIRONMENTS);

    struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
        static constexpr T ALPHA = 3e-4;
        static constexpr T EPSILON = 1e-5;
        static constexpr T EPSILON_SQRT = 1e-5;
    };

    template<typename CAPABILITY, typename T_TYPE_POLICY = TYPE_POLICY>
    struct StudentActor{
        static constexpr TI STEPS = 1;
        static constexpr TI FORWARD_BATCH_SIZE = BATCH_SIZE;
        using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
        using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

        static constexpr TI HIDDEN_DIM_MULTIPLIER = 2;
        using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 16*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
        using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 32*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
        using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 64*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
        using CONV4_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 128*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV4 = rlt::nn::layers::conv2d::BindConfiguration<CONV4_CONFIG>;
        using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, TI>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
        using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
        using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
        using IMAGE_BRANCH = rlt::nn_models::sequential::Module<CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

        using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, TI>;
        using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
        using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
        using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
        using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

        using HEAD_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
        using HEAD_DENSE1 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE1_CONFIG>;
        using HEAD_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
        using HEAD_DENSE2 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE2_CONFIG>;
        using HEAD_DENSE_OUT_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, TARGET_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
        using HEAD_DENSE_OUT = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_OUT_CONFIG>;
        using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
        using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
        using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
        using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
    };

    using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
    using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
    using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
    using CAPABILITY_FORWARD_CPU = rlt::nn::capability::Forward<true>;
    using CPU_STUDENT_TYPE = typename StudentActor<CAPABILITY_FORWARD_CPU, TEACHER_TYPE_POLICY>::MODEL;
    using CPU_STUDENT_INIT_TYPE = typename StudentActor<CAPABILITY_ADAM, TEACHER_TYPE_POLICY>::MODEL;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    using ROLLOUT_STUDENT_TYPE = typename STUDENT_TYPE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
    // the batched on-policy runner collects the composed observations (student input, activation
    // precision) and the teacher's privileged observations per row; the teacher's recurrent state is
    // the rollout policy state
    using RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, MULTI_ENVIRONMENT, typename RAPTOR_MODEL::State<true>, typename TASK_WORLD::Observation, RAPTOR_OBSERVATION_TYPE, T_ACTIVATION, T, EPISODE_STEP_LIMIT, false, true, false>;
    using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
    using RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<RUNNER_SPEC>;
    using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
    static_assert(RUNNER_SPEC::STEP_LIMIT == EPISODE_STEP_LIMIT);
}
