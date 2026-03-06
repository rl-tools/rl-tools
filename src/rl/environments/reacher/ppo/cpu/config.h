template <typename DEVICE, typename TYPE_POLICY, bool DYNAMIC_ALLOCATION>
struct CONFIG_FACTORY{
    using TI = typename DEVICE::index_t;
    using T = typename TYPE_POLICY::DEFAULT;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
    using REACHER_SPEC = rlt::rl::environments::reacher::Specification<typename TYPE_POLICY::DEFAULT, TI, rlt::rl::environments::reacher::DefaultParameters<typename TYPE_POLICY::DEFAULT>>;
    using ENVIRONMENT = rlt::rl::environments::ReacherVisual<REACHER_SPEC>;

    struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
        static constexpr T ALPHA = 3e-4;
        static constexpr T EPSILON = 1e-5;
        static constexpr T EPSILON_SQRT = 1e-5;
    };

    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
        static constexpr TI BATCH_SIZE = 800;
        static constexpr TI ACTOR_HIDDEN_DIM = 512;
        static constexpr TI CRITIC_HIDDEN_DIM = 512;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 50;
        static constexpr TI N_ENVIRONMENTS = 512;
        static constexpr TI TOTAL_STEP_LIMIT = 10000000;
        static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
        static constexpr TI EPISODE_STEP_LIMIT = 200;
        using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
            static constexpr TI N_EPOCHS = 4;
            static constexpr T GAMMA = 0.8;
            static constexpr T LAMBDA = 0.9;
            static constexpr T EPSILON_CLIP = 0.2;
            static constexpr T INITIAL_ACTION_STD = 0.6065306597633104; // exp(-0.5), i.e. log_std initialized to -0.5
            static constexpr bool NORMALIZE_OBSERVATIONS = false;
        };
    };

    template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION=true>
    struct ConfigApproximatorsVisual{
        static constexpr TI STEPS = 1;
        static constexpr TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;
        template <typename CAPABILITY>
        struct Actor{
            using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
            using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
            // NatureCNN conv layers (adapted for 32x32 with padding)
            // Conv1: 32x32x3 → 8x8x32
            using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 8, 8, 4, 4, 2, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
            using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
            // Conv2: 8x8x32 → 4x4x64
            using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 64, 4, 4, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
            using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
            // Conv3: 4x4x64 → 4x4x64
            using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 64, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
            using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
            // AvgPool: 4x4x64 → 64
            using AVG_POOL_CONFIG = rlt::nn::layers::avg_pool2d::Configuration<T_TYPE_POLICY, T_TI>;
            using AVG_POOL = rlt::nn::layers::avg_pool2d::BindConfiguration<AVG_POOL_CONFIG>;
            // MLP head: 64 → 512 → action_dim (+ learned log_std)
            using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
            using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;

            using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV1, rlt::nn_models::sequential::Module<CONV2, rlt::nn_models::sequential::Module<CONV3, rlt::nn_models::sequential::Module<AVG_POOL, rlt::nn_models::sequential::Module<MLP>>>>>;
            using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
        };
        template <typename CAPABILITY>
        struct Critic{
            using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
            using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
            using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
            using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
            // Critic MLP: 4 → 512 → 512 → 1
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

    using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsVisual, DYNAMIC_ALLOCATION>;
    template <typename NEXT>
    struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, NEXT>{
        static constexpr TI EVALUATION_INTERVAL = 5;
        static constexpr TI NUM_EVALUATION_EPISODES = 100;
        static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
    };

    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
};
