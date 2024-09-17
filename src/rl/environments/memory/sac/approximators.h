template<typename T, typename TI, TI SEQUENCE_LENGTH, typename ENVIRONMENT, typename PARAMETERS>
struct ConfigApproximatorsSequential{
    static constexpr bool USE_GRU = true;
    template <typename CAPABILITY>
    struct Actor{
        using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU = rlt::nn::layers::gru::BindSpecification<GRU_CONFIG>;
        using GRU2_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2 = rlt::nn::layers::gru::BindSpecification<GRU2_CONFIG>;
        using OUTPUT_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 2*ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT = rlt::nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;
        struct SAMPLE_AND_SQUASH_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr bool UPDATE_ALPHA_WITH_ACTOR = false;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_CONFIG = rlt::nn::layers::sample_and_squash::Configuration<T, TI, SAMPLE_AND_SQUASH_PARAMETERS>;
        using SAMPLE_AND_SQUASH = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_CONFIG>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;

        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;

        using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT, Module<SAMPLE_AND_SQUASH>>>>;
        using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT, SAMPLE_AND_SQUASH>>;

        using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
//        using MODULE_CHAIN = MODULE_CHAIN_GRU;
        using MODEL = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    template <typename CAPABILITY>
    struct Critic{
        using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU = rlt::nn::layers::gru::BindSpecification<GRU_CONFIG>;
        using GRU2_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2 = rlt::nn::layers::gru::BindSpecification<GRU2_CONFIG>;
        using OUTPUT_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT = rlt::nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE, INPUT_DIM>;

        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;

        using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT>>>;
        using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT>>;
        using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
//        using MODULE_CHAIN = MODULE_CHAIN_GRU;

        using MODEL = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };

    using CAPABILITY_ACTOR = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CAPABILITY_CRITIC = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
    using CRITIC_TARGET_TYPE = typename Critic<rlt::nn::layer_capability::Forward<>>::MODEL;
    using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS>;
    using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS>;
    using ALPHA_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::ALPHA_OPTIMIZER_PARAMETERS>;
    using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
    using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
    using ALPHA_OPTIMIZER = rlt::nn::optimizers::Adam<ALPHA_OPTIMIZER_SPEC>;

};
