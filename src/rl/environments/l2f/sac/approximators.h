template<typename T, typename TI, TI SEQUENCE_LENGTH, typename ENVIRONMENT, typename PARAMETERS>
struct ConfigApproximatorsSequential{
    static constexpr bool USE_GRU = true;
    template <typename CAPABILITY>
    struct Actor{
        using GRU_SPEC = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU2_SPEC>;
        using DENSE_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
        using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 2*ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
        struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr bool UPDATE_ALPHA_WITH_ACTOR = false;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Configuration<T, TI, SAMPLE_AND_SQUASH_LAYER_PARAMETERS>;
        using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindConfiguration<SAMPLE_AND_SQUASH_LAYER_SPEC>;
        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;
        using SAMPLE_AND_SQUASH_MODULE = Module<SAMPLE_AND_SQUASH_LAYER>;
        using MODULE_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>>>;
        using MODULE_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
        using MODEL_GRU = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_GRU, INPUT_SHAPE>;
//        using MODEL = MODEL_GRU_TWO_LAYER;
        using MODEL = MODEL_GRU;
    };
    template <typename CAPABILITY>
    struct Critic{
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using GRU_SPEC = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU2_SPEC>;
        using DENSE_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
        using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;
        using MODEL_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>>>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE, INPUT_DIM>;
        using MODULE_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>;
        using MODEL_GRU = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_GRU, INPUT_SHAPE>;
//        using MODEL = MODEL_GRU_TWO_LAYER;
        using MODEL = MODEL_GRU;
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
