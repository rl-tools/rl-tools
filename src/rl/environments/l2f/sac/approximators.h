template<typename T, typename TI, TI SEQUENCE_LENGTH, typename ENVIRONMENT, typename PARAMETERS>
struct ConfigApproximatorsSequential{
    static constexpr bool USE_GRU = true;
    template <typename CAPABILITY>
    struct Actor{
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, ENVIRONMENT::Observation::DIM, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU2_SPEC>;
        using DENSE_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER_SPEC>;
        using OUTPUT_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, 2*ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal,rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<OUTPUT_LAYER_SPEC>;
        struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr bool UPDATE_ALPHA_WITH_ACTOR = false;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, ENVIRONMENT::ACTION_DIM, SAMPLE_AND_SQUASH_LAYER_PARAMETERS, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Interface<CAPABILITY>::template Module<T_CONTENT, T_NEXT_MODULE>;
        using SAMPLE_AND_SQUASH_MODULE = Module<SAMPLE_AND_SQUASH_LAYER>;
        using MODEL_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>>>;
        using MODEL_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>;
//        using MODEL = MODEL_GRU_TWO_LAYER;
        using MODEL = MODEL_GRU;
    };
    template <typename CAPABILITY>
    struct Critic{
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU2_SPEC>;
        using DENSE_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal,rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER_SPEC>;
        using OUTPUT_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<OUTPUT_LAYER_SPEC>;
        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Interface<CAPABILITY>::template Module<T_CONTENT, T_NEXT_MODULE>;
        using MODEL_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>>>;
        using MODEL_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>;
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
