template<typename T, typename TI, TI SEQUENCE_LENGTH, typename ENVIRONMENT, typename PARAMETERS, typename CONTAINER_TYPE_TAG>
struct ConfigApproximatorsSequential{
    static constexpr bool USE_GRU = true;
    template <typename CAPABILITY>
    struct Actor{
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, ENVIRONMENT::Observation::DIM, PARAMETERS::ACTOR_HIDDEN_DIM>; //, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_HIDDEN_DIM>; //, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU2_SPEC>;
        using OUTPUT_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, 2*ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<OUTPUT_LAYER_SPEC>;
//        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, 2*ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  rlt::nn::activation_functions::IDENTITY, typename PARAMETERS::INITIALIZER, CONTAINER_TYPE_TAG, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
//        using ACTOR_TYPE = rlt::nn_models::mlp::BindSpecification<ACTOR_SPEC>;
        using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
        struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, ENVIRONMENT::ACTION_DIM, SAMPLE_AND_SQUASH_LAYER_PARAMETERS, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
        using SAMPLE_AND_SQUASH_MODULE = typename IF::template Module<SAMPLE_AND_SQUASH_LAYER::template Layer>;
        using MODEL_GRU = typename IF::template Module<GRU_TEMPLATE::template Layer, typename IF::template Module<GRU2_TEMPLATE::template Layer, typename IF::template Module<OUTPUT_LAYER_TEMPLATE ::template Layer, SAMPLE_AND_SQUASH_MODULE>>>;
//        using MODEL_NN = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>;
        using MODEL = MODEL_GRU; //rlt::utils::typing::conditional_t<USE_GRU, MODEL_GRU, MODEL_NN>;
    };
    template <typename CAPABILITY>
    struct Critic{
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, PARAMETERS::CRITIC_HIDDEN_DIM>; //, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using GRU2_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_HIDDEN_DIM>; //, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU2_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU2_SPEC>;
        using OUTPUT_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using OUTPUT_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<OUTPUT_LAYER_SPEC>;
//        using SPEC = rlt::nn_models::mlp::Specification<T, TI, INPUT_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY, typename PARAMETERS::INITIALIZER, CONTAINER_TYPE_TAG, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
//        using SPEC = rlt::nn_models::mlp::Specification<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY, typename PARAMETERS::INITIALIZER, CONTAINER_TYPE_TAG, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
//        using TYPE = rlt::nn_models::mlp::BindSpecification<SPEC>;
        using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
//        using MODEL = typename IF::template Module<GRU_TEMPLATE::template Layer, typename IF::template Module<TYPE::template NeuralNetwork>>;
        using MODEL_GRU = typename IF::template Module<GRU_TEMPLATE::template Layer, typename IF::template Module<GRU2_TEMPLATE::template Layer, typename IF::template Module<OUTPUT_LAYER_TEMPLATE ::template Layer>>>;
//        using MODEL_NN = typename IF::template Module<TYPE::template NeuralNetwork>;
        using MODEL = MODEL_GRU;
    };

    using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
    using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
    using CAPABILITY_ACTOR = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CAPABILITY_CRITIC = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
    using CRITIC_TARGET_TYPE = typename Critic<rlt::nn::layer_capability::Forward>::MODEL;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

};
