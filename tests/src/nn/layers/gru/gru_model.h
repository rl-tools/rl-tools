

template <typename T, typename TI>
struct Config{
    struct BASE{
        static constexpr TI NUM_CLASSES = 2<<7;
        static constexpr TI BATCH_SIZE = 32;
        static constexpr TI OUTPUT_DIM = NUM_CLASSES;
        static constexpr TI EMBEDDING_DIM = 32;
        static constexpr TI HIDDEN_DIM = 64;
        static constexpr TI SEQUENCE_LENGTH = 64;
    };
    struct USEFUL: BASE{
        static constexpr TI SEQUENCE_LENGTH = 128;
        static constexpr TI EMBEDDING_DIM = 64;
        static constexpr TI HIDDEN_DIM = 256;
    };

//    using PARAMS = BASE;
    using PARAMS = USEFUL;

    template <TI T_BATCH_SIZE>
    using INPUT_SHAPE_TEMPLATE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, T_BATCH_SIZE>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMS::BATCH_SIZE>;
    using INPUT_SHAPE = INPUT_SHAPE_TEMPLATE<PARAMS::BATCH_SIZE>;
    using INPUT_SPEC = rlt::tensor::Specification<unsigned char, TI, INPUT_SHAPE>;
    using EMBEDDING_LAYER_SPEC = rlt::nn::layers::embedding::Specification<T, TI, PARAMS::NUM_CLASSES, PARAMS::EMBEDDING_DIM, INPUT_SHAPE_TEMPLATE>;
    using EMBEDDING_LAYER_TEMPLATE = rlt::nn::layers::embedding::BindSpecification<EMBEDDING_LAYER_SPEC>;
    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, PARAMS::SEQUENCE_LENGTH, PARAMS::EMBEDDING_DIM, PARAMS::HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
    using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
    using GRU2_SPEC = rlt::nn::layers::gru::Specification<T, TI, PARAMS::SEQUENCE_LENGTH, PARAMS::HIDDEN_DIM, PARAMS::HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
    using GRU2_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU2_SPEC>;
    using DOWN_PROJECTION_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMS::HIDDEN_DIM, PARAMS::EMBEDDING_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, PARAMS::SEQUENCE_LENGTH>>;
    using DOWN_PROJECTION_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DOWN_PROJECTION_LAYER_SPEC>;
    using DENSE_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMS::EMBEDDING_DIM, PARAMS::OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal,rlt::nn::layers::dense::SequenceInputShapeFactory<TI, PARAMS::SEQUENCE_LENGTH>>;
    using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER_SPEC>;
    using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
    using MODEL = typename IF::template Module<EMBEDDING_LAYER_TEMPLATE::template Layer, typename IF::template Module<GRU_TEMPLATE:: template Layer, typename IF::template Module<GRU2_TEMPLATE:: template Layer, typename IF::template Module<DOWN_PROJECTION_LAYER_TEMPLATE::template Layer, typename IF::template Module<DENSE_LAYER_TEMPLATE::template Layer>>>>>;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, PARAMS::BATCH_SIZE, PARAMS::OUTPUT_DIM>;
    using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;
    using OUTPUT_TARGET_SHAPE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, PARAMS::BATCH_SIZE, 1>;
    using OUTPUT_TARGET_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_TARGET_SHAPE>;
    struct ADAM_PARAMS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
        static constexpr T ALPHA = 0.003;
    };
    using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMS>;
    using ADAM = rlt::nn::optimizers::Adam<ADAM_SPEC>;
};
