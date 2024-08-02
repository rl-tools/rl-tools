

template <typename T, typename TI>
struct Config{
    static constexpr TI NUM_CLASSES = 2<<7;
    static constexpr TI BATCH_SIZE = 32;
    static constexpr TI SEQUENCE_LENGTH = 128;
    static constexpr TI OUTPUT_DIM = NUM_CLASSES;
//    static constexpr TI EMBEDDING_DIM = 32;
//    static constexpr TI HIDDEN_DIM = 64;
    static constexpr TI EMBEDDING_DIM = 128;
    static constexpr TI HIDDEN_DIM = 256;

    template <TI T_BATCH_SIZE>
    using INPUT_SHAPE_TEMPLATE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, T_BATCH_SIZE>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using INPUT_SHAPE = INPUT_SHAPE_TEMPLATE<BATCH_SIZE>;
    using INPUT_SPEC = rlt::tensor::Specification<unsigned char, TI, INPUT_SHAPE>;
    using EMBEDDING_LAYER_SPEC = rlt::nn::layers::embedding::Specification<T, TI, NUM_CLASSES, EMBEDDING_DIM, INPUT_SHAPE_TEMPLATE>;
    using EMBEDDING_LAYER_TEMPLATE = rlt::nn::layers::embedding::BindSpecification<EMBEDDING_LAYER_SPEC>;
    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, rlt::nn::parameters::Gradient, rlt::TensorDynamicTag, true>;
    using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
    using DENSE_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
    using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER_SPEC>;
    using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
    using MODEL = typename IF::template Module<EMBEDDING_LAYER_TEMPLATE::template Layer, typename IF::template Module<GRU_TEMPLATE:: template Layer, typename IF::template Module<DENSE_LAYER_TEMPLATE::template Layer>>>;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
    using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;
    using OUTPUT_TARGET_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>;
    using OUTPUT_TARGET_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_TARGET_SHAPE>;
    using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using ADAM = rlt::nn::optimizers::Adam<ADAM_SPEC>;
};
