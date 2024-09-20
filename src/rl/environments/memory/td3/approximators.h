template<typename T, typename TI, TI SEQUENCE_LENGTH, typename ENVIRONMENT, typename PARAMETERS>
struct ConfigApproximatorsSequential{
//    static constexpr bool USE_GRU = true;
    using TD3_PARAMETERS = typename PARAMETERS::TD3_PARAMETERS;
    template <typename CAPABILITY>
    struct Actor{
        using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
        using GRU2_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2 = rlt::nn::layers::gru::BindConfiguration<GRU2_CONFIG>;
        using OUTPUT_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT = rlt::nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, TD3_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;

        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;

        using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT>>>;
        using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT>>;

        using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
//        using MODULE_CHAIN = MODULE_CHAIN_GRU;
        using MODEL = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    template <typename CAPABILITY>
    struct Critic{
        using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
        using GRU2_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
        using GRU2 = rlt::nn::layers::gru::BindConfiguration<GRU2_CONFIG>;
        using OUTPUT_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal>;
        using OUTPUT = rlt::nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, TD3_PARAMETERS::CRITIC_BATCH_SIZE, INPUT_DIM>;

        template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential_v2::OutputModule>
        using Module = typename rlt::nn_models::sequential_v2::Module<T_CONTENT, T_NEXT_MODULE>;

        using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT>>>;
        using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT>>;
        using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
//        using MODULE_CHAIN = MODULE_CHAIN_GRU;

        using MODEL = rlt::nn_models::sequential_v2::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };

    using CAPABILITY_ACTOR = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam>;
    using CAPABILITY_CRITIC = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
    using CRITIC_TARGET_TYPE = typename Critic<rlt::nn::layer_capability::Forward<>>::MODEL;
    using ACTOR_TARGET_TYPE = typename Actor<rlt::nn::layer_capability::Forward<>>::MODEL;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

};
