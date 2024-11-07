#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_APPROXIMATORS_GRU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_APPROXIMATORS_GRU_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::td3::loop::core{
    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
    struct ConfigApproximatorsGRU{
    //    static constexpr bool USE_GRU = true;
        using TD3_PARAMETERS = typename PARAMETERS::TD3_PARAMETERS;
        template <typename CAPABILITY>
        struct Actor{
            using INPUT_SHAPE = tensor::Shape<TI, TD3_PARAMETERS::SEQUENCE_LENGTH, TD3_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
            using GRU_CONFIG = nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU = nn::layers::gru::BindConfiguration<GRU_CONFIG>;
            using GRU2_CONFIG = nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU2 = nn::layers::gru::BindConfiguration<GRU2_CONFIG>;
            using OUTPUT_CONFIG = nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using OUTPUT = nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;

            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

            using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT>>>;
            using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT>>;

            using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
    //        using MODULE_CHAIN = MODULE_CHAIN_GRU;
            using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
        };
        template <typename CAPABILITY>
        struct Critic{
            using INPUT_SHAPE = tensor::Shape<TI, TD3_PARAMETERS::SEQUENCE_LENGTH, TD3_PARAMETERS::CRITIC_BATCH_SIZE, ENVIRONMENT::ObservationPrivileged::DIM + ENVIRONMENT::ACTION_DIM>;
            using GRU_CONFIG = nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU = nn::layers::gru::BindConfiguration<GRU_CONFIG>;
            using GRU2_CONFIG = nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU2 = nn::layers::gru::BindConfiguration<GRU2_CONFIG>;
            using OUTPUT_CONFIG = nn::layers::dense::Configuration<T, TI, 1, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using OUTPUT = nn::layers::dense::BindConfiguration<OUTPUT_CONFIG>;
            static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;

            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

            using MODULE_CHAIN_GRU_TWO_LAYER = Module<GRU, Module<GRU2, Module<OUTPUT>>>;
            using MODULE_CHAIN_GRU = Module<GRU, Module<OUTPUT>>;
            using MODULE_CHAIN = MODULE_CHAIN_GRU_TWO_LAYER;
    //        using MODULE_CHAIN = MODULE_CHAIN_GRU;

            using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
        };

        using CAPABILITY_ACTOR = nn::capability::Gradient<nn::parameters::Adam>;
        using CAPABILITY_CRITIC = nn::capability::Gradient<nn::parameters::Adam>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
        using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
        using CRITIC_TARGET_TYPE = typename Critic<nn::capability::Forward<>>::MODEL;
        using ACTOR_TARGET_TYPE = typename Actor<nn::capability::Forward<>>::MODEL;
        using OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif


