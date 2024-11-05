#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_APPROXIMATORS_GRU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_APPROXIMATORS_GRU_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac::loop::core{
    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
    struct ConfigApproximatorsGRU{
        static constexpr bool USE_GRU = true;
        using SAC_PARAMETERS = typename PARAMETERS::SAC_PARAMETERS;
        template <typename CAPABILITY>
        struct Actor{
            using GRU_SPEC = nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU_TEMPLATE = nn::layers::gru::BindConfiguration<GRU_SPEC>;
            using GRU2_SPEC = nn::layers::gru::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU2_TEMPLATE = nn::layers::gru::BindConfiguration<GRU2_SPEC>;
            using DENSE_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using DENSE_LAYER_TEMPLATE = nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
            using OUTPUT_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, 2*ENVIRONMENT::ACTION_DIM, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using OUTPUT_LAYER_TEMPLATE = nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
            struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
                static constexpr T LOG_STD_LOWER_BOUND = SAC_PARAMETERS::LOG_STD_LOWER_BOUND;
                static constexpr T LOG_STD_UPPER_BOUND = SAC_PARAMETERS::LOG_STD_UPPER_BOUND;
                static constexpr T LOG_PROBABILITY_EPSILON = SAC_PARAMETERS::LOG_PROBABILITY_EPSILON;
                static constexpr bool ADAPTIVE_ALPHA = SAC_PARAMETERS::ADAPTIVE_ALPHA;
                static constexpr bool UPDATE_ALPHA_WITH_ACTOR = false;
                static constexpr T ALPHA = SAC_PARAMETERS::ALPHA;
                static constexpr T TARGET_ENTROPY = SAC_PARAMETERS::TARGET_ENTROPY;
            };
            using SAMPLE_AND_SQUASH_LAYER_SPEC = nn::layers::sample_and_squash::Configuration<T, TI, SAMPLE_AND_SQUASH_LAYER_PARAMETERS>;
            using SAMPLE_AND_SQUASH_LAYER = nn::layers::sample_and_squash::BindConfiguration<SAMPLE_AND_SQUASH_LAYER_SPEC>;
            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
            using SAMPLE_AND_SQUASH_MODULE = Module<SAMPLE_AND_SQUASH_LAYER>;
            using MODULE_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>>>;
            using MODULE_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE, SAMPLE_AND_SQUASH_MODULE>>;
            using INPUT_SHAPE = tensor::Shape<TI, SAC_PARAMETERS::SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
            using MODEL_GRU = nn_models::sequential::Build<CAPABILITY, MODULE_GRU, INPUT_SHAPE>;
    //        using MODEL = MODEL_GRU_TWO_LAYER;
            using MODEL = MODEL_GRU;
        };
        template <typename CAPABILITY>
        struct Critic{
            static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
            using GRU_SPEC = nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU_TEMPLATE = nn::layers::gru::BindConfiguration<GRU_SPEC>;
            using GRU2_SPEC = nn::layers::gru::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU2_TEMPLATE = nn::layers::gru::BindConfiguration<GRU2_SPEC>;
            using DENSE_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using DENSE_LAYER_TEMPLATE = nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
            using OUTPUT_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, 1, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<T, TI>, nn::parameters::groups::Normal>;
            using OUTPUT_LAYER_TEMPLATE = nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
            using MODEL_GRU_TWO_LAYER = Module<GRU_TEMPLATE, Module<GRU2_TEMPLATE, Module<DENSE_LAYER_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>>>;
            using INPUT_SHAPE = tensor::Shape<TI, SAC_PARAMETERS::SEQUENCE_LENGTH, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE, INPUT_DIM>;
            using MODULE_GRU = Module<GRU_TEMPLATE, Module<OUTPUT_LAYER_TEMPLATE>>;
            using MODEL_GRU = nn_models::sequential::Build<CAPABILITY, MODULE_GRU, INPUT_SHAPE>;
    //        using MODEL = MODEL_GRU_TWO_LAYER;
            using MODEL = MODEL_GRU;
        };

        using CAPABILITY_ACTOR = nn::capability::Gradient<nn::parameters::Adam>;
        using CAPABILITY_CRITIC = nn::capability::Gradient<nn::parameters::Adam>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
        using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
        using CRITIC_TARGET_TYPE = typename Critic<nn::capability::Forward<>>::MODEL;
        using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS>;
        using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS>;
        using ALPHA_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::ALPHA_OPTIMIZER_PARAMETERS>;
        using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
        using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
        using ALPHA_OPTIMIZER = nn::optimizers::Adam<ALPHA_OPTIMIZER_SPEC>;

    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

