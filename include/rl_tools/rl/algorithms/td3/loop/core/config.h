#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_CONFIG_H

#include "../../../../../nn_models/sequential/model.h"
#include "../../../../../nn_models/mlp/network.h"
#include "../../../../../nn_models/random_uniform/model.h"
#include "../../../../../rl/algorithms/td3/td3.h"
#include "../../../../../nn/optimizers/adam/adam.h"
#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::td3::loop::core{
    // Config State (Init/Step)

    template<typename T, typename TI, typename ENVIRONMENT>
    struct DefaultParameters{
        using TD3_PARAMETERS = rl::algorithms::td3::DefaultParameters<T, TI>;
        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI N_WARMUP_STEPS = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
        static constexpr TI STEP_LIMIT = 10000;
        static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT; // Note: when inheriting from this class for overwriting the default STEP_LIMIT you need to set the REPLAY_BUFFER_CAP as well otherwise it will be the default step limit
        static constexpr TI EPISODE_STEP_LIMIT = 200;

        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;

        static constexpr bool COLLECT_EPISODE_STATS = true;
        static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;

        static constexpr T EXPLORATION_NOISE = 0.1;
        static constexpr bool SHARED_BATCH = true;

        using OPTIMIZER_PARAMETERS = nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>;
    };

    // The approximator config sets up any types that support the usual rl_tools::forward and rl_tools::backward operations (can be custom as well)
    // We provide approximators based on the sequential and mlp models. The latter (mlp) allows for a variable number of layers, but is restricted to a uniform hidden layer size while the former allows for arbitrary layers to be combined in a sequential manner. Both support compile-time autodiff
    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
    struct ConfigApproximatorsSequential{
        template <nn::LayerCapability CAPABILITY, typename PARAMETER_TYPE>
        struct ACTOR{
            static constexpr TI HIDDEN_DIM = PARAMETERS::ACTOR_HIDDEN_DIM;
            static constexpr TI BATCH_SIZE = PARAMETERS::TD3_PARAMETERS::ACTOR_BATCH_SIZE;
            static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::ACTOR_ACTIVATION_FUNCTION;
            using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
            using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
            static constexpr TI ACTOR_OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
            using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ACTOR_OUTPUT_DIM, nn::activation_functions::ActivationFunction::TANH, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>;
        };

        template <nn::LayerCapability CAPABILITY, typename PARAMETER_TYPE>
        struct CRITIC{
            static constexpr TI HIDDEN_DIM = PARAMETERS::CRITIC_HIDDEN_DIM;
            static constexpr TI BATCH_SIZE = PARAMETERS::TD3_PARAMETERS::CRITIC_BATCH_SIZE;
            static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::CRITIC_ACTIVATION_FUNCTION;

            using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
            using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
            using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>;
        };

        using OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;

        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

        using ACTOR_TYPE = typename ACTOR<nn::LayerCapability::Gradient, nn::parameters::Adam>::MODEL;
        using ACTOR_TARGET_TYPE = typename ACTOR<nn::LayerCapability::Forward, nn::parameters::Adam>::MODEL;
        using CRITIC_TYPE = typename CRITIC<nn::LayerCapability::Gradient, nn::parameters::Adam>::MODEL;
        using CRITIC_TARGET_TYPE = typename CRITIC<nn::LayerCapability::Forward, nn::parameters::Adam>::MODEL;
    };

    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
    struct ConfigApproximatorsMLP{
        using ACTOR_STRUCTURE_SPEC = nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::activation_functions::TANH, PARAMETERS::TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CRITIC_STRUCTURE_SPEC = nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY, PARAMETERS::TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
        using OPTIMIZER_SPEC = typename nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;
        using ACTOR_SPEC = nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC >;
        using ACTOR_TYPE = nn_models::mlp::NeuralNetworkAdam<ACTOR_SPEC>;

        using ACTOR_TARGET_SPEC = nn_models::mlp::ForwardSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TARGET_TYPE = nn_models::mlp::NeuralNetworkForward<ACTOR_TARGET_SPEC>;

        using CRITIC_SPEC = nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TYPE = nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

        using CRITIC_TARGET_SPEC = nn_models::mlp::ForwardSpecification<CRITIC_STRUCTURE_SPEC >;
        using CRITIC_TARGET_TYPE = nn_models::mlp::NeuralNetworkForward<CRITIC_TARGET_SPEC>;
    };

    template<typename T_T, typename T_TI, typename T_RNG, typename T_ENVIRONMENT, typename T_PARAMETERS = DefaultParameters<T_T, T_TI, T_ENVIRONMENT>, template<typename, typename, typename, typename> class APPROXIMATOR_CONFIG=ConfigApproximatorsMLP>
    struct Config{
        using T = T_T;
        using TI = T_TI;
        using RNG = T_RNG;
        using ENVIRONMENT = T_ENVIRONMENT;
        using ENVIRONMENT_EVALUATION = T_ENVIRONMENT;

        using NN = APPROXIMATOR_CONFIG<T, TI, T_ENVIRONMENT, T_PARAMETERS>;
//        using NN = ConfigApproximatorsMLP<T, TI, T_ENVIRONMENT, T_PARAMETERS>;

        using CORE_PARAMETERS = T_PARAMETERS;

        using EXPLORATION_POLICY_SPEC = nn_models::random_uniform::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
        using EXPLORATION_POLICY = nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;

        using ACTOR_CRITIC_SPEC = rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, typename NN::ACTOR_TYPE, typename NN::ACTOR_TARGET_TYPE, typename NN::CRITIC_TYPE, typename NN::CRITIC_TARGET_TYPE, typename NN::OPTIMIZER, typename CORE_PARAMETERS::TD3_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        struct OFF_POLICY_RUNNER_PARAMETERS{
            static constexpr TI N_ENVIRONMENTS = CORE_PARAMETERS::N_ENVIRONMENTS;
            static constexpr bool ASYMMETRIC_OBSERVATIONS = false;
            static constexpr TI REPLAY_BUFFER_CAPACITY = CORE_PARAMETERS::REPLAY_BUFFER_CAP;
            static constexpr TI EPISODE_STEP_LIMIT = CORE_PARAMETERS::EPISODE_STEP_LIMIT;
            static constexpr bool STOCHASTIC_POLICY = false;
            static constexpr bool COLLECT_EPISODE_STATS = CORE_PARAMETERS::COLLECT_EPISODE_STATS;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = CORE_PARAMETERS::EPISODE_STATS_BUFFER_SIZE;
            static constexpr T EXPLORATION_NOISE = CORE_PARAMETERS::EXPLORATION_NOISE;
        };

        using OFF_POLICY_RUNNER_SPEC = rl::components::off_policy_runner::Specification<
                T,
                TI,
                ENVIRONMENT,
                OFF_POLICY_RUNNER_PARAMETERS
        >;
        static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        template <typename CONFIG>
        using State = State<CONFIG>;
    };
}

#endif

