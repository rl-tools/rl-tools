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
        template <typename CAPABILITY>
        struct ACTOR{
            static constexpr TI HIDDEN_DIM = PARAMETERS::ACTOR_HIDDEN_DIM;
            static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::ACTOR_ACTIVATION_FUNCTION;
            using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION>;
            using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
            using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION>;
            using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
            static constexpr TI ACTOR_OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
            using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ACTOR_OUTPUT_DIM, nn::activation_functions::ActivationFunction::TANH>;
            using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>;
        };

        template <typename CAPABILITY>
        struct CRITIC{
            static constexpr TI HIDDEN_DIM = PARAMETERS::CRITIC_HIDDEN_DIM;
            static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::CRITIC_ACTIVATION_FUNCTION;

            using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION>;
            using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
            using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION>;
            using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
            using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, nn::activation_functions::ActivationFunction::IDENTITY>;
            using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>;
        };

        using OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;

        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

        using ACTOR_TYPE = typename ACTOR<nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::TD3_PARAMETERS::ACTOR_BATCH_SIZE>>::MODEL;
        using ACTOR_TARGET_TYPE = typename ACTOR<nn::layer_capability::Forward>::MODEL;
        using CRITIC_TYPE = typename CRITIC<nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::TD3_PARAMETERS::CRITIC_BATCH_SIZE>>::MODEL;
        using CRITIC_TARGET_TYPE = typename CRITIC<nn::layer_capability::Forward>::MODEL;
    };

    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
    struct ConfigApproximatorsMLP{
        using ACTOR_SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::activation_functions::TANH>;
        using CRITIC_SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
        using OPTIMIZER_SPEC = typename nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

        using ACTOR_CAPABILITY = nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
        using ACTOR_TYPE = nn_models::mlp::NeuralNetwork<ACTOR_CAPABILITY, ACTOR_SPEC>;
        using ACTOR_TARGET_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<nn::layer_capability::Forward>; //nn_models::mlp::NeuralNetwork<nn::layer_capability::Forward, ACTOR_SPEC>; // todo: replace with something like: ACTOR_TYPE::CHANGE_CAPABILITY<nn::layer_capability::Forward>

        using CRITIC_CAPABILITY = nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
        using CRITIC_TYPE = nn_models::mlp::NeuralNetwork<CRITIC_CAPABILITY, CRITIC_SPEC>;
        using CRITIC_TARGET_TYPE = typename CRITIC_TYPE::template CHANGE_CAPABILITY<nn::layer_capability::Forward>; //nn_models::mlp::NeuralNetwork<nn::layer_capability::Forward, CRITIC_SPEC>;
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

        static constexpr TI ENVIRONMENT_STEPS_PER_LOOP_STEP = CORE_PARAMETERS::N_ENVIRONMENTS;

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

