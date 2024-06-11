#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_CONFIG_H

#include "../../../../../nn_models/sequential/model.h"
#include "../../../../../nn_models/mlp/network.h"
#include "../../../../../nn_models/random_uniform/model.h"
#include "../../../../../rl/algorithms/sac/sac.h"
#include "../../../../../nn/optimizers/adam/adam.h"
#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac::loop::core{
    // Config State (Init/Step)
    template<typename T, typename TI, typename ENVIRONMENT>
    struct DefaultParameters{
        using SAC_PARAMETERS = rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>;
        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI N_WARMUP_STEPS = 100;
        static_assert(N_WARMUP_STEPS >= SAC_PARAMETERS::ACTOR_BATCH_SIZE);
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

        static constexpr bool SHARED_BATCH = true;

        static constexpr T TARGET_ENTROPY = -((T)ENVIRONMENT::ACTION_DIM);
        static constexpr T ALPHA = 0.5;
        static constexpr bool ADAPTIVE_ALPHA = true;
        static constexpr T LOG_STD_LOWER_BOUND = -20;
        static constexpr T LOG_STD_UPPER_BOUND = 2;
        static constexpr T LOG_PROBABILITY_EPSILON = 1e-6;

        using OPTIMIZER_PARAMETERS = nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>;
    };

    // The approximator config sets up any types that support the usual rl_tools::forward and rl_tools::backward operations (can be custom as well)
    // We provide approximators based on the sequential and mlp models. The latter (mlp) allows for a variable number of layers, but is restricted to a uniform hidden layer size while the former allows for arbitrary layers to be combined in a sequential manner. Both support compile-time autodiff
    template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS, typename CONTAINER_TYPE_TAG>
    struct ConfigApproximatorsSequential{
        template <typename CAPABILITY>
        struct Actor{
            using ACTOR_SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 2*ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  nn::activation_functions::IDENTITY>;
            using ACTOR_TYPE = nn_models::mlp::BindSpecification<ACTOR_SPEC>;
            using IF = nn_models::sequential::Interface<CAPABILITY>;
            struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
                static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
                static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
                static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
                static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
                static constexpr T ALPHA = PARAMETERS::ALPHA;
                static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
            };
            using SAMPLE_AND_SQUASH_LAYER_SPEC = nn::layers::sample_and_squash::Specification<T, TI, ENVIRONMENT::ACTION_DIM, SAMPLE_AND_SQUASH_LAYER_PARAMETERS>;
            using SAMPLE_AND_SQUASH_LAYER = nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
            using SAMPLE_AND_SQUASH_MODULE = typename IF::template Module<SAMPLE_AND_SQUASH_LAYER::template Layer>;
            using MODEL = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>;
        };
        template <typename CAPABILITY>
        struct Critic{
            static constexpr TI INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM+ENVIRONMENT::ACTION_DIM;
            using SPEC = nn_models::mlp::Specification<T, TI, INPUT_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
            using TYPE = nn_models::mlp::BindSpecification<SPEC>;
            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<TYPE::template NeuralNetwork>;
        };

        using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI>;
        using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI>;
        using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
        using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
        using CAPABILITY_ACTOR = nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CAPABILITY_CRITIC = nn::layer_capability::Gradient<nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
        using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
        using CRITIC_TARGET_TYPE = typename Critic<nn::layer_capability::Forward>::MODEL;
        using OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
        using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

    };

    template<typename T_T, typename T_TI, typename T_RNG, typename T_ENVIRONMENT, typename T_PARAMETERS = DefaultParameters<T_T, T_TI, T_ENVIRONMENT>, template<typename, typename, typename, typename, typename> class APPROXIMATOR_CONFIG=ConfigApproximatorsSequential, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Config{
        using T = T_T;
        using TI = T_TI;
        using RNG = T_RNG;
        using ENVIRONMENT = T_ENVIRONMENT;
        using ENVIRONMENT_EVALUATION = T_ENVIRONMENT;
        using CORE_PARAMETERS = T_PARAMETERS;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;

        static constexpr TI ENVIRONMENT_STEPS_PER_LOOP_STEP = CORE_PARAMETERS::N_ENVIRONMENTS;

        using NN = APPROXIMATOR_CONFIG<T, TI, T_ENVIRONMENT, CORE_PARAMETERS, CONTAINER_TYPE_TAG>;
//        using NN = ConfigApproximatorsMLP<T, TI, T_ENVIRONMENT, T_PARAMETERS>;

        using EXPLORATION_POLICY_SPEC = nn_models::random_uniform::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
        using EXPLORATION_POLICY = nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;

        using ALPHA_PARAMETER_TYPE = nn::parameters::Adam;
        using ALPHA_OPTIMIZER = nn::optimizers::Adam<typename NN::OPTIMIZER_SPEC>;

        using ACTOR_CRITIC_SPEC = rl::algorithms::sac::Specification<T, TI, ENVIRONMENT, typename NN::ACTOR_TYPE, typename NN::CRITIC_TYPE, typename NN::CRITIC_TARGET_TYPE, ALPHA_PARAMETER_TYPE, typename NN::OPTIMIZER, typename NN::OPTIMIZER, ALPHA_OPTIMIZER, typename CORE_PARAMETERS::SAC_PARAMETERS, CONTAINER_TYPE_TAG>;
        using ACTOR_CRITIC_TYPE = rl::algorithms::sac::ActorCritic<ACTOR_CRITIC_SPEC>;

        struct OFF_POLICY_RUNNER_PARAMETERS{
            static constexpr TI N_ENVIRONMENTS = CORE_PARAMETERS::N_ENVIRONMENTS;
            static constexpr bool ASYMMETRIC_OBSERVATIONS = false;
            static constexpr TI REPLAY_BUFFER_CAPACITY = CORE_PARAMETERS::REPLAY_BUFFER_CAP;
            static constexpr TI EPISODE_STEP_LIMIT = CORE_PARAMETERS::EPISODE_STEP_LIMIT;
            static constexpr bool STOCHASTIC_POLICY = true;
            static constexpr bool COLLECT_EPISODE_STATS = CORE_PARAMETERS::COLLECT_EPISODE_STATS;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = CORE_PARAMETERS::EPISODE_STATS_BUFFER_SIZE;
            static constexpr T EXPLORATION_NOISE = 0.1;
        };

        using OFF_POLICY_RUNNER_SPEC = rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, OFF_POLICY_RUNNER_PARAMETERS, CONTAINER_TYPE_TAG>;
        static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        template <typename CONFIG>
        using State = State<CONFIG>;
    };
}

#endif

