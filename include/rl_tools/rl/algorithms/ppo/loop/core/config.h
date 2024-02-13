#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_CONFIG_H

#include "../../../../../nn_models/sequential/model.h"
#include "../../../../../nn_models/mlp_unconditional_stddev/network.h"
#include "../../../../../rl/algorithms/ppo/ppo.h"
#include "../../../../../rl/components/on_policy_runner/on_policy_runner.h"
#include "../../../../../nn/optimizers/adam/adam.h"
#include "../../../../../rl/loop/loop.h"
#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::loop::core{
        // Config State (Init/Step)
        using namespace nn_models::sequential::interface;

        struct ParametersTag{};
        template<typename T, typename TI, typename ENVIRONMENT>
        struct Parameters{
            using TAG = ParametersTag;
            using PPO_PARAMETERS = rl::algorithms::ppo::DefaultParameters<T, TI>;
            static constexpr TI STEP_LIMIT = 100;

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;

            static constexpr TI EPISODE_STEP_LIMIT = 1000;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
            static constexpr TI BATCH_SIZE = ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS;
        };

        template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS>
        struct ConfigApproximatorsMLP{
            using ACTOR_STRUCTURE_SPEC = nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY, PARAMETERS::BATCH_SIZE>;
            using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI>;
            using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI>;
            using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
            using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
            using ACTOR_SPEC = nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
            using ACTOR_TYPE = nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
            using ACTOR_TYPE_INFERENCE = nn_models::mlp_unconditional_stddev::NeuralNetwork<ACTOR_SPEC>;
            using CRITIC_STRUCTURE_SPEC = nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY, PARAMETERS::BATCH_SIZE>;
            using CRITIC_SPEC = nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
            using CRITIC_TYPE = nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;
        };

        struct ConfigTag{};
        template<typename T_T, typename T_TI, typename T_RNG, typename T_ENVIRONMENT, typename T_PARAMETERS = Parameters<T_T, T_TI, T_ENVIRONMENT>, template<typename, typename, typename, typename> class APPROXIMATOR_CONFIG=ConfigApproximatorsMLP>
        struct Config: rl::loop::Config{
            using TAG = ConfigTag;
            using T = T_T;
            using TI = T_TI;
            using RNG = T_RNG;
            using ENVIRONMENT = T_ENVIRONMENT;
            using ENVIRONMENT_EVALUATION = T_ENVIRONMENT;
            using CORE_PARAMETERS = T_PARAMETERS;

            using NN = APPROXIMATOR_CONFIG<T, TI, ENVIRONMENT, CORE_PARAMETERS>;
//        using NN = ConfigApproximatorsMLP<T, TI, T_ENVIRONMENT, T_PARAMETERS>;


            static constexpr T OBSERVATION_NORMALIZATION_WARMUP_STEPS = CORE_PARAMETERS::PPO_PARAMETERS::NORMALIZE_OBSERVATIONS ? 1 : 0;
            using PPO_SPEC = rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, typename NN::ACTOR_TYPE, typename NN::CRITIC_TYPE, typename CORE_PARAMETERS::PPO_PARAMETERS>;
            using PPO_TYPE = rl::algorithms::PPO<PPO_SPEC>;
            using PPO_BUFFERS_TYPE = rl::algorithms::ppo::Buffers<PPO_SPEC>;

            using ON_POLICY_RUNNER_SPEC = rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, CORE_PARAMETERS::N_ENVIRONMENTS, CORE_PARAMETERS::EPISODE_STEP_LIMIT>;
            using ON_POLICY_RUNNER_TYPE = rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
            using ON_POLICY_RUNNER_DATASET_SPEC = rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV>;
            using ON_POLICY_RUNNER_DATASET_TYPE = rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;


            using ACTOR_EVAL_BUFFERS = typename NN::ACTOR_TYPE::template Buffer<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
            using ACTOR_BUFFERS = typename NN::ACTOR_TYPE::template Buffer<CORE_PARAMETERS::BATCH_SIZE>;
            using CRITIC_BUFFERS = typename NN::CRITIC_TYPE::template Buffer<CORE_PARAMETERS::BATCH_SIZE>;
            using CRITIC_BUFFERS_GAE = typename NN::CRITIC_TYPE::template Buffer<ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
            template <typename CONFIG>
            using State = State<CONFIG>;
        };
    }
}

#endif

