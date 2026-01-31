#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_CONFIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_CONFIG_H

#include "../../../../../nn/layers/standardize/layer.h"
#include "../../../../../nn_models/sequential/model.h"
#include "../../../../../nn_models/mlp_unconditional_stddev/network.h"
#include "../../../../../nn_models/multi_agent_wrapper/model.h"
#include "../../../../../rl/algorithms/ppo/ppo.h"
#include "../../../../../rl/components/on_policy_runner/on_policy_runner.h"
#include "../../../../../nn/optimizers/adam/adam.h"
#include "../../../../../rl/loop/loop.h"
#include "state.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::loop::core{
        // Config State (Init/Step)

        struct ParametersTag{};
        template<typename TYPE_POLICY, typename TI, typename ENVIRONMENT>
        struct DefaultParameters{
            using T = typename TYPE_POLICY::DEFAULT;
            using TAG = ParametersTag;
            static constexpr TI STEP_LIMIT = 100;

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;

            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
            static constexpr TI DATASET_SIZE = ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS;
            static constexpr TI BATCH_SIZE = 64 * 8;

            static constexpr bool NORMALIZE_OBSERVATIONS = false;
            static constexpr bool NORMALIZE_OBSERVATIONS_CONTINUOUSLY = false;

            using ACTOR_OPTIMIZER_PARAMETERS = nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>;
            using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;

            using PPO_PARAMETERS = rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>;
        };

        template<typename TYPE_POLICY, typename TI, typename ENVIRONMENT, typename PARAMETERS, bool DYNAMIC_ALLOCATION=true>
        struct ConfigApproximatorsSequential{
            static constexpr TI STEPS = PARAMETERS::PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV : 1;
            static constexpr TI FORWARD_BATCH_SIZE = PARAMETERS::PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? PARAMETERS::N_ENVIRONMENTS : PARAMETERS::BATCH_SIZE;
            template <typename CAPABILITY>
            struct Actor{
                using INPUT_SHAPE = tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
                using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  nn::activation_functions::IDENTITY>;
                using TYPE = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

                template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

                using MODULE_CHAIN = Module<STANDARDIZATION_LAYER, Module<TYPE>>;
                using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
            };
            template <typename CAPABILITY>
            struct Critic{
                using INPUT_SHAPE = tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ENVIRONMENT::ObservationPrivileged::DIM>;
                using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
                using TYPE = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

                template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

                using MODULE_CHAIN = Module<STANDARDIZATION_LAYER, Module<TYPE>>;
                using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
            };

            using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
            using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
            using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
            using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
            using CAPABILITY_ADAM = nn::capability::Gradient<nn::parameters::Adam, DYNAMIC_ALLOCATION>;
            using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
            using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
        };
        template <bool T_GRU_CRITIC = false>
        struct ConfigApproximatorsGRU{
            template<typename TYPE_POLICY, typename TI, typename ENVIRONMENT, typename PARAMETERS, bool DYNAMIC_ALLOCATION=true>
            struct Approximators{
                static constexpr bool USE_GRU = true;
                using PPO_PARAMETERS = typename PARAMETERS::PPO_PARAMETERS;
                static_assert(PPO_PARAMETERS::SHUFFLE_EPOCH == false, "When using sequence models for the actor and critic, SHUFFLE_EPOCH has to be disabled (in the PPO parameters) because it scrambles the episodes.");
                static_assert(PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC == true, "When using sequence models for the actor and critic, STATEFUL_ACTOR_AND_CRITIC has to be enabled.");
                template <typename CAPABILITY>
                struct Actor{
                    using INPUT_SHAPE = tensor::Shape<TI, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, PARAMETERS::N_ENVIRONMENTS, ENVIRONMENT::Observation::DIM>;
                    using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                    using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                    using INPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>;
                    using INPUT_LAYER = nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
                    using GRU_SPEC = nn::layers::gru::Configuration<TYPE_POLICY, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal>;
                    using GRU = nn::layers::gru::BindConfiguration<GRU_SPEC>;
                    using DENSE_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Normal>;
                    using DENSE_LAYER = nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
                    using OUTPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, nn::activation_functions::IDENTITY, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Output>;
                    using OUTPUT_LAYER = nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
                    using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  nn::activation_functions::IDENTITY>;
                    using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

                    template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                    using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
                    using MODULE = Module<STANDARDIZATION_LAYER, Module<INPUT_LAYER, Module<GRU, Module<MLP>>>>;
                    using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>;
                };
                template <typename CAPABILITY>
                struct Critic{
                    using INPUT_SHAPE = tensor::Shape<TI, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, PARAMETERS::N_ENVIRONMENTS, ENVIRONMENT::ObservationPrivileged::DIM>;
                    using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                    using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                    using INPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>;
                    using INPUT_LAYER = nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
                    using GRU_SPEC = nn::layers::gru::Configuration<TYPE_POLICY, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal>;
                    using GRU = nn::layers::gru::BindConfiguration<GRU_SPEC>;
                    using DENSE_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Normal>;
                    using DENSE_LAYER = nn::layers::dense::BindConfiguration<DENSE_LAYER_CONFIG>;
                    using OUTPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, nn::activation_functions::IDENTITY, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Output>;
                    using OUTPUT_LAYER = nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
                    using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
                    using MLP = nn_models::mlp::BindConfiguration<CONFIG>;
                    template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                    using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
                    using MODULE_DENSE = Module<STANDARDIZATION_LAYER, Module<INPUT_LAYER, Module<DENSE_LAYER, Module<MLP>>>>;
                    using MODULE_GRU = Module<STANDARDIZATION_LAYER, Module<INPUT_LAYER, Module<GRU, Module<MLP>>>>;
                    using MODULE = rl_tools::utils::typing::conditional_t<T_GRU_CRITIC, MODULE_GRU, MODULE_DENSE>;
                    using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>;
                };

                using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
                using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
                using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
                using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
                using CAPABILITY_ADAM = nn::capability::Gradient<nn::parameters::Adam, DYNAMIC_ALLOCATION>;
                using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
                using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
            };
        };

        template<typename TYPE_POLICY, typename TI, typename ENVIRONMENT, typename PARAMETERS, bool DYNAMIC_ALLOCATION=true>
        struct ConfigApproximatorsSequentialMultiAgent{
            template <typename CAPABILITY>
            struct Actor{
                static constexpr TI N_AGENTS = ENVIRONMENT::N_AGENTS;
                static_assert(ENVIRONMENT::Observation::DIM % N_AGENTS == 0);
                static_assert(ENVIRONMENT::ACTION_DIM % N_AGENTS == 0);
                using INPUT_SHAPE = tensor::Shape<TI, 1, PARAMETERS::BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
                using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM/N_AGENTS, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  nn::activation_functions::IDENTITY>;
                using TYPE = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

                template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

                using INNER_MODULE_CHAIN = Module<STANDARDIZATION_LAYER, Module<TYPE>>;
                using WRAPPER_CONFIG = nn_models::multi_agent_wrapper::Configuration<TYPE_POLICY, TI, N_AGENTS, INNER_MODULE_CHAIN>;
                using MODEL = nn_models::multi_agent_wrapper::Build<CAPABILITY, WRAPPER_CONFIG, INPUT_SHAPE>;
            };
            template <typename CAPABILITY>
            struct Critic{
                using INPUT_SHAPE = tensor::Shape<TI, 1, PARAMETERS::BATCH_SIZE, ENVIRONMENT::ObservationPrivileged::DIM>;
                using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
                using TYPE = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;
//                using IF = nn_models::sequential::Interface<CAPABILITY>;
//                using CRITIC_MODULE = typename IF::template Module<TYPE::template NeuralNetwork>;
                using STANDARDIZATION_LAYER_SPEC = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
                using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_SPEC>;
//                using MODEL = typename IF::template Module<STANDARDIZATION_LAYER::template Layer, CRITIC_MODULE>;
                template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
                using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

                using MODULE_CHAIN = Module<STANDARDIZATION_LAYER, Module<TYPE>>;
                using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;


//                using WRAPPER_CONFIG = nn_models::multi_agent_wrapper::Configuration<T, TI, N_AGENTS, INNER_MODULE_CHAIN>;
//                using WRAPPER = nn_models::multi_agent_wrapper::BindConfiguration<WRAPPER_CONFIG>;
//                using MODULE_CHAIN = Module<WRAPPER>;
//                using MODEL = nn_models::multi_agent_wrapper::Module<WRAPPER_CONFIG, CAPABILITY, INPUT_SHAPE>;
            };

            using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS>;
            using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS>;
            using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
            using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
            using CAPABILITY_ADAM = nn::capability::Gradient<nn::parameters::Adam, DYNAMIC_ALLOCATION>;
            using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
            using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
        };

        struct ConfigTag{};
        template<typename T_TYPE_POLICY, typename T_TI, typename T_RNG, typename T_ENVIRONMENT, typename T_PARAMETERS = DefaultParameters<T_TYPE_POLICY, T_TI, T_ENVIRONMENT>, template<typename, typename, typename, typename, bool> class APPROXIMATOR_CONFIG=ConfigApproximatorsSequential, bool T_DYNAMIC_ALLOCATION=true>
        struct Config: rl::loop::Config{
            using TAG = ConfigTag;
            using TYPE_POLICY = T_TYPE_POLICY;
            using TI = T_TI;
            using RNG = T_RNG;
            using ENVIRONMENT = T_ENVIRONMENT;
            using ENVIRONMENT_EVALUATION = T_ENVIRONMENT;
            using CORE_PARAMETERS = T_PARAMETERS;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;

            static constexpr TI ENVIRONMENT_STEPS_PER_LOOP_STEP = CORE_PARAMETERS::N_ENVIRONMENTS * CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;

            using NN = APPROXIMATOR_CONFIG<TYPE_POLICY, TI, ENVIRONMENT, CORE_PARAMETERS, DYNAMIC_ALLOCATION>;


            using T = typename TYPE_POLICY::DEFAULT;
            static constexpr T OBSERVATION_NORMALIZATION_WARMUP_STEPS = CORE_PARAMETERS::NORMALIZE_OBSERVATIONS ? 1 : 0;
            using PPO_SPEC = rl::algorithms::ppo::Specification<TYPE_POLICY, TI, ENVIRONMENT, typename NN::ACTOR_TYPE, typename NN::CRITIC_TYPE, typename CORE_PARAMETERS::PPO_PARAMETERS>;
            using PPO_TYPE = rl::algorithms::PPO<PPO_SPEC>;
            using PPO_BUFFERS_TYPE = rl::algorithms::ppo::Buffers<rl::algorithms::ppo::BufferSpecification<PPO_SPEC, DYNAMIC_ALLOCATION>>;
            static constexpr TI NUM_NNS = 2;

            static constexpr TI AGENTS_PER_ENV = 1;
            using ON_POLICY_RUNNER_SPEC = rl::components::on_policy_runner::Specification<TYPE_POLICY, TI, ENVIRONMENT, typename NN::ACTOR_TYPE::template State<DYNAMIC_ALLOCATION>, CORE_PARAMETERS::N_ENVIRONMENTS, CORE_PARAMETERS::EPISODE_STEP_LIMIT, AGENTS_PER_ENV, CORE_PARAMETERS::PPO_PARAMETERS::TRUNCATE_ON_EACH_ITERATION, DYNAMIC_ALLOCATION>;
            using ON_POLICY_RUNNER_TYPE = rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
            using ON_POLICY_RUNNER_DATASET_SPEC = rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, DYNAMIC_ALLOCATION>;
            using ON_POLICY_RUNNER_DATASET_TYPE = rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;


            using ACTOR_EVAL_BUFFERS = typename NN::ACTOR_TYPE::template Buffer<DYNAMIC_ALLOCATION>;
            using ACTOR_BUFFERS = typename NN::ACTOR_TYPE::template Buffer<DYNAMIC_ALLOCATION>;
            using CRITIC_BUFFERS = typename NN::CRITIC_TYPE::template Buffer<DYNAMIC_ALLOCATION>;
            using CRITIC_GAE = typename NN::CRITIC_TYPE::template CHANGE_BATCH_SIZE<TI, ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
            using CRITIC_BUFFERS_GAE = typename CRITIC_GAE::template Buffer<DYNAMIC_ALLOCATION>;
            template <typename CONFIG>
            using State = State<CONFIG>;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

