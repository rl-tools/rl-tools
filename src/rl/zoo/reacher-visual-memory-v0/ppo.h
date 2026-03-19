#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/unflatten/layer.h>
#include <rl_tools/nn/layers/flatten/layer.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::reacher_visual_memory_v0::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI N_ENVIRONMENTS = 128;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI BATCH_SIZE = N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI TOTAL_STEP_LIMIT = N_ENVIRONMENTS * ON_POLICY_RUNNER_STEPS_PER_ENV * 1500;
            static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            struct ACTOR_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
                static constexpr T ALPHA = 3e-4;
            };
            using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.001;
                static constexpr TI N_EPOCHS = 1;
                static constexpr T GAMMA = 0.9;
                static constexpr T LAMBDA = 0.95;
                static constexpr T EPSILON_CLIP = 0.2;
                static constexpr T INITIAL_ACTION_STD = 0.6065306597633104;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr bool SHUFFLE_EPOCH = false;
                static constexpr bool STATEFUL_ACTOR_AND_CRITIC = true;
                static constexpr bool TRUNCATE_ON_EACH_ITERATION = true;
            };
        };

        template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION=true>
        struct ConfigApproximatorsCNNGRU{
            static constexpr bool USE_GRU = true;
            using PPO_PARAMETERS = typename PARAMETERS::PPO_PARAMETERS;
            static_assert(PPO_PARAMETERS::SHUFFLE_EPOCH == false);
            static_assert(PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC == true);
            template <typename CAPABILITY>
            struct Actor{
                using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
                using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, PARAMETERS::N_ENVIRONMENTS>, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV>;
                static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
                static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;
                static constexpr T_TI IMG_C = T_ENVIRONMENT::Observation::CHANNELS;
                using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
                using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
                using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
                using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<T_TYPE_POLICY, T_TI, IMG_H, IMG_W, IMG_C>;
                using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
                using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
                using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
                using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
                using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
                using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
                using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
                using INPUT_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T_TYPE_POLICY, T_TI>, rlt::nn::parameters::groups::Input>;
                using INPUT_DENSE = rlt::nn::layers::dense::BindConfiguration<INPUT_DENSE_CONFIG>;
                static constexpr bool FAST_TANH = false;
                using GRU_SPEC = rlt::nn::layers::gru::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, FAST_TANH>;
                using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_SPEC>;
                using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
                using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;

                using MODULE_CHAIN = rlt::nn_models::sequential::Module<INPUT_FLATTEN, STANDARDIZATION_LAYER, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, INPUT_DENSE, GRU, MLP>;
                using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
            };
            template <typename CAPABILITY>
            struct Critic{
                using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
                using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, PARAMETERS::N_ENVIRONMENTS>, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV>;
                using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
                using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::layers::dense::DefaultInitializer<T_TYPE_POLICY, T_TI>, rlt::nn::parameters::groups::Input>;
                using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
                static constexpr bool FAST_TANH = false;
                using GRU_SPEC = rlt::nn::layers::gru::Configuration<T_TYPE_POLICY, T_TI, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, FAST_TANH>;
                using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_SPEC>;
                using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
                using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;

                using MODULE = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<INPUT_LAYER, rlt::nn_models::sequential::Module<GRU, rlt::nn_models::sequential::Module<MLP>>>>;
                using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>;
            };

            using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
            using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
            using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
            using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
            using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, T_DYNAMIC_ALLOCATION>;
            using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
            using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
        };

        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsCNNGRU, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
