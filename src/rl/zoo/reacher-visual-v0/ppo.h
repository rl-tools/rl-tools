#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/unflatten/layer.h>
#include <rl_tools/nn/layers/flatten/layer.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::reacher_visual_v0::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI BATCH_SIZE = 512;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI TOTAL_STEP_LIMIT = 1000000;
            static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            struct ACTOR_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
                static constexpr T ALPHA = 3e-4;
                static constexpr T EPSILON = 1e-5;
                static constexpr T EPSILON_SQRT = 1e-5;
            };
            using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
                static constexpr TI N_EPOCHS = 4;
                static constexpr T GAMMA = 0.9;
                static constexpr T LAMBDA = 0.95;
                static constexpr T EPSILON_CLIP = 0.2;
                static constexpr T INITIAL_ACTION_STD = 0.6065306597633104; // exp(-0.5)
            };
        };

        template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION=true>
        struct ConfigApproximatorsCNN{
            static constexpr T_TI STEPS = 1;
            static constexpr T_TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;
            template <typename CAPABILITY>
            struct Actor{
                using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
                using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                // Standardize(H*W*C) -> Unflatten(H,W,C) -> Conv1 -> Conv2 -> Flatten -> MLP
                static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
                static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;
                static constexpr T_TI IMG_C = T_ENVIRONMENT::Observation::CHANNELS;
                using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
                using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<T_TYPE_POLICY, T_TI, IMG_H, IMG_W, IMG_C>;
                using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
                using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
                using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
                using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
                using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
                using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
                using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
                using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
                using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;

                using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<UNFLATTEN, rlt::nn_models::sequential::Module<CONV1, rlt::nn_models::sequential::Module<CONV2, rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<MLP>>>>>>;
                using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
            };
            template <typename CAPABILITY>
            struct Critic{
                using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
                using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
                using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
                using CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
                using TYPE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

                using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<TYPE>>;
                using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
            };

            using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
            using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
            using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
            using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
            using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, T_DYNAMIC_ALLOCATION>;
            using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
            using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
        };

        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsCNN, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
