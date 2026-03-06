#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::reacher_v0::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
            static constexpr TI BATCH_SIZE = 512;
            static constexpr TI TOTAL_STEP_LIMIT = 1000000;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
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
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
