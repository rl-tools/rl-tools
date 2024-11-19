#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::pendulum_v1::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            static constexpr TI N_ENVIRONMENTS = 8;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
            static constexpr TI BATCH_SIZE = 128;
            static constexpr TI TOTAL_STEP_LIMIT = 1000000;
            static constexpr TI ACTOR_HIDDEN_DIM = 32;
            static constexpr TI CRITIC_HIDDEN_DIM = 32;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
                static constexpr T ALPHA = 0.001;
            };
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI, BATCH_SIZE>{
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
                static constexpr TI N_EPOCHS = 1;
                static constexpr T GAMMA = 0.9;
                static constexpr T INITIAL_ACTION_STD = 2.0;
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
