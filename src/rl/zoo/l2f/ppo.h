#include "environment.h"

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<T, T, TI>::ENVIRONMENT;

        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            static constexpr TI STEP_LIMIT = 6000; // ~2.5M env steps

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI BATCH_SIZE = 2048 * 2;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            static constexpr bool NORMALIZE_OBSERVATIONS_CONTINUOUSLY = false;

            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI, BATCH_SIZE>{
                static constexpr TI N_EPOCHS = 1;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr T INITIAL_ACTION_STD = 0.5;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
                static constexpr bool NORMALIZE_ADVANTAGE = true;
                static constexpr T GAMMA = 0.99;
                static constexpr bool ADAPTIVE_LEARNING_RATE = false;
                static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
