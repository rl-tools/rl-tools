#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::ppo{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct BottleneckV0{
        struct ENVIRONMENT_PARAMETERS: rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<T, TI>{
            static constexpr TI N_AGENTS = 2;
            static constexpr T BOTTLENECK_WIDTH = 5;
        };
        using ENVIRONMENT_SPEC = rlt::rl::environments::multi_agent::bottleneck::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::multi_agent::Bottleneck<ENVIRONMENT_SPEC>;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct PPO_PARAMETERS: rl::algorithms::ppo::DefaultParameters<T, TI>{
                static constexpr T GAMMA = 0.99;
//                static constexpr T ACTION_ENTROPY_COEFFICIENT = 1.0;
                static constexpr TI N_EPOCHS = 2;
            };
            static constexpr TI STEP_LIMIT = 5000; // 1024 * 4 * 74 ~ 300k steps

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
            static constexpr TI BATCH_SIZE = 256;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
