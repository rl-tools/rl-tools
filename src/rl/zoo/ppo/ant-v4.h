#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::ppo::ant_v4{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct AntV4{
        using T_ENVIRONMENT = double;
        using ENVIRONMENT_PARAMETERS = rlt::rl::environments::mujoco::ant::DefaultParameters<T_ENVIRONMENT, TI>;
        using ENVIRONMENT_SPEC = rlt::rl::environments::mujoco::ant::Specification<T_ENVIRONMENT, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;

        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
                static constexpr TI N_EPOCHS = 4;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr T INITIAL_ACTION_STD = 0.5;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
                static constexpr bool NORMALIZE_ADVANTAGE = false;
                static constexpr T GAMMA = 0.99;
                static constexpr bool ADAPTIVE_LEARNING_RATE = true;
                static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
            };
//            static constexpr TI STEP_LIMIT = 600; // ~2.5M env steps
            static constexpr TI STEP_LIMIT = 200; // ~2.5M env steps

            static constexpr TI ACTOR_HIDDEN_DIM = 256;
            static constexpr TI CRITIC_HIDDEN_DIM = 256;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI BATCH_SIZE = 2048;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            static constexpr bool NORMALIZE_OBSERVATIONS_CONTINUOUSLY = false;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
