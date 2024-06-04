#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::ppo::bottleneck_v0{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct BottleneckV0{
        struct ENVIRONMENT_PARAMETERS: rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<T, TI>{
            static constexpr TI N_AGENTS = 5;
            static constexpr TI LIDAR_RESOLUTION = 5;
            static constexpr T LIDAR_FOV = math::PI<T> * 90/180; // in radians (0 to PI)
            static constexpr T BOTTLENECK_WIDTH = 5;
            static constexpr TI EPISODE_STEP_LIMIT = 200;
            static constexpr bool SPAWN_BOTH_SIDES = false;
            static constexpr T AGENT_MAX_SPEED = 4;
            static constexpr T AGENT_MAX_ANGULAR_VELOCITY = 4;
            static constexpr T AGENT_MAX_ACCELERATION = 20;
            static constexpr T AGENT_MAX_ANGULAR_ACCELERATION = 20;
            static constexpr T DT = 0.05;
        };
        using ENVIRONMENT_SPEC = rlt::rl::environments::multi_agent::bottleneck::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::multi_agent::Bottleneck<ENVIRONMENT_SPEC>;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct PPO_PARAMETERS: rl::algorithms::ppo::DefaultParameters<T, TI>{
                static constexpr T GAMMA = 0.98;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
                static constexpr TI N_EPOCHS = 2;
                static constexpr bool IGNORE_TERMINATION = true;
//                static constexpr bool ADAPTIVE_LEARNING_RATE = true;
            };
            static constexpr TI STEP_LIMIT = 5000; // 1024 * 4 * 74 ~ 300k steps

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI CRITIC_HIDDEN_DIM = 128;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 128;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
            static constexpr TI BATCH_SIZE = 1024;
            struct OPTIMIZER_PARAMETERS: nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
                static constexpr T ALPHA = 0.1e-4;
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequentialMultiAgent>;
        template <typename BASE>
        struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{
//            static constexpr TI EPISODE_STEP_LIMIT = 20 / ENVIRONMENT_PARAMETERS::DT;
        };
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
