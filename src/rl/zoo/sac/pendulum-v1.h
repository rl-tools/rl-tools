#include <rl_tools/rl/environments/pendulum/operations_cpu.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>

namespace rlt = rl_tools;

namespace rl_tools::rl::zoo::sac{
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct PendulumV1{
        struct _EnvironmentSpec{ // underscore such that only the ENVIRONMENT and LOOP_CONFIG are exposed
            using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
        };
        template <typename ENVIRONMENT>
        struct _LoopSpec{ // underscore such that only the ENVIRONMENT and LOOP_CONFIG are exposed
            struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
                static constexpr TI STEP_LIMIT = 20000;
                static constexpr TI ACTOR_NUM_LAYERS = 3;
                static constexpr TI ACTOR_HIDDEN_DIM = 64;
                static constexpr TI CRITIC_NUM_LAYERS = 3;
                static constexpr TI CRITIC_HIDDEN_DIM = 64;
            };
            using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsSequential>;
            using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;
            struct LOOP_CHECKPOINT_PARAMETERS: rlt::rl::loop::steps::checkpoint::Parameters<T, TI>{
                static constexpr TI CHECKPOINT_INTERVAL = 1000;
            };
            using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
            struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
                static constexpr TI NUM_EVALUATION_EPISODES = 100;
            };
            using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVAL_PARAMETERS>;
            struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{ };
            using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVAL_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
            using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
            using LOOP_CONFIG = LOOP_TIMING_CONFIG;
        };
        using ENVIRONMENT = rlt::rl::environments::Pendulum<typename _EnvironmentSpec::PENDULUM_SPEC>;
        using LOOP_CONFIG = typename _LoopSpec<ENVIRONMENT>::LOOP_CONFIG;
    };
}
