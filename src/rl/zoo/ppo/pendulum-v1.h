#include <rl_tools/rl/environments/pendulum/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::ppo::pendulum_v1{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct PendulumV1{
        using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
        using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct PPO_PARAMETERS: rl::algorithms::ppo::DefaultParameters<T, TI>{
                static constexpr T GAMMA = 0.95;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 1.0;
                static constexpr TI N_EPOCHS = 2;
            };
            static constexpr TI STEP_LIMIT = 74; // 1024 * 4 * 74 ~ 300k steps

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 4;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024;
            static constexpr TI BATCH_SIZE = 256;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
