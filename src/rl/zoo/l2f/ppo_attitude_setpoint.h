#include "environment_attitude_setpoint.h"

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f::ppo_attitude_setpoint{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using BASE = ENVIRONMENT_ATTITUDE_SETPOINT_FACTORY<DEVICE, TYPE_POLICY, TI>;
        using PARAMETERS_TYPE = typename BASE::PARAMETERS_TYPE;
        static constexpr PARAMETERS_TYPE nominal_parameters = [](){
            auto parameters = BASE::nominal_parameters;
            parameters.mdp.termination.enabled = false;
            return parameters;
        }();
        struct ENVIRONMENT_STATIC_PARAMETERS: BASE::ENVIRONMENT_STATIC_PARAMETERS{
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
        };
        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;

        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI STEP_LIMIT = 2500;

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 32;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
            static constexpr TI BATCH_SIZE = 2048;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            static constexpr bool NORMALIZE_OBSERVATIONS_CONTINUOUSLY = false;

            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                using T = typename TYPE_POLICY::DEFAULT;
                static constexpr TI N_EPOCHS = 2;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr T INITIAL_ACTION_STD = 0.5;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.005;
                static constexpr bool NORMALIZE_ADVANTAGE = true;
                static constexpr T GAMMA = 0.98;
                static constexpr bool ADAPTIVE_LEARNING_RATE = false;
                static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
