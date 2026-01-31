#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::flag::ppo_gru_asymmetric{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr TI MAX_EPISODE_LENGTH = 50;
        static constexpr bool ACTOR_PRIVILEGED_OBSERVATION = false;
        static constexpr bool CRITIC_PRIVILEGED_OBSERVATION = true;
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, MAX_EPISODE_LENGTH, ACTOR_PRIVILEGED_OBSERVATION, CRITIC_PRIVILEGED_OBSERVATION>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI N_ENVIRONMENTS = 128;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI BATCH_SIZE = N_ENVIRONMENTS*ON_POLICY_RUNNER_STEPS_PER_ENV;
            static constexpr TI TOTAL_STEP_LIMIT = 10 * ((TI)100 * 1000 * 1000);
            static constexpr TI ACTOR_HIDDEN_DIM = 32;
            static constexpr TI ACTOR_NUM_LAYERS = 2;
            static constexpr TI CRITIC_HIDDEN_DIM = 32;
            static constexpr TI CRITIC_NUM_LAYERS = 2;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            struct ACTOR_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
                static constexpr T ALPHA = 0.001;
            };
            struct CRITIC_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
                static constexpr T ALPHA = 0.001;
            };
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.001;
                static constexpr TI N_EPOCHS = 1;
                static constexpr T GAMMA = 0.98;
                static constexpr T LAMBDA = 0.95;
                static constexpr T INITIAL_ACTION_STD = 0.2;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr bool SHUFFLE_EPOCH = false;
                static constexpr bool STATEFUL_ACTOR_AND_CRITIC = true;
                static constexpr bool TRUNCATE_ON_EACH_ITERATION = true;
            };
        };
        static constexpr bool CRITIC_GRU = false;
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsGRU<CRITIC_GRU>::template Approximators, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
