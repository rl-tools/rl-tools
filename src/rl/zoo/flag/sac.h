#include "environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::flag::sac{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct SAC_PARAMETERS: rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
                static constexpr TI ACTOR_BATCH_SIZE = 32;
                static constexpr TI CRITIC_BATCH_SIZE = 32;
                static constexpr T GAMMA = 0.98;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 10;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 20;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 20;
            };
            static constexpr TI STEP_LIMIT = 2000000;
            static constexpr TI N_ENVIRONMENTS = 32;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT * N_ENVIRONMENTS;
            static constexpr TI N_WARMUP_STEPS = 100;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 100;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 100;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 32;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 32;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr T ALPHA = 1.0;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
