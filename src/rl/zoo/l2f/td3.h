#include "environment.h"


#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/utils/generic/typing.h>

namespace rl_tools::rl::zoo::l2f::td3{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct TD3_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI TRAINING_INTERVAL = 10;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.9;
                static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.3;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
            };
            static constexpr T EXPLORATION_NOISE = 0.3;
            static constexpr TI STEP_LIMIT = 3000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::FAST_TANH;
            static constexpr TI EPISODE_STEP_LIMIT = 500;
//            static constexpr bool SHARED_BATCH = false;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
    };
}
