#include "environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>

#include <rl_tools/utils/generic/typing.h>

namespace rl_tools::rl::zoo::l2f::sac{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI>{
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI TRAINING_INTERVAL = 16;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
                static constexpr T TARGET_ENTROPY = -((T)4);
                static constexpr TI SEQUENCE_LENGTH = 1;
                static constexpr bool ENTROPY_BONUS_NEXT_STEP = true;
            };
            static constexpr TI STEP_LIMIT = 20000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 256;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 256;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI EPISODE_STEP_LIMIT = 500;
        //            static constexpr bool SHARED_BATCH = false;
            static constexpr TI N_WARMUP_STEPS = 10000; // Exploration executed with a uniform random policy for N_WARMUP_STEPS steps
            static constexpr TI N_WARMUP_STEPS_CRITIC = 10000; // Number of steps before critic training starts
            static constexpr TI N_WARMUP_STEPS_ACTOR = 10000; // Number of steps before actor training starts
            struct OPTIMIZER_PARAMETERS_COMMON: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
                static constexpr bool ENABLE_GRADIENT_CLIPPING = false;
                static constexpr T GRADIENT_CLIP_VALUE = 1;
                static constexpr bool ENABLE_WEIGHT_DECAY = false;
                static constexpr T WEIGHT_DECAY = 0.0001;
            };
            struct ACTOR_OPTIMIZER_PARAMETERS: OPTIMIZER_PARAMETERS_COMMON{
                static constexpr T ALPHA = 3e-4;
            };
            struct CRITIC_OPTIMIZER_PARAMETERS: OPTIMIZER_PARAMETERS_COMMON{
                static constexpr T ALPHA = 3e-4;
            };
            struct ALPHA_OPTIMIZER_PARAMETERS: OPTIMIZER_PARAMETERS_COMMON{
                static constexpr T ALPHA = 3e-4;
            };
            static constexpr bool SAMPLE_ENVIRONMENT_PARAMETERS = true;
        };

        using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP>;
    };
}
