#include <rl_tools/rl/algorithms/sac/loop/core/config.h>

#include <rl_tools/utils/generic/typing.h>

namespace builder{
    template <typename DEVICE, typename T, typename TI, typename RNG, typename OPTIONS, bool DYNAMIC_ALLOCATION=true>
    struct FACTORY{
        using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY<DEVICE, T, TI, OPTIONS>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct SAC_PARAMETERS: rl::algorithms::sac::DefaultParameters<T, TI>{
                static constexpr TI ACTOR_BATCH_SIZE = 128;
                static constexpr TI CRITIC_BATCH_SIZE = 128;
                static constexpr TI TRAINING_INTERVAL = 1;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
                // static constexpr T ALPHA = 0.05;
                static constexpr T TARGET_ENTROPY = -((T)2);
                static constexpr TI SEQUENCE_LENGTH = 1;
                static constexpr bool ENTROPY_BONUS_NEXT_STEP = false;
            };
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI STEP_LIMIT = 1000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 256;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI EPISODE_STEP_LIMIT = 500;
        //            static constexpr bool SHARED_BATCH = false;
            static constexpr TI N_WARMUP_STEPS = 0; // Exploration executed with a uniform random policy for N_WARMUP_STEPS steps
            static constexpr TI N_WARMUP_STEPS_CRITIC = 10000; // Number of steps before critic training starts
            static constexpr TI N_WARMUP_STEPS_ACTOR = 10000; // Number of steps before actor training starts
            struct OPTIMIZER_PARAMETERS_COMMON: nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
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
                static constexpr T ALPHA = 1e-4;
            };
            static constexpr bool SAMPLE_ENVIRONMENT_PARAMETERS = true;
            struct BATCH_SAMPLING_PARAMETERS{
                static constexpr bool INCLUDE_FIRST_STEP_IN_TARGETS = false;
                static constexpr bool ALWAYS_SAMPLE_FROM_INITIAL_STATE = false;
                static constexpr bool RANDOM_SEQ_LENGTH = false;
                static constexpr bool ENABLE_NOMINAL_SEQUENCE_LENGTH_PROBABILITY = true;
                static constexpr T NOMINAL_SEQUENCE_LENGTH_PROBABILITY = 0.1;
            };
        };
        // this config is competitive with mlp but 15x slower

        using LOOP_CORE_CONFIG = rl_tools::utils::typing::conditional_t<OPTIONS::SEQUENTIAL_MODEL,
            rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rl::algorithms::sac::loop::core::ConfigApproximatorsGRU, DYNAMIC_ALLOCATION>,
            rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rl::algorithms::sac::loop::core::ConfigApproximatorsMLP, DYNAMIC_ALLOCATION>
        >;
        static constexpr TI RNG_PARAMS_WARMUP_STEPS = 100;
    };
}

namespace builder{


    template <typename T_CORE_CONFIG>
    struct LOOP_ASSEMBLY{
        using T = typename T_CORE_CONFIG::T;
        using TI = typename T_CORE_CONFIG::TI;
        static constexpr TI NUM_CHECKPOINTS = 10;
        static constexpr TI NUM_EVALUATIONS = 100;
        static constexpr TI NUM_SAVE_TRAJECTORIES = 10;
        static constexpr TI TIMING_INTERVAL = 10000;

        using LOOP_TIMING_CONFIG = rl::loop::steps::timing::Config<T_CORE_CONFIG, rl::loop::steps::timing::Parameters<TI, TIMING_INTERVAL>>;
        using LOOP_EXTRACK_CONFIG = rl::loop::steps::extrack::Config<LOOP_TIMING_CONFIG>;
        struct LOOP_CHECKPOINT_PARAMETERS: rl::loop::steps::checkpoint::Parameters<T, TI>{
            static constexpr TI CHECKPOINT_INTERVAL_TEMP = T_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_CHECKPOINTS;
            static constexpr TI CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_TEMP == 0 ? 1 : CHECKPOINT_INTERVAL_TEMP;
        };
        using LOOP_CHECKPOINT_CONFIG = rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
        static constexpr TI EVALUATION_INTERVAL_TEMP = T_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
        static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
        static constexpr TI NUM_EVALUATION_EPISODES = 100;
        using LOOP_EVALUATION_PARAMETERS = rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG, NUM_EVALUATION_EPISODES, EVALUATION_INTERVAL>;
        using LOOP_EVALUATION_CONFIG = rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
        struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
            static constexpr TI INTERVAL_TEMP = T_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
            static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
            static constexpr TI NUM_EPISODES = 100;
        };
        using LOOP_SAVE_TRAJECTORIES_CONFIG = rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
        struct LOOP_NN_ANALYTICS_PARAMETERS: rl::loop::steps::nn_analytics::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
            static constexpr TI INTERVAL_TEMP = LOOP_SAVE_TRAJECTORIES_PARAMETERS::INTERVAL;
            static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
        };
        using LOOP_NN_ANALYTICS_CONFIG = rl::loop::steps::nn_analytics::Config<LOOP_SAVE_TRAJECTORIES_CONFIG, LOOP_NN_ANALYTICS_PARAMETERS>;
        using LOOP_CONFIG = LOOP_NN_ANALYTICS_CONFIG;
    };
}
