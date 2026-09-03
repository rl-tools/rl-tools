#include "environment_big_generic.h"

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/curriculum/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f::ppo{
    namespace rlt = rl_tools;
    struct CurriculumTag{};
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        struct OPTIONS{
            static constexpr bool SEQUENTIAL_MODEL = false;
            static constexpr bool MOTOR_DELAY = true;
            static constexpr bool RANDOMIZE_MOTOR_MAPPING = false;
            static constexpr bool RANDOMIZE_THRUST_CURVES = false;
            static constexpr bool OBSERVE_THRASH_MARKOV = false;
        };
        using ENVIRONMENT = typename ENVIRONMENT_BIG_FACTORY<DEVICE, TYPE_POLICY, TI, OPTIONS>::ENVIRONMENT;

        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI STEP_LIMIT = 15000;

            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI BATCH_SIZE = 2048;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
            static constexpr bool NORMALIZE_OBSERVATIONS = false; // disabled for CUDA (AccumulateMode not yet implemented for CUDA standardize layer)
            static constexpr bool NORMALIZE_OBSERVATIONS_CONTINUOUSLY = false;

            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                using T = typename TYPE_POLICY::DEFAULT;
                static constexpr TI N_EPOCHS = 2;
                static constexpr bool LEARN_ACTION_STD = true;
                static constexpr T INITIAL_ACTION_STD = 0.5;
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
                static constexpr bool NORMALIZE_ADVANTAGE = true;
                static constexpr T GAMMA = 0.99;
                static constexpr bool ADAPTIVE_LEARNING_RATE = false;
                static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
                static constexpr bool SHUFFLE_EPOCH = false; // CUDA: shuffling not yet supported (uses host-side swap on GPU data)
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
    };
}

// Curriculum function overload for L2F PPO
namespace rl_tools{
    template <typename DEVICE, typename CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void curriculum(DEVICE& device, rl::loop::steps::curriculum::State<CONFIG>& ts, rl::zoo::l2f::ppo::CurriculumTag){
        using T = typename CONFIG::TYPE_POLICY::DEFAULT;
        using TI = typename CONFIG::TI;
        constexpr TI EVAL_INTERVAL = CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL;
        constexpr TI EPISODE_STEP_LIMIT = CONFIG::CORE_PARAMETERS::EPISODE_STEP_LIMIT;
        constexpr T FULL_MAX_ANGLE = 1.5707963267948966;
        constexpr T FULL_MAX_POSITION = 2.199102;
        constexpr T MIN_START_ANGLE = 0.3;
        constexpr T MIN_START_POSITION = 0.1;
        TI eval_idx = ts.step / EVAL_INTERVAL;
        if(eval_idx > 0){
            eval_idx--;
        }
        if(eval_idx >= CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS){
            eval_idx = CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS - 1;
        }
        auto& result = get(ts.evaluation_results, 0, eval_idx);
        T survival_share = result.episode_length_mean / (T)EPISODE_STEP_LIMIT;
        if(survival_share > (T)0.95){
            ts.curriculum_level = math::min(device.math, (T)1, ts.curriculum_level + (T)0.05);
        }
        T current_max_angle = MIN_START_ANGLE + ts.curriculum_level * (FULL_MAX_ANGLE - MIN_START_ANGLE);
        T current_max_position = MIN_START_POSITION + ts.curriculum_level * (FULL_MAX_POSITION - MIN_START_POSITION);
        for(TI env_i = 0; env_i < CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
            auto& env = get_ref(device, ts.environment.environments, env_i);
            env.parameters.mdp.init.max_angle = current_max_angle;
            env.parameters.mdp.init.max_position = current_max_position;
        }
        ts.env_eval.parameters.mdp.init.max_angle = current_max_angle;
        ts.env_eval.parameters.mdp.init.max_position = current_max_position;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
