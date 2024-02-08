#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_PPO_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_PPO_H

#include "../../../rl/components/running_normalizer/running_normalizer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms{
    namespace ppo{
        template<typename T, typename TI>
        struct DefaultParameters {
            static constexpr T GAMMA = 0.99;
            static constexpr T LAMBDA = 0.95;
            static constexpr T EPSILON_CLIP = 0.2;
            static constexpr T INITIAL_ACTION_STD = 0.5;
            static constexpr bool LEARN_ACTION_STD = true;
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
            static constexpr T ADVANTAGE_EPSILON = 1e-8;
            static constexpr bool NORMALIZE_ADVANTAGE = true;
            static constexpr bool ADAPTIVE_LEARNING_RATE = false;
            static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
            static constexpr T POLICY_KL_EPSILON = 1e-5;
            static constexpr T ADAPTIVE_LEARNING_RATE_DECAY = (T)1/(T)1.5;
            static constexpr T ADAPTIVE_LEARNING_RATE_MIN = 1e-6;
            static constexpr T ADAPTIVE_LEARNING_RATE_MAX = 1e-2;
            static constexpr bool NORMALIZE_OBSERVATIONS = false;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 0;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 0;
            static constexpr TI N_EPOCHS = 10;
            static constexpr bool IGNORE_TERMINATION = false; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
        };

        template<
                typename T_T,
                typename T_TI,
                typename T_ENVIRONMENT,
                typename T_ACTOR_TYPE,
                typename T_CRITIC_TYPE,
                typename T_PARAMETERS = DefaultParameters<T_T, T_TI>,
                typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag
        >
        struct Specification {
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            using ACTOR_TYPE = T_ACTOR_TYPE;
            using CRITIC_TYPE = T_CRITIC_TYPE;
            static constexpr TI BATCH_SIZE = ACTOR_TYPE::SPEC::BATCH_SIZE;
            using PARAMETERS = T_PARAMETERS;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;

            static_assert(ACTOR_TYPE::SPEC::BATCH_SIZE == CRITIC_TYPE::SPEC::BATCH_SIZE);
            static_assert(ACTOR_TYPE::INPUT_DIM == ENVIRONMENT::OBSERVATION_DIM);
            static_assert(CRITIC_TYPE::INPUT_DIM == ENVIRONMENT::OBSERVATION_DIM);
            static_assert(ACTOR_TYPE::OUTPUT_DIM == ENVIRONMENT::ACTION_DIM);
            static_assert(CRITIC_TYPE::OUTPUT_DIM == 1);
        };

        template <typename SPEC>
        struct Buffers{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI BATCH_SIZE = SPEC::BATCH_SIZE;
            static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
            static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> current_batch_actions;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 1>> d_critic_output;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_action_log_prob_d_action;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_action_log_prob_d_action_log_std;
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, ACTION_DIM>> rollout_log_std;
        };
    }

    template<typename T_SPEC>
    struct PPO {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        typename SPEC::ACTOR_TYPE actor;
        typename SPEC::CRITIC_TYPE critic;
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        bool initialized = false;
#endif

    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
