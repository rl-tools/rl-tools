#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components{
    namespace on_policy_runner{
        template <typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_ENVIRONMENTS = 1, T_TI T_STEP_LIMIT = 0, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
            static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        };

        template <typename T_SPEC, typename T_SPEC::TI T_STEPS_PER_ENV, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        struct DatasetSpecification{
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
            static constexpr TI STEPS_PER_ENV = T_STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * SPEC::N_ENVIRONMENTS;
            static constexpr TI STEPS_TOTAL_ALL = (STEPS_PER_ENV+1) * SPEC::N_ENVIRONMENTS; // +1 for the final observation
        };

        template <typename T_SPEC>
        struct Dataset{
            using SPEC = typename T_SPEC::SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI STEPS_PER_ENV = T_SPEC::STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = T_SPEC::STEPS_TOTAL;
            // structure: OBSERVATION - ACTION - ACTION_LOG_P - REWARD - TERMINATED - TRUNCATED - VALUE - ADVANTAGE - TARGEt_VALUE
            static constexpr TI DATA_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM * 2 + SPEC::ENVIRONMENT::ACTION_DIM * 2 + 7;

            // mem
            // todo: evaluate transposing this / storing in column major order for better memory access in the single dimensional columns
            typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, STEPS_TOTAL + SPEC::N_ENVIRONMENTS, DATA_DIM>> data; // +1 * SPEC::N_ENVIRONMENTS for the final observation

            // views
            template<TI VIEW_DIM, bool ALL = false>
            using DATA_VIEW = typename decltype(data)::template VIEW<STEPS_TOTAL + (ALL ? SPEC::N_ENVIRONMENTS : 0), VIEW_DIM>;

            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM, true> all_observations;
            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM> observations;
            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM, true> all_observations_normalized;
            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM> observations_normalized;
            DATA_VIEW<SPEC::ENVIRONMENT::ACTION_DIM> actions_mean;
            DATA_VIEW<SPEC::ENVIRONMENT::ACTION_DIM> actions;
            DATA_VIEW<1> action_log_probs;
            DATA_VIEW<1> rewards;
            DATA_VIEW<1> terminated;
            DATA_VIEW<1> truncated;
            DATA_VIEW<1, true> all_values;
            DATA_VIEW<1> values;
            DATA_VIEW<1> advantages;
            DATA_VIEW<1> target_values;
        };
        template <typename TI, TI T_NUM_THREADS>
        struct ExecutionHints{
            static constexpr TI NUM_THREADS = T_NUM_THREADS;
        };
    }

    template <typename T_SPEC>
    struct OnPolicyRunner{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        TI step = 0;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename SPEC::ENVIRONMENT       , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> environments;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> states;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool                             , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> truncated;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<TI                               , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> episode_step;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T                                , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> episode_return;
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
