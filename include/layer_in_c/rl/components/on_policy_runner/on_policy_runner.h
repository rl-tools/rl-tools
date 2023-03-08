#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H

namespace layer_in_c::rl::components{
    namespace on_policy_runner{
        template <typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_ENVIRONMENTS = 1, T_TI T_STEP_LIMIT = 0>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
            static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
        };

        template <typename T_SPEC, typename T_SPEC::TI T_STEPS_PER_ENV>
        struct BufferSpecification{
            using SPEC = T_SPEC;
            static constexpr typename SPEC::TI STEPS_PER_ENV = T_STEPS_PER_ENV;
        };

        template <typename T_SPEC>
        struct Buffer{
            using SPEC = typename T_SPEC::SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI STEPS_PER_ENV = T_SPEC::STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * SPEC::N_ENVIRONMENTS;
            // structure: OBSERVATION - ACTION - ACTION_LOG_P - REWARD - TERMINATED - TRUNCATED - VALUE - ADVANTAGE - TARGEt_VALUE
            static constexpr TI DATA_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM + SPEC::ENVIRONMENT::ACTION_DIM + 7;

            // mem
            Matrix<matrix::Specification<T, TI, STEPS_TOTAL + SPEC::N_ENVIRONMENTS, DATA_DIM>> data; // +1 * SPEC::N_ENVIRONMENTS for the final observation

            // views
            template<TI VIEW_DIM, bool ALL = false>
            using DATA_VIEW = typename decltype(data)::template VIEW<STEPS_TOTAL + (ALL ? SPEC::N_ENVIRONMENTS : 0), VIEW_DIM>;

            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM, true> all_observations;
            DATA_VIEW<SPEC::ENVIRONMENT::OBSERVATION_DIM> observations;
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
    }

    template <typename T_SPEC>
    struct OnPolicyRunner{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        TI step = 0;

        Matrix<matrix::Specification<typename SPEC::ENVIRONMENT       , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> environments;
        Matrix<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> states;
        Matrix<matrix::Specification<bool                             , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> truncated;
        Matrix<matrix::Specification<TI                               , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> episode_step;
        Matrix<matrix::Specification<T                                , TI, 1, SPEC::N_ENVIRONMENTS, matrix::layouts::RowMajorAlignment<TI, 1>>> episode_return;
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}
#endif