#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>

namespace lic = layer_in_c;


/* requirements
- Multiple environments
- Batched action inference


*/

namespace layer_in_c::rl::components::off_policy_runner {
    template <typename TI, TI T_NUM_THREADS>
    struct ExecutionHints{
        static constexpr TI NUM_THREADS = T_NUM_THREADS;
    };
    template<typename T>
    struct DefaultParameters{
        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_ENVIRONMENTS, T_TI T_REPLAY_BUFFER_CAPACITY, T_TI T_STEP_LIMIT, typename T_PARAMETERS, bool T_COLLECT_EPISODE_STATS = false, T_TI T_EPISODE_STATS_BUFFER_SIZE = 0, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT =  T_ENVIRONMENT;
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        static constexpr TI REPLAY_BUFFER_CAPACITY = T_REPLAY_BUFFER_CAPACITY;
        static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
        using PARAMETERS = T_PARAMETERS;
        static constexpr bool COLLECT_EPISODE_STATS = T_COLLECT_EPISODE_STATS;
        static constexpr TI EPISODE_STATS_BUFFER_SIZE = T_EPISODE_STATS_BUFFER_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

    template<typename SPEC>
    struct Buffers{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>> actions;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> next_observations;
    };



    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
    struct BatchSpecification {
        using SPEC = T_SPEC;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr typename SPEC::TI BATCH_SIZE = T_BATCH_SIZE;
    };

    template <typename T_SPEC>
    struct Batch{
        using SPEC = typename T_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        static constexpr TI BATCH_SIZE = T_SPEC::BATCH_SIZE;
        static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, 2*OBSERVATION_DIM + ACTION_DIM>> observations_actions_next_observations;

        template<typename SPEC::TI DIM>
        using OANO_VIEW = typename decltype(observations_actions_next_observations)::template VIEW<BATCH_SIZE, DIM>;

        OANO_VIEW<OBSERVATION_DIM> observations;
        OANO_VIEW<ACTION_DIM> actions;
        OANO_VIEW<OBSERVATION_DIM + ACTION_DIM> observations_and_actions;
        OANO_VIEW<OBSERVATION_DIM> next_observations;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, BATCH_SIZE>> rewards;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, BATCH_SIZE>> terminated;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, BATCH_SIZE>> truncated;
    };

    template<typename SPEC>
    struct EpisodeStats{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BUFFER_SIZE = SPEC::EPISODE_STATS_BUFFER_SIZE;
        MatrixStatic<matrix::Specification<T, TI, BUFFER_SIZE, 2>> data;

        TI next_episode_i = 0;
        template<typename SPEC::TI DIM>
        using STATS_VIEW = typename decltype(data)::template VIEW<BUFFER_SIZE, DIM>;
        STATS_VIEW<1> returns;
        STATS_VIEW<1> steps;
    };
}

namespace layer_in_c::rl::components{
    template<typename T_SPEC>
    struct OffPolicyRunner {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using REPLAY_BUFFER_SPEC = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::REPLAY_BUFFER_CAPACITY, typename SPEC::CONTAINER_TYPE_TAG>;
        using REPLAY_BUFFER_TYPE = ReplayBuffer<REPLAY_BUFFER_SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        static constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
//        using POLICY_EVAL_BUFFERS = typename POLICY::template Buffers<N_ENVIRONMENTS>;

        off_policy_runner::Buffers<SPEC> buffers;

        ENVIRONMENT envs[N_ENVIRONMENTS];
        off_policy_runner::EpisodeStats<SPEC> episode_stats[N_ENVIRONMENTS];
        REPLAY_BUFFER_TYPE replay_buffers[N_ENVIRONMENTS];

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, N_ENVIRONMENTS>> states;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, N_ENVIRONMENTS>> episode_return;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<TI, TI, 1, N_ENVIRONMENTS>> episode_step;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, N_ENVIRONMENTS>> truncated; // init to true !!
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}

#endif
