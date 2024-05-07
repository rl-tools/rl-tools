#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include "../../../rl/components/replay_buffer/replay_buffer.h"


/* requirements
- Multiple environments
- Batched action inference


*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::off_policy_runner {
    template <typename TI, TI T_NUM_THREADS>
    struct ExecutionHints{
        static constexpr TI NUM_THREADS = T_NUM_THREADS;
    };
    template <typename T_T, typename T_TI>
    struct ParametersDefault{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr bool ASYMMETRIC_OBSERVATIONS = false;
        static constexpr TI REPLAY_BUFFER_CAPACITY = 10000;
        static constexpr TI EPISODE_STEP_LIMIT = 1000;
        static constexpr bool STOCHASTIC_POLICY = false;
        static constexpr bool COLLECT_EPISODE_STATS = false;
        static constexpr TI EPISODE_STATS_BUFFER_SIZE = 0;

        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template <typename SPEC>
    struct ParametersRuntime{
        using T = typename SPEC::T;
        T exploration_noise = SPEC::PARAMETERS::EXPLORATION_NOISE;
    };
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, typename T_PARAMETERS, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT =  T_ENVIRONMENT;
        using PARAMETERS = T_PARAMETERS;
        static_assert((PARAMETERS::ASYMMETRIC_OBSERVATIONS && ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED > 0) == PARAMETERS::ASYMMETRIC_OBSERVATIONS, "ASYMMETRIC_OBSERVATIONS requested but not available in the environment");
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = PARAMETERS::ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED_ACTUAL = PARAMETERS::ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : 0;
        static constexpr bool ACTION_CLAMPING_TANH = PARAMETERS::STOCHASTIC_POLICY;

        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

    template<typename SPEC>
    struct Buffers{
        // todo: make the buffer exploit the observation = observation_priviliged to save memory in the case of symmetric observations
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::OBSERVATION_DIM_PRIVILEGED>> observations_privileged;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM * (SPEC::PARAMETERS::STOCHASTIC_POLICY ? 2 : 1)>> actions;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> next_observations;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::OBSERVATION_DIM_PRIVILEGED>> next_observations_privileged;
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
        static constexpr bool ASYMMETRIC_OBSERVATIONS = SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = SPEC::OBSERVATION_DIM_PRIVILEGED;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        static constexpr TI DATA_DIM = OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL + ACTION_DIM + OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BATCH_SIZE, DATA_DIM>> observations_actions_next_observations;

        template<typename SPEC::TI DIM>
        using OANO_VIEW = typename decltype(observations_actions_next_observations)::template VIEW<BATCH_SIZE, DIM>;

        OANO_VIEW<OBSERVATION_DIM> observations;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> observations_privileged;
        OANO_VIEW<ACTION_DIM> actions;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED + ACTION_DIM> observations_and_actions;
        OANO_VIEW<OBSERVATION_DIM> next_observations;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> next_observations_privileged;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, BATCH_SIZE>> rewards;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, BATCH_SIZE>> terminated;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, BATCH_SIZE>> truncated;
    };

    template<typename SPEC>
    struct EpisodeStats{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BUFFER_SIZE = SPEC::PARAMETERS::EPISODE_STATS_BUFFER_SIZE;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, BUFFER_SIZE, 2>> data;

        TI next_episode_i = 0;
        template<typename SPEC::TI DIM>
        using STATS_VIEW = typename decltype(data)::template VIEW<BUFFER_SIZE, DIM>;
        STATS_VIEW<1> returns;
        STATS_VIEW<1> steps;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components{
    template<typename T_SPEC>
    struct OffPolicyRunner {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using REPLAY_BUFFER_SPEC = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED, SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::PARAMETERS::REPLAY_BUFFER_CAPACITY, typename SPEC::CONTAINER_TYPE_TAG>;
        using REPLAY_BUFFER_WITH_STATES_SPEC = replay_buffer::SpecificationWithStates<ENVIRONMENT, REPLAY_BUFFER_SPEC>;
        using REPLAY_BUFFER_TYPE = ReplayBufferWithStates<REPLAY_BUFFER_WITH_STATES_SPEC>;
        static constexpr TI N_ENVIRONMENTS = SPEC::PARAMETERS::N_ENVIRONMENTS;
//        using POLICY_EVAL_BUFFERS = typename POLICY::template Buffers<N_ENVIRONMENTS>;

        off_policy_runner::ParametersRuntime<SPEC> parameters;
        template<typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        using Batch = off_policy_runner::Batch<typename off_policy_runner::BatchSpecification<SPEC, T_BATCH_SIZE, T_CONTAINER_TYPE_TAG>>;

        off_policy_runner::Buffers<SPEC> buffers;

        // todo: change to "environments"
        ENVIRONMENT envs[N_ENVIRONMENTS];
        off_policy_runner::EpisodeStats<SPEC> episode_stats[N_ENVIRONMENTS];
        REPLAY_BUFFER_TYPE replay_buffers[N_ENVIRONMENTS];

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, N_ENVIRONMENTS>> states;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, N_ENVIRONMENTS>> episode_return;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<TI, TI, 1, N_ENVIRONMENTS>> episode_step;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<bool, TI, 1, N_ENVIRONMENTS>> truncated; // init to true !!
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
