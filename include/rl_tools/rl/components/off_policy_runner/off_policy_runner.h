#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include "../../../rl/components/replay_buffer/replay_buffer.h"
#include "../../../utils/generic/tuple/tuple.h"


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
        static constexpr bool COLLECT_EPISODE_STATS = false;
        static constexpr TI EPISODE_STATS_BUFFER_SIZE = 0;
        static constexpr bool SAMPLE_PARAMETERS = true;

        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template <typename SPEC>
    struct ParametersRuntime{
        using T = typename SPEC::T;
        T exploration_noise = SPEC::PARAMETERS::EXPLORATION_NOISE;
    };
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, typename T_POLICIES, typename T_PARAMETERS, bool T_DYNAMIC_ALLOCATION=true>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT =  T_ENVIRONMENT;
        using POLICIES = T_POLICIES;
        using PARAMETERS = T_PARAMETERS;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        static_assert((PARAMETERS::ASYMMETRIC_OBSERVATIONS && ENVIRONMENT::ObservationPrivileged::DIM > 0) == PARAMETERS::ASYMMETRIC_OBSERVATIONS, "ASYMMETRIC_OBSERVATIONS requested but not available in the environment");
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = PARAMETERS::ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::ObservationPrivileged::DIM : ENVIRONMENT::Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED_ACTUAL = PARAMETERS::ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::ObservationPrivileged::DIM : 0;
    };

    template<typename SPEC, bool DYNAMIC_ALLOCATION = SPEC::DYNAMIC_ALLOCATION>
    struct Buffers{
        // todo: make the buffer exploit the observation = observation_priviliged to save memory in the case of symmetric observations
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        Matrix<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::Observation::DIM, DYNAMIC_ALLOCATION>> observations;
        using OBSERVATIONS_PRIVILEGED_STANDALONE = Matrix<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::OBSERVATION_DIM_PRIVILEGED, DYNAMIC_ALLOCATION>>;
        using OBSERVATIONS_PRIVILEGED_VIEW = typename decltype(observations)::template VIEW<>;
        using OBSERVATIONS_PRIVILEGED_TYPE = rl_tools::utils::typing::conditional_t<SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS, OBSERVATIONS_PRIVILEGED_STANDALONE, OBSERVATIONS_PRIVILEGED_VIEW>;
        OBSERVATIONS_PRIVILEGED_TYPE observations_privileged;
        Matrix<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM, DYNAMIC_ALLOCATION>> actions;
        Matrix<matrix::Specification<T, TI, SPEC::PARAMETERS::N_ENVIRONMENTS, SPEC::ENVIRONMENT::Observation::DIM, DYNAMIC_ALLOCATION>> next_observations;
        OBSERVATIONS_PRIVILEGED_TYPE next_observations_privileged;
    };



    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, bool T_DYNAMIC_ALLOCATION=true>
    struct BatchSpecification {
        using SPEC = T_SPEC;
        static constexpr typename SPEC::TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <typename T_SPEC>
    struct Batch{
        using SPEC = typename T_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        static constexpr TI BATCH_SIZE = T_SPEC::BATCH_SIZE;
        static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::Observation::DIM;
        static constexpr bool ASYMMETRIC_OBSERVATIONS = SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = SPEC::OBSERVATION_DIM_PRIVILEGED;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        static constexpr TI DATA_DIM = OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL + ACTION_DIM + OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, DATA_DIM, T_SPEC::DYNAMIC_ALLOCATION>> observations_actions_next_observations;

        template<typename SPEC::TI DIM>
        using OANO_VIEW = typename decltype(observations_actions_next_observations)::template VIEW<BATCH_SIZE, DIM>;

        OANO_VIEW<OBSERVATION_DIM> observations;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> observations_privileged;
        OANO_VIEW<ACTION_DIM> actions;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED + ACTION_DIM> observations_and_actions;
        OANO_VIEW<OBSERVATION_DIM> next_observations;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> next_observations_privileged;

        Matrix<matrix::Specification<T, TI, 1, BATCH_SIZE, T_SPEC::DYNAMIC_ALLOCATION>> rewards;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE, T_SPEC::DYNAMIC_ALLOCATION>> terminated;
        Matrix<matrix::Specification<bool, TI, 1, BATCH_SIZE, T_SPEC::DYNAMIC_ALLOCATION>> truncated;
    };

    template<typename T_SPEC, typename T_SPEC::TI T_SEQUENCE_LENGTH, typename T_SPEC::TI T_BATCH_SIZE, bool T_DYNAMIC_ALLOCATION=true>
    struct SequentialBatchSpecification {
        using SPEC = T_SPEC;
        static constexpr typename SPEC::TI SEQUENCE_LENGTH = T_SEQUENCE_LENGTH;
        static constexpr typename SPEC::TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;

    };

    template <typename T_SPEC>
    struct SequentialBatch{
        using SPEC = typename T_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        static constexpr TI SEQUENCE_LENGTH = T_SPEC::SEQUENCE_LENGTH;
        static constexpr TI BATCH_SIZE = T_SPEC::BATCH_SIZE;
        static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::Observation::DIM;
        static constexpr bool ASYMMETRIC_OBSERVATIONS = SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = SPEC::OBSERVATION_DIM_PRIVILEGED;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static constexpr bool DYNAMIC_ALLOCATION = T_SPEC::DYNAMIC_ALLOCATION;

        static constexpr TI DATA_DIM = OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL + ACTION_DIM + OBSERVATION_DIM + ACTION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, DATA_DIM>, DYNAMIC_ALLOCATION>> observations_actions_next_observations;

        template<typename SPEC::TI DIM>
        using OANO_VIEW = typename decltype(observations_actions_next_observations)::template VIEW_RANGE<tensor::ViewSpec<2, DIM>>;

        OANO_VIEW<OBSERVATION_DIM> observations;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> observations_privileged;
        OANO_VIEW<ACTION_DIM> actions;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED + ACTION_DIM> observations_and_actions;
        OANO_VIEW<OBSERVATION_DIM> next_observations;
        OANO_VIEW<ACTION_DIM> next_actions;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED + ACTION_DIM> next_observations_and_actions;
        OANO_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> next_observations_privileged;

        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>, DYNAMIC_ALLOCATION>> rewards;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>, DYNAMIC_ALLOCATION>> terminated;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>, DYNAMIC_ALLOCATION>> truncated;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>, DYNAMIC_ALLOCATION>> reset;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>, DYNAMIC_ALLOCATION>> final_step_mask;
    };

    template <typename T_OPR_SPEC, bool T_DYNAMIC_ALLOCATION=true>
    struct EpisodeStatsSpecification{
        using OPR_SPEC = T_OPR_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template<typename T_SPEC>
    struct EpisodeStats{
        using SPEC = typename T_SPEC::OPR_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BUFFER_SIZE = SPEC::PARAMETERS::EPISODE_STATS_BUFFER_SIZE;
        Matrix<matrix::Specification<T, TI, BUFFER_SIZE, 2, T_SPEC::DYNAMIC_ALLOCATION>> data;

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
    struct OffPolicyRunner{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using POLICIES = typename SPEC::POLICIES;
        template <typename INPUT>
        struct GET_STATE{
            using CONTENT = typename INPUT::template State<SPEC::DYNAMIC_ALLOCATION>;
        };
        using POLICY_STATES = rl_tools::utils::MapTuple<POLICIES, GET_STATE>;
        using REPLAY_BUFFER_SPEC = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::Observation::DIM, ENVIRONMENT::ObservationPrivileged::DIM, SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::PARAMETERS::REPLAY_BUFFER_CAPACITY, SPEC::DYNAMIC_ALLOCATION>;
        using REPLAY_BUFFER_WITH_STATES_SPEC = replay_buffer::SpecificationWithStates<ENVIRONMENT, REPLAY_BUFFER_SPEC>;
        using REPLAY_BUFFER_TYPE = ReplayBufferWithStates<REPLAY_BUFFER_WITH_STATES_SPEC>;
        static constexpr TI N_ENVIRONMENTS = SPEC::PARAMETERS::N_ENVIRONMENTS;
//        using POLICY_EVAL_BUFFERS = typename POLICY::template Buffers<N_ENVIRONMENTS>;

        off_policy_runner::ParametersRuntime<SPEC> parameters;
        template<typename T_SPEC::TI T_SEQUENCE_LENGTH, typename T_SPEC::TI T_BATCH_SIZE, bool T_DYNAMIC_ALLOCATION=true>
        using SequentialBatch = off_policy_runner::SequentialBatch<typename off_policy_runner::SequentialBatchSpecification<SPEC, T_SEQUENCE_LENGTH, T_BATCH_SIZE, T_DYNAMIC_ALLOCATION>>;

        off_policy_runner::Buffers<SPEC> buffers;

        TI previous_policy = 0;
        bool previous_policy_set = false;

        // todo: change to "environments"
        Matrix<matrix::Specification<ENVIRONMENT, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> envs;
        POLICY_STATES policy_states;
        Matrix<matrix::Specification<off_policy_runner::EpisodeStats<off_policy_runner::EpisodeStatsSpecification<SPEC, SPEC::DYNAMIC_ALLOCATION>>, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> episode_stats;
        Matrix<matrix::Specification<REPLAY_BUFFER_TYPE, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> replay_buffers = {};

        Matrix<matrix::Specification<typename SPEC::ENVIRONMENT::State, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> states;
        Matrix<matrix::Specification<typename SPEC::ENVIRONMENT::Parameters, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> env_parameters;
        Matrix<matrix::Specification<T, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> episode_return;
        Matrix<matrix::Specification<TI, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> episode_step;
        Matrix<matrix::Specification<bool, TI, 1, N_ENVIRONMENTS, SPEC::DYNAMIC_ALLOCATION>> truncated; // init to true !!
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
