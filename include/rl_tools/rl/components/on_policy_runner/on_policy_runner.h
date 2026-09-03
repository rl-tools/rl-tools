#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_ON_POLICY_RUNNER_H

#include "../../../utils/generic/typing.h"
#include "../../../rl/environments/observation.h"
#include "../../../containers/tensor/tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components{
    namespace on_policy_runner{
        enum class EpisodeEndReason: unsigned char {
            NONE = 0,
            TERMINATED = 1,
            TIME_LIMIT = 2,
            FORCED = 3
        };

        template <typename T_BATCH_ENVIRONMENT, typename = void>
        struct LogicalEnvironment{
            using TYPE = T_BATCH_ENVIRONMENT;
        };
        template <typename T_BATCH_ENVIRONMENT>
        struct LogicalEnvironment<T_BATCH_ENVIRONMENT, utils::typing::void_t<typename T_BATCH_ENVIRONMENT::ENVIRONMENT>>{
            using TYPE = typename T_BATCH_ENVIRONMENT::ENVIRONMENT;
        };

        template <typename T_TYPE_POLICY, typename T_BATCH_ENVIRONMENT, typename T_POLICY_STATE, typename T_OBSERVATION = typename T_BATCH_ENVIRONMENT::Observation, typename T_OBSERVATION_PRIVILEGED = typename T_BATCH_ENVIRONMENT::ObservationPrivileged, typename T_OBSERVATION_T = typename T_TYPE_POLICY::DEFAULT, typename T_OBSERVATION_PRIVILEGED_T = typename T_TYPE_POLICY::DEFAULT, typename T_BATCH_ENVIRONMENT::TI T_STEP_LIMIT = T_BATCH_ENVIRONMENT::EPISODE_STEP_LIMIT, bool T_TRUNCATE_ON_EACH_ITERATION = false, bool T_DYNAMIC_ALLOCATION = true>
        struct Specification{
            using TYPE_POLICY = T_TYPE_POLICY;
            using T = typename TYPE_POLICY::DEFAULT;
            using BATCH_ENVIRONMENT = T_BATCH_ENVIRONMENT;
            using ENVIRONMENT = typename LogicalEnvironment<BATCH_ENVIRONMENT>::TYPE;
            using TI = typename BATCH_ENVIRONMENT::TI;
            using POLICY_STATE = T_POLICY_STATE;
            using OBSERVATION = T_OBSERVATION;
            using OBSERVATION_PRIVILEGED = T_OBSERVATION_PRIVILEGED;
            using OBSERVATION_T = T_OBSERVATION_T;
            using OBSERVATION_PRIVILEGED_T = T_OBSERVATION_PRIVILEGED_T;
            static constexpr TI N_ENVIRONMENTS = BATCH_ENVIRONMENT::INSTANCES;
            static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
            static constexpr bool ASYMMETRIC_OBSERVATIONS = !rl_tools::utils::typing::is_same_v<OBSERVATION, OBSERVATION_PRIVILEGED>;
            static constexpr TI N_AGENTS_PER_ENV = BATCH_ENVIRONMENT::N_AGENTS;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
            static constexpr bool TRUNCATE_ON_EACH_ITERATION = T_TRUNCATE_ON_EACH_ITERATION;
            static_assert(BATCH_ENVIRONMENT::ACTION_DIM == ENVIRONMENT::ACTION_DIM, "the batch and logical environments must have the same action dimension");
            static_assert(BATCH_ENVIRONMENT::N_AGENTS == ENVIRONMENT::N_AGENTS, "the batch and logical environments must have the same number of agents");
        };

        template <typename T_T, typename T_TI>
        struct EpisodeStatistics {
            using T = T_T;
            using TI = T_TI;
            TI finished = 0;
            TI terminated = 0;
            TI time_limit = 0;
            TI forced = 0;
            TI in_progress = 0;
            T length_sum = 0;
            T return_sum = 0;
            T in_progress_length_sum = 0;
            T mean_length = 0;
            T mean_return = 0;
            T terminated_share = 0;
            T mean_in_progress_length = 0;
        };

        template <typename T_SPEC, typename T_SPEC::TI T_STEPS_PER_ENV, bool T_DYNAMIC_ALLOCATION = true>
        struct DatasetSpecification{
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            static constexpr TI STEPS_PER_ENV = T_STEPS_PER_ENV;
            static constexpr bool ASYMMETRIC_OBSERVATIONS = SPEC::ASYMMETRIC_OBSERVATIONS;
            static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * SPEC::N_ENVIRONMENTS;
            static constexpr TI STEPS_TOTAL_ALL = (STEPS_PER_ENV+1) * SPEC::N_ENVIRONMENTS; // +1 for the final observation
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        };

        template <typename T_DATASET_SPEC>
        struct Dataset{
            using DATASET_SPEC = T_DATASET_SPEC;
            using SPEC = typename DATASET_SPEC::SPEC;
            using TYPE_POLICY = typename SPEC::TYPE_POLICY;
            using T = typename TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            static constexpr TI STEPS_PER_ENV = DATASET_SPEC::STEPS_PER_ENV;
            static constexpr TI STEPS_TOTAL = DATASET_SPEC::STEPS_TOTAL;

            using OBS_SHAPE = typename SPEC::OBSERVATION::SHAPE;
            using OBS_PRIV_SHAPE = typename SPEC::OBSERVATION_PRIVILEGED::SHAPE;

            // Observation tensor storage (always flat to ensure matrix_view gives (N, DIM) rows)
            using ALL_OBS_STORAGE_SHAPE = tensor::Shape<TI, DATASET_SPEC::STEPS_TOTAL_ALL, SPEC::OBSERVATION::DIM>;
            using ALL_OBS_PRIV_STORAGE_SHAPE = tensor::Shape<TI, DATASET_SPEC::STEPS_TOTAL_ALL, SPEC::OBSERVATION_PRIVILEGED::DIM>;
            // the observation storage types follow the specification (e.g. bf16 frames), the scalar data is T
            static_assert(DATASET_SPEC::ASYMMETRIC_OBSERVATIONS || rl_tools::utils::typing::is_same_v<typename SPEC::OBSERVATION_T, typename SPEC::OBSERVATION_PRIVILEGED_T>, "symmetric observations share one storage");
            Tensor<tensor::Specification<typename SPEC::OBSERVATION_T, TI, ALL_OBS_STORAGE_SHAPE, DATASET_SPEC::DYNAMIC_ALLOCATION>> all_observations;
            Tensor<tensor::Specification<typename SPEC::OBSERVATION_PRIVILEGED_T, TI, ALL_OBS_PRIV_STORAGE_SHAPE, DATASET_SPEC::DYNAMIC_ALLOCATION>> all_observations_privileged;
            Tensor<tensor::Specification<EpisodeEndReason, TI, tensor::Shape<TI, DATASET_SPEC::STEPS_PER_ENV + 1, SPEC::N_ENVIRONMENTS>, DATASET_SPEC::DYNAMIC_ALLOCATION>> episode_end_reason;
            Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, DATASET_SPEC::STEPS_PER_ENV + 1, SPEC::N_ENVIRONMENTS>, DATASET_SPEC::DYNAMIC_ALLOCATION>> episode_length;
            Tensor<tensor::Specification<typename SPEC::BATCH_ENVIRONMENT::T, TI, tensor::Shape<TI, DATASET_SPEC::STEPS_PER_ENV + 1, SPEC::N_ENVIRONMENTS>, DATASET_SPEC::DYNAMIC_ALLOCATION>> episode_return;

            // Scalar data (actions, rewards, flags, values, advantages)
            static constexpr TI SCALAR_DATA_DIM = SPEC::ENVIRONMENT::ACTION_DIM * 2 + 8;
            Matrix<matrix::Specification<T, TI, STEPS_TOTAL + SPEC::N_ENVIRONMENTS, SCALAR_DATA_DIM, DATASET_SPEC::DYNAMIC_ALLOCATION>> scalar_data;

            template<TI VIEW_DIM, bool ALL = false>
            using SCALAR_VIEW = typename decltype(scalar_data)::template VIEW<STEPS_TOTAL + (ALL ? SPEC::N_ENVIRONMENTS : 0), VIEW_DIM>;

            SCALAR_VIEW<SPEC::ENVIRONMENT::ACTION_DIM> actions_mean;
            SCALAR_VIEW<SPEC::ENVIRONMENT::ACTION_DIM> actions;
            SCALAR_VIEW<1> action_log_probs;
            SCALAR_VIEW<1> rewards;
            SCALAR_VIEW<1> terminated;
            SCALAR_VIEW<1> truncated;
            SCALAR_VIEW<1, true> all_reset;
            SCALAR_VIEW<1> reset; // = truncation delayed by one step for the reset of stateful actors and critics
            SCALAR_VIEW<1, true> all_values;
            SCALAR_VIEW<1> values;
            SCALAR_VIEW<1> advantages;
            SCALAR_VIEW<1> target_values;
        };
        template <typename T_SPEC>
        struct Buffer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using BATCH_ENVIRONMENT = typename SPEC::BATCH_ENVIRONMENT;
            Tensor<tensor::Specification<typename BATCH_ENVIRONMENT::State, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>> next_states;
            Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS, BATCH_ENVIRONMENT::ACTION_DIM>, SPEC::DYNAMIC_ALLOCATION>> actions;
            Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>> rewards;
            Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>> terminated;
        };
    }

    template <typename T_SPEC>
    struct OnPolicyRunner {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using BATCH_ENVIRONMENT = typename SPEC::BATCH_ENVIRONMENT;
        using EPISODE_T = typename BATCH_ENVIRONMENT::T;
        using FLAG_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>;
        using COUNTER_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>;
        using VALUE_SPEC = tensor::Specification<EPISODE_T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>;
        using REASON_SPEC = tensor::Specification<on_policy_runner::EpisodeEndReason, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>;

        TI step = 0;

        typename SPEC::POLICY_STATE policy_state;
        Tensor<tensor::Specification<typename BATCH_ENVIRONMENT::Parameters, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>> env_parameters;
        Tensor<tensor::Specification<typename BATCH_ENVIRONMENT::State, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>, SPEC::DYNAMIC_ALLOCATION>> states;
        Tensor<COUNTER_SPEC> episode_step;
        Tensor<FLAG_SPEC> reset;
        Tensor<VALUE_SPEC> episode_return;
        Tensor<COUNTER_SPEC> completed_episode_length;
        Tensor<VALUE_SPEC> completed_episode_return;
        Tensor<REASON_SPEC> completed_episode_reason;
        TI episode_step_limit = SPEC::STEP_LIMIT;
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        bool initialized = false;
#endif
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
