#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_INFERENCE_EXECUTOR_EXECUTOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_INFERENCE_EXECUTOR_EXECUTOR_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace inference{
        namespace executor{
            template <typename T_T, typename T_TI, typename T_TIMESTAMP, typename T_POLICY, T_TIMESTAMP T_CONTROL_INTERVAL_INFERENCE_NS, T_TIMESTAMP T_CONTROL_INTERVAL_TRAINING_NS, bool T_DYNAMIC_ALLOCATION=true>
            struct Specification{
                using T = T_T;
                using TI = T_TI;
                using TIMESTAMP = T_TIMESTAMP;
                static_assert(sizeof(TIMESTAMP) >= 8, "The TIMESTAMP should be unsigned and at least 8 bytes to prevent overflow (it shall be measured in nanoseconds)");
                using POLICY = T_POLICY;
                static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
                static constexpr TI TIMING_STATS_NUM_STEPS = 100;
                static constexpr TIMESTAMP CONTROL_INTERVAL_INFERENCE_NS = T_CONTROL_INTERVAL_INFERENCE_NS;
                static constexpr TIMESTAMP CONTROL_INTERVAL_TRAINING_NS = T_CONTROL_INTERVAL_TRAINING_NS;
                static constexpr T TIMING_JITTER_HIGH_THRESHOLD_NS = 1.2;
                static constexpr T TIMING_JITTER_LOW_THRESHOLD_NS = 0.8;
                static constexpr T TIMING_BIAS_HIGH_THRESHOLD = 1.2;
                static constexpr T TIMING_BIAS_LOW_THRESHOLD = 0.8;
            };
            template <typename T_SPEC>
            struct JitterStatus{
                using SPEC = T_SPEC;
                using T = typename SPEC::T;
                bool OK;
                T MAGNITUDE;

            };
            template <typename T_SPEC>
            struct BiasStatus{
                using SPEC = T_SPEC;
                using T = typename SPEC::T;
                bool OK;
                T MAGNITUDE;
            };
            template <typename T_SPEC>
            struct Status{
                using SPEC = T_SPEC;
                using T = typename SPEC::T;
                bool OK;
                bool TIMESTAMP_INVALID;
                bool LAST_CONTROL_TIMESTAMP_GREATER_THAN_LAST_OBSERVATION_TIMESTAMP;
                enum Source{
                    OBSERVATION,
                    CONTROL
                };
                Source source;
                enum StepType{
                    INFERENCE,
                    ORIGINAL
                };
                StepType step_type;
                JitterStatus<SPEC> jitter;
                BiasStatus<SPEC> bias;
            };
        }
        template <typename T_SPEC>
        struct Executor{
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using POLICY = typename SPEC::POLICY;
            static constexpr bool DYNAMIC_ALLOCATION = SPEC::DYNAMIC_ALLOCATION;
            using TIMESTAMP = typename SPEC::TIMESTAMP;

            TIMESTAMP last_observation_timestamp, last_control_timestamp, last_control_timestamp_original; // last_control_timestamp runs at the higher rate, while last_control_timestamp_original runs at the original control rate of the simulation
            bool last_observation_timestamp_set, last_control_timestamp_set, last_control_timestamp_original_set;
            TIMESTAMP control_dt[SPEC::TIMING_STATS_NUM_STEPS];
            TIMESTAMP control_dt_index = 0;
            TIMESTAMP control_original_dt[SPEC::TIMING_STATS_NUM_STEPS];
            TIMESTAMP control_original_dt_index = 0;

            static constexpr TI INPUT_DIM = SPEC::POLICY::INPUT_SHAPE::LAST;
            Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, tensor::Shape<typename SPEC::TI, 1, INPUT_DIM>, DYNAMIC_ALLOCATION>> observation;
            typename POLICY::template State<DYNAMIC_ALLOCATION> policy_state, policy_state_temp;
            typename POLICY::template Buffer<DYNAMIC_ALLOCATION> policy_buffer;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
