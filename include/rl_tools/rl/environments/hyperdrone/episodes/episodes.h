#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_EPISODES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_EPISODES_H

#include "../../../../containers/tensor/tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::episodes {
    // Per-instance episode bookkeeping over the mask verbs. Same-step autoreset: begin_step applies
    // the reset before the step's observe/render, so the stored observation is the first of the new
    // episode and no final observation exists. Flags follow the rl_tools dataset convention:
    // terminated implies truncated, and reset is truncated delayed by one step (the mask that
    // sample_initial_*, render, task caches and recurrent policies consume).
    template <typename T_TI>
    struct EndReason {
        static constexpr T_TI NONE = 0;
        static constexpr T_TI TERMINATED = 1;
        static constexpr T_TI TIME_LIMIT = 2;
        static constexpr T_TI FORCED = 3;
    };

    template <typename T_ENVIRONMENT>
    struct Specification {
        using ENVIRONMENT = T_ENVIRONMENT;  // World, task World, or MultiEnvironment
        using T = typename ENVIRONMENT::T;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
        // derive-and-shadow
        static constexpr TI STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;  // 0: no time limit
        static constexpr bool SYNCHRONIZED = false;                         // all-or-none resets (fixed-length tasks)
    };

    template <typename T_SPEC>
    struct Episodes {
        using SPEC = T_SPEC;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        using FLAG_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, INSTANCES>>;
        using COUNTER_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, INSTANCES>>;
        using VALUE_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, INSTANCES>>;
        Tensor<COUNTER_SPEC> episode_step;
        Tensor<FLAG_SPEC> terminated;   // terminal state after the last step
        Tensor<FLAG_SPEC> truncated;    // episode ended after the last step: terminated, time limit or forced
        Tensor<FLAG_SPEC> reset;        // mask applied by the last begin_step
        Tensor<FLAG_SPEC> forced;       // pending external reset (scene rotation, epoch boundary)
        Tensor<VALUE_SPEC> episode_return;
        Tensor<COUNTER_SPEC> end_reason;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 1>>> synchronized_due;  // SYNCHRONIZED: any instance due
        TI step_limit = SPEC::STEP_LIMIT;  // runtime override (curricula)
    };

    // per-rollout log written by begin_step(step_i): one row per step, one column per instance
    template <typename T_SPEC, typename T_SPEC::TI T_STEPS>
    struct Log {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI STEPS = T_STEPS;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        using VALUE_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, STEPS, INSTANCES>>;
        using COUNTER_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, STEPS, INSTANCES>>;
        Tensor<VALUE_SPEC> finished_length;   // -1 where no episode ended at that step
        Tensor<VALUE_SPEC> finished_return;
        Tensor<COUNTER_SPEC> finished_reason;
    };

    template <typename T_T, typename T_TI>
    struct Statistics {
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
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
