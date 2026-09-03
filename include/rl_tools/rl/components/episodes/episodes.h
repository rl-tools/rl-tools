#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_EPISODES_EPISODES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_EPISODES_EPISODES_H

#include "../../../containers/tensor/tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::episodes {
    // Per-instance episode bookkeeping over the mask verbs. Same-step autoreset: begin_step applies
    // the reset before the step's observe/render, so the stored observation is the first of the new
    // episode and no final observation exists. Flags follow the rl_tools dataset convention:
    // terminated implies truncated, and reset is truncated delayed by one step (the mask that
    // sample_initial_*, render, task caches and recurrent policies consume). init puts every
    // instance into the due state and discards in-progress episodes (epoch boundaries).
    enum class EndReason: unsigned char {
        NONE = 0,
        TERMINATED = 1,
        TIME_LIMIT = 2,
        FORCED = 3
    };

    template <typename T_ENVIRONMENT, bool T_DYNAMIC_ALLOCATION = true>
    struct Specification {
        using ENVIRONMENT = T_ENVIRONMENT;  // World, task World, or MultiEnvironment
        using T = typename ENVIRONMENT::T;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
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
        using FLAG_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>;
        using COUNTER_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>;
        using VALUE_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>;
        using REASON_SPEC = tensor::Specification<EndReason, TI, tensor::Shape<TI, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>;
        Tensor<COUNTER_SPEC> episode_step;
        Tensor<FLAG_SPEC> terminated;   // terminal state after the last step
        Tensor<FLAG_SPEC> truncated;    // episode ended after the last step: terminated, time limit or forced
        Tensor<FLAG_SPEC> reset;        // mask applied by the last begin_step
        Tensor<FLAG_SPEC> forced;       // pending external reset (scene rotation, epoch boundary)
        Tensor<VALUE_SPEC> episode_return;
        Tensor<REASON_SPEC> end_reason;
        // per-step outputs of begin_step: the episode that ended for each instance reset in that step
        Tensor<FLAG_SPEC> finished;
        Tensor<COUNTER_SPEC> finished_length;
        Tensor<VALUE_SPEC> finished_return;
        Tensor<REASON_SPEC> finished_reason;
        TI step_limit = SPEC::STEP_LIMIT;  // runtime override (curricula)
    };

    // per-rollout storage of the per-step outputs, one row per step (filled by record)
    template <typename T_SPEC, typename T_SPEC::TI T_STEPS>
    struct Log {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI STEPS = T_STEPS;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, STEPS, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> finished;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, STEPS, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> finished_length;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, STEPS, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> finished_return;
        Tensor<tensor::Specification<EndReason, TI, tensor::Shape<TI, STEPS, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> finished_reason;
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
