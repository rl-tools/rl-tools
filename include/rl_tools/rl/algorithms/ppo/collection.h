#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_COLLECTION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_COLLECTION_H

#include "ppo.h"
#include "../../components/on_policy_runner/on_policy_runner.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::ppo{
    template <typename T_CRITIC, typename T_DATASET_SPEC, bool = T_DATASET_SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS>
    struct CollectionBuffer;
    template <typename T_CRITIC, typename T_DATASET_SPEC>
    struct CollectionBuffer<T_CRITIC, T_DATASET_SPEC, true>{
        using RUNNER_SPEC = typename T_DATASET_SPEC::SPEC;
        using TI = typename RUNNER_SPEC::TI;
        using CRITIC = typename T_CRITIC::template CHANGE_BATCH_SIZE<TI, RUNNER_SPEC::N_ENVIRONMENTS>;
        typename CRITIC::template State<RUNNER_SPEC::DYNAMIC_ALLOCATION> state, bootstrap_state;
        typename CRITIC::template Buffer<RUNNER_SPEC::DYNAMIC_ALLOCATION> buffer;
    };
    template <typename T_CRITIC, typename T_DATASET_SPEC>
    struct CollectionBuffer<T_CRITIC, T_DATASET_SPEC, false>{
        using RUNNER_SPEC = typename T_DATASET_SPEC::SPEC;
        using TI = typename RUNNER_SPEC::TI;
        using CRITIC = typename T_CRITIC::template CHANGE_BATCH_SIZE<TI, T_DATASET_SPEC::STEPS_TOTAL_ALL>;
        typename CRITIC::template Buffer<RUNNER_SPEC::DYNAMIC_ALLOCATION> buffer;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
