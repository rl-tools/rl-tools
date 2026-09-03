#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_ON_POLICY_RUNNER_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_ON_POLICY_RUNNER_OPERATIONS_CPU_H

#include "../operations_cpu.h"
#include "../../../../components/on_policy_runner/on_policy_runner.h"

// glue between the episode bookkeeping and the on-policy dataset columns (terminated, truncated,
// reset = truncated delayed by one step, rewards); kept apart from the core component
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::episodes {
        template <typename SPEC, typename DATASET_SPEC>
        constexpr bool check_dataset(){
            static_assert(DATASET_SPEC::SPEC::N_ENVIRONMENTS == SPEC::INSTANCES, "the dataset must cover all instances");
            return true;
        }
        template <typename DEVICE, typename SPEC, typename DATASET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _record_rollout_start(DEVICE& device, const Episodes<SPEC>& episodes, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename SPEC::TI instance_i){
            using T = typename rl::components::on_policy_runner::Dataset<DATASET_SPEC>::T;
            set(dataset.reset, instance_i, 0, _due(device, episodes, instance_i) ? (T)1 : (T)0);
        }
        template <typename DEVICE, typename SPEC, typename REWARD_SPEC, typename DATASET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _record(DEVICE& device, const Episodes<SPEC>& episodes, const Tensor<REWARD_SPEC>& rewards, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename SPEC::TI step_i, typename SPEC::TI instance_i){
            using T = typename rl::components::on_policy_runner::Dataset<DATASET_SPEC>::T;
            using TI = typename SPEC::TI;
            const TI pos = step_i * SPEC::INSTANCES + instance_i;
            const bool truncated = get(device, episodes.truncated, instance_i);
            set(dataset.rewards, pos, 0, (T)get(device, rewards, instance_i));
            set(dataset.terminated, pos, 0, get(device, episodes.terminated, instance_i) ? (T)1 : (T)0);
            set(dataset.truncated, pos, 0, truncated ? (T)1 : (T)0);
            set(dataset.all_reset, pos + SPEC::INSTANCES, 0, truncated ? (T)1 : (T)0);
        }
    }
    // the reset column of step 0: instances due for a reset at the start of the rollout
    template <typename DEVICE, typename SPEC, typename DATASET_SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void record_rollout_start(DEVICE& device, const rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset){
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_dataset<SPEC, DATASET_SPEC>());
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            rl::environments::hyperdrone::episodes::_record_rollout_start(device, episodes, dataset, instance_i);
        }
    }
    // after end_step: the step's reward and flags, and the next step's reset column
    template <typename DEVICE, typename SPEC, typename REWARD_SPEC, typename DATASET_SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void record(DEVICE& device, const rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, const Tensor<REWARD_SPEC>& rewards, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename SPEC::TI step_i){
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_dataset<SPEC, DATASET_SPEC>());
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            rl::environments::hyperdrone::episodes::_record(device, episodes, rewards, dataset, step_i, instance_i);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
