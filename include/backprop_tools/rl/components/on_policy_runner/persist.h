#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H

#include <highfive/H5Group.hpp>
#include "../../../containers/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, HighFive::Group group){
        save(device, dataset.data, group, "data");
        save(device, dataset.all_observations, group, "all_observations");
        save(device, dataset.observations, group, "observations");
        save(device, dataset.actions, group, "actions");
        save(device, dataset.action_log_probs, group, "action_log_probs");
        save(device, dataset.rewards, group, "rewards");
        save(device, dataset.terminated, group, "terminated");
        save(device, dataset.truncated, group, "truncated");
        save(device, dataset.all_values, group, "all_values");
        save(device, dataset.values, group, "values");
        save(device, dataset.advantages, group, "advantages");
        save(device, dataset.target_values, group, "target_values");
    }
    template <typename DEVICE, typename SPEC>
    void load(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, HighFive::Group group){
        load(device, dataset.data, group, "data");
        load(device, dataset.all_observations, group, "all_observations");
        load(device, dataset.observations, group, "observations");
        load(device, dataset.actions, group, "actions");
        load(device, dataset.action_log_probs, group, "action_log_probs");
        load(device, dataset.rewards, group, "rewards");
        load(device, dataset.terminated, group, "terminated");
        load(device, dataset.truncated, group, "truncated");
        load(device, dataset.all_values, group, "all_values");
        load(device, dataset.values, group, "values");
        load(device, dataset.advantages, group, "advantages");
        load(device, dataset.target_values, group, "target_values");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
