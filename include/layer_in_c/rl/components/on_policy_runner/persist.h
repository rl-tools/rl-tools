#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H

#include <highfive/H5Group.hpp>
#include <layer_in_c/containers/persist.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer, HighFive::Group group){
        save(device, buffer.data, group, "data");
        save(device, buffer.all_observations, group, "all_observations");
        save(device, buffer.observations, group, "observations");
        save(device, buffer.actions, group, "actions");
        save(device, buffer.action_log_probs, group, "action_log_probs");
        save(device, buffer.rewards, group, "rewards");
        save(device, buffer.terminated, group, "terminated");
        save(device, buffer.truncated, group, "truncated");
        save(device, buffer.all_values, group, "all_values");
        save(device, buffer.values, group, "values");
        save(device, buffer.advantages, group, "advantages");
        save(device, buffer.target_values, group, "target_values");
    }
    template <typename DEVICE, typename SPEC>
    void load(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer, HighFive::Group group){
        load(device, buffer.data, group, "data");
        load(device, buffer.all_observations, group, "all_observations");
        load(device, buffer.observations, group, "observations");
        load(device, buffer.actions, group, "actions");
        load(device, buffer.action_log_probs, group, "action_log_probs");
        load(device, buffer.rewards, group, "rewards");
        load(device, buffer.terminated, group, "terminated");
        load(device, buffer.truncated, group, "truncated");
        load(device, buffer.all_values, group, "all_values");
        load(device, buffer.values, group, "values");
        load(device, buffer.advantages, group, "advantages");
        load(device, buffer.target_values, group, "target_values");
    }
}
#endif
