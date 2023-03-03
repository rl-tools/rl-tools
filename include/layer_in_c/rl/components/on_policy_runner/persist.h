#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H

#include <highfive/H5Group.hpp>
#include <layer_in_c/containers/persist.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer, HighFive::Group group){
        save(device, buffer.data, group, "data");
        save(device, buffer.observations, group, "observations");
        save(device, buffer.actions, group, "actions");
        save(device, buffer.action_log_probs, group, "action_log_probs");
        save(device, buffer.rewards, group, "rewards");
        save(device, buffer.terminated, group, "terminated");
        save(device, buffer.truncated, group, "truncated");
        save(device, buffer.value, group, "value");
        save(device, buffer.advantage, group, "advantage");
        save(device, buffer.target_value, group, "target_value");
    }
    template <typename DEVICE, typename SPEC>
    void load(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer, HighFive::Group group){
        load(device, buffer.data, group, "data");
        load(device, buffer.observations, group, "observations");
        load(device, buffer.actions, group, "actions");
        load(device, buffer.action_log_probs, group, "action_log_probs");
        load(device, buffer.rewards, group, "rewards");
        load(device, buffer.terminated, group, "terminated");
        load(device, buffer.truncated, group, "truncated");
        load(device, buffer.value, group, "value");
        load(device, buffer.advantage, group, "advantage");
        load(device, buffer.target_value, group, "target_value");
    }
}
#endif
