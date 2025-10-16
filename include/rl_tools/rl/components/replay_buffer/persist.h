#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_PERSIST_H

#include "replay_buffer.h"
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb, GROUP& group) {
        static_assert(decltype(rb.rewards)::COLS == 1);
        static_assert(decltype(rb.terminated)::COLS == 1);
        static_assert(decltype(rb.truncated)::COLS == 1);
        save(device, rb.observations, group, "observations");
        save(device, rb.actions, group, "actions");
        save(device, rb.rewards, group, "rewards");
        save(device, rb.next_observations, group, "next_observations");
        save(device, rb.terminated, group, "terminated");
        save(device, rb.truncated, group, "truncated");

        save(device, rb.episode_start, group, "episode_start");

        std::vector<decltype(rb.position)> position;
        position.push_back(rb.position);
        create_dataset(device, group, "position", position);

        std::vector<decltype(rb.position)> full;
        full.push_back(rb.full);
        create_dataset(device, group, "full", full);
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb, GROUP& group) {
        static_assert(decltype(rb.rewards)::COLS == 1);
        static_assert(decltype(rb.terminated)::COLS == 1);
        static_assert(decltype(rb.truncated)::COLS == 1);
        load(device, rb.observations, group, "observations");
        load(device, rb.actions, group, "actions");
        load(device, rb.rewards, group, "rewards");
        load(device, rb.next_observations, group, "next_observations");
        load(device, rb.terminated, group, "terminated");
        load(device, rb.truncated, group, "truncated");

        load(device, rb.episode_start, group, "episode_start");

        std::vector<decltype(rb.position)> position;
        read_dataset(device, group, "position", position);
        rb.position = position[0];

        std::vector<decltype(rb.position)> full;
        read_dataset(device, group, "full", full);
        rb.full = full[0];
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
