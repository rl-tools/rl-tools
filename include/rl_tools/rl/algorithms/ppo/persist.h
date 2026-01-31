#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_PERSIST_H
#include "ppo.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo, GROUP& group) {
        auto actor_group = create_group(device, group, "actor");
        save(device, ppo.actor, actor_group);
        auto critic_group = create_group(device, group, "critic");
        save(device, ppo.critic, critic_group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo, GROUP& group) {
        auto actor_group = get_group(device, group, "actor");
        bool success = load(device, ppo.actor, actor_group);
        auto critic_group = get_group(device, group, "critic");
        success &= load(device, ppo.critic, critic_group);
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        ppo.initialized = true;
#endif
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
