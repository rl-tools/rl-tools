#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_EPISODES_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_EPISODES_OPERATIONS_GENERIC_H

#include "episodes.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::episodes {
    // single-instance logic shared by the CPU loops and the CUDA kernels
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool _due(DEVICE& device, const Episodes<SPEC>& episodes, typename SPEC::TI instance_i){
        return get(device, episodes.truncated, instance_i) || get(device, episodes.forced, instance_i);
    }
    // applies the reset decision: publishes the finished episode (if one ended) and clears the
    // instance's counters
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void _begin_step(DEVICE& device, Episodes<SPEC>& episodes, typename SPEC::TI instance_i, bool due){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        const TI episode_step = get(device, episodes.episode_step, instance_i);
        const bool finished = due && episode_step > 0;
        set(device, episodes.finished, finished, instance_i);
        set(device, episodes.finished_length, finished ? episode_step : (TI)0, instance_i);
        set(device, episodes.finished_return, finished ? get(device, episodes.episode_return, instance_i) : (T)0, instance_i);
        set(device, episodes.finished_reason, finished ? get(device, episodes.end_reason, instance_i) : EndReason::NONE, instance_i);
        set(device, episodes.reset, due, instance_i);
        if(due){
            set(device, episodes.episode_step, (TI)0, instance_i);
            set(device, episodes.episode_return, (T)0, instance_i);
            set(device, episodes.end_reason, EndReason::NONE, instance_i);
            set(device, episodes.truncated, false, instance_i);
            set(device, episodes.forced, false, instance_i);
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void _end_step(DEVICE& device, Episodes<SPEC>& episodes, typename SPEC::TI instance_i, typename SPEC::T reward){
        using TI = typename SPEC::TI;
        const TI episode_step = get(device, episodes.episode_step, instance_i) + 1;
        set(device, episodes.episode_step, episode_step, instance_i);
        set(device, episodes.episode_return, get(device, episodes.episode_return, instance_i) + reward, instance_i);
        const bool terminated = get(device, episodes.terminated, instance_i);
        const bool time_limit = episodes.step_limit > 0 && episode_step >= episodes.step_limit;
        const bool truncated = terminated || time_limit;
        set(device, episodes.truncated, truncated, instance_i);
        if(truncated){
            set(device, episodes.end_reason, terminated ? EndReason::TERMINATED : EndReason::TIME_LIMIT, instance_i);
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void _force(DEVICE& device, Episodes<SPEC>& episodes, typename SPEC::TI instance_i){
        set(device, episodes.forced, true, instance_i);
        if(!get(device, episodes.truncated, instance_i) && get(device, episodes.episode_step, instance_i) > 0){
            set(device, episodes.end_reason, EndReason::FORCED, instance_i);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
