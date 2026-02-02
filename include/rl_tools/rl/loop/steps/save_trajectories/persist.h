#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_PERSIST_H
#include "state.h"
#include "../../../../random/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    void save(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        save(device, static_cast<typename STATE::NEXT&>(ts), group);
        auto save_trajectories_group = create_group(device, group, "save_trajectories");
        auto rng_group = create_group(device, save_trajectories_group, "rng_save_trajectories");
        save(device, ts.rng_save_trajectories, rng_group);
    }
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    bool load(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        bool success = load(device, static_cast<typename STATE::NEXT&>(ts), group);
        auto save_trajectories_group = get_group(device, group, "save_trajectories");
        auto rng_group = get_group(device, save_trajectories_group, "rng_save_trajectories");
        bool step_result = load(device, ts.rng_save_trajectories, rng_group);
        if(!step_result){ log(device, device.logger, "Save trajectories loop load failed: rng_save_trajectories"); }
        success &= step_result;
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
