#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_PERSIST_H
#include "multirotor_visual.h"
#include "../l2f/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& environment, GROUP& group){
        static_assert(std::is_trivially_copyable_v<typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters>);
        auto dynamics = create_group(device, group, "dynamics");
        save(device, environment.dynamics, dynamics);
        save_binary(device, &environment.parameters, 1, group, "parameters");
        save_binary(device, &environment.use_target_mode, 1, group, "use_target_mode");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& environment, GROUP& group){
        auto dynamics = get_group(device, group, "dynamics");
        bool success = load(device, environment.dynamics, dynamics);
        success &= load_binary(device, &environment.parameters, 1, group, "parameters");
        success &= load_binary(device, &environment.use_target_mode, 1, group, "use_target_mode");
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
