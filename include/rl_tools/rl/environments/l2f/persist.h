#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PERSIST_H
#include "multirotor.h"
#include <type_traits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::environments::Multirotor<SPEC>& environment, GROUP& group){
        static_assert(std::is_trivially_copyable_v<typename SPEC::PARAMETERS>);
        save_binary(device, &environment.parameters, 1, group, "parameters");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::environments::Multirotor<SPEC>& environment, GROUP& group){
        return load_binary(device, &environment.parameters, 1, group, "parameters");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
