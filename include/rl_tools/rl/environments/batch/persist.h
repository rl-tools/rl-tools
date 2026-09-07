#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_BATCH_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_BATCH_PERSIST_H
#include "environment.h"
#include <string>
#include <type_traits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::environments::batch::Independent<SPEC>& environment, GROUP& group){
        if constexpr(!std::is_empty_v<typename SPEC::ENVIRONMENT>){
            for(typename SPEC::TI i = 0; i < SPEC::INSTANCES; i++){
                auto instance = create_group(device, group, std::to_string(i).c_str());
                save(device, get_ref(device, environment.environments, i), instance);
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::environments::batch::Independent<SPEC>& environment, GROUP& group){
        bool success = true;
        if constexpr(!std::is_empty_v<typename SPEC::ENVIRONMENT>){
            for(typename SPEC::TI i = 0; i < SPEC::INSTANCES; i++){
                auto instance = get_group(device, group, std::to_string(i).c_str());
                success &= load(device, get_ref(device, environment.environments, i), instance);
            }
        }
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
