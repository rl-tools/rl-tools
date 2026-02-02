#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_NN_ANALYTICS_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_NN_ANALYTICS_PERSIST_H
#include "state.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    void save(DEVICE& device, rl::loop::steps::nn_analytics::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::nn_analytics::State<T_CONFIG>;
        save(device, static_cast<typename STATE::NEXT&>(ts), group);
    }
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    bool load(DEVICE& device, rl::loop::steps::nn_analytics::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::nn_analytics::State<T_CONFIG>;
        return load(device, static_cast<typename STATE::NEXT&>(ts), group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
