#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_TIMING_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_TIMING_OPERATIONS_GENERIC_H

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::timing::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::timing::State<T_CONFIG>;
        ts.start_time = std::chrono::high_resolution_clock::now();
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::timing::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::timing::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::timing::State<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::timing::State<CONFIG>;
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        if(finished){
            auto now = std::chrono::high_resolution_clock::now();
            log(device, device.logger, "Time: ", std::chrono::duration_cast<std::chrono::milliseconds>(now - ts.start_time).count()/1000.0, "s");
        }
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
