#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_TIMING_OPERATIONS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_TIMING_OPERATIONS_H

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename T_CONFIG>
    void init(rl::loop::steps::timing::TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::timing::TrainingState<T_CONFIG>;
        ts.start_time = std::chrono::high_resolution_clock::now();
        init(static_cast<typename STATE::NEXT&>(ts), seed);
    }

    template <typename T_CONFIG>
    void destroy(rl::loop::steps::timing::TrainingState<T_CONFIG>& ts){
        using STATE = rl::loop::steps::timing::TrainingState<T_CONFIG>;
        destroy(static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename CONFIG>
    bool step(rl::loop::steps::timing::TrainingState<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::timing::TrainingState<CONFIG>;
        bool finished = step(static_cast<typename STATE::NEXT&>(ts));
        if(finished){
            auto now = std::chrono::high_resolution_clock::now();
            log(ts.device, ts.device.logger, "Time: ", std::chrono::duration_cast<std::chrono::milliseconds>(now - ts.start_time).count()/1000.0, "s");
        }
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
