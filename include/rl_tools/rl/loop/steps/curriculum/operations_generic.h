#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_CURRICULUM_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_CURRICULUM_OPERATIONS_GENERIC_H

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void curriculum(DEVICE& device, rl::loop::steps::curriculum::State<T_CONFIG>& ts, rl::loop::steps::curriculum::DefaultTag){
        // default: no-op
    }
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::loop::steps::curriculum::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::curriculum::State<T_CONFIG>;
        malloc(device, static_cast<typename STATE::NEXT&>(ts));
    }
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::loop::steps::curriculum::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::curriculum::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::loop::steps::curriculum::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::curriculum::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        ts.curriculum_level = 0;
    }
    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::curriculum::State<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::curriculum::State<CONFIG>;
        using PARAMS = typename CONFIG::CURRICULUM_PARAMETERS;
        constexpr TI EVAL_INTERVAL = CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL;
        constexpr TI INTERVAL = PARAMS::INTERVAL > 0 ? PARAMS::INTERVAL : EVAL_INTERVAL;
        TI step_before = ts.step;
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        if(step_before % INTERVAL == 0){
            curriculum(device, ts, typename CONFIG::CURRICULUM_TAG{});
            add_scalar(device, device.logger, "curriculum/level", ts.curriculum_level);
        }
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
