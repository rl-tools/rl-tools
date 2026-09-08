#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_FLOW_AUXILIARY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_FLOW_AUXILIARY_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::l2f_visual::flow_auxiliary{
        template<typename T_TYPE_POLICY, typename T_TI, T_TI T_HEIGHT, T_TI T_WIDTH, T_TI T_GRID>
        struct Specification{
            using TYPE_POLICY = T_TYPE_POLICY;
            using T = typename TYPE_POLICY::DEFAULT;
            using TI = T_TI;
            static constexpr TI HEIGHT = T_HEIGHT;
            static constexpr TI WIDTH = T_WIDTH;
            static constexpr TI GRID = T_GRID;
            static constexpr TI DIM = 2 * GRID * GRID;
            static constexpr T SCALE = 5;
            static_assert(GRID > 0 && GRID <= HEIGHT && GRID <= WIDTH);
        };
    }

    template<typename DEVICE, typename SPEC, typename T_INPUT>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T pool_flow_component(DEVICE& device, const SPEC&, const T_INPUT* flow, typename SPEC::TI camera, typename SPEC::TI component){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        const TI axis = component % 2;
        const TI cell = component / 2;
        const TI y0 = (cell / SPEC::GRID) * SPEC::HEIGHT / SPEC::GRID;
        const TI y1 = (cell / SPEC::GRID + 1) * SPEC::HEIGHT / SPEC::GRID;
        const TI x0 = (cell % SPEC::GRID) * SPEC::WIDTH / SPEC::GRID;
        const TI x1 = (cell % SPEC::GRID + 1) * SPEC::WIDTH / SPEC::GRID;
        const T dimension = static_cast<T>(axis == 0 ? SPEC::WIDTH : SPEC::HEIGHT);
        T sum = 0;
        for(TI y = y0; y < y1; y++){
            for(TI x = x0; x < x1; x++){
                const TI offset = ((camera * SPEC::HEIGHT + y) * SPEC::WIDTH + x) * 2 + axis;
                sum += math::clamp(device.math, static_cast<T>(flow[offset]) / dimension, static_cast<T>(-1), static_cast<T>(1));
            }
        }
        return SPEC::SCALE * sum / static_cast<T>((y1 - y0) * (x1 - x0));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
