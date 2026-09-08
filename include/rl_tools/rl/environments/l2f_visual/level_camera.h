#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_LEVEL_CAMERA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_LEVEL_CAMERA_H


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::level_camera{
    template<typename T_TYPE_POLICY, typename T_TI, T_TI T_WIDTH, T_TI T_HEIGHT>
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr TI WIDTH = T_WIDTH;
        static constexpr TI HEIGHT = T_HEIGHT;
    };
    template<typename T_SPEC>
    struct Camera{
        using SPEC = T_SPEC;
        typename SPEC::T rotation[3][3];
        typename SPEC::T focal;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
