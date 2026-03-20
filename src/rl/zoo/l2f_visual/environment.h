#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_VISUAL_ENVIRONMENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_VISUAL_ENVIRONMENT_H

#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include "../l2f/environment.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f_visual {
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;

    template <typename DEVICE, typename TYPE_POLICY, typename TI,
        TI NUM_ENVS = 64,
        TI CAM_WIDTH = 64, TI CAM_HEIGHT = 64, TI NUM_PROBES = 64,
        TI EPISODE_LENGTH_S = 5, TI SIMULATION_FREQUENCY = 100,
        typename DOMAIN_RANDOMIZATION_OPTIONS = DefaultParametersDomainRandomizationOptions>
    struct ENVIRONMENT_FACTORY {
        using T = typename TYPE_POLICY::DEFAULT;

        using L2F_FACTORY = rl::zoo::l2f::ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, EPISODE_LENGTH_S, SIMULATION_FREQUENCY, DOMAIN_RANDOMIZATION_OPTIONS>;
        using DYNAMICS_STATIC_PARAMETERS = typename L2F_FACTORY::ENVIRONMENT_STATIC_PARAMETERS;

        using VISUAL_SPEC = rl::environments::l2f_visual::Specification<T, TI,
            DYNAMICS_STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
        using ENVIRONMENT = rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
