#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_H

#include "multirotor_visual.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters) {
        parameters = env.parameters;
        initial_parameters(device, env.dynamics, parameters.dynamics);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, RNG& rng) {
        using T = typename SPEC::T;
        parameters = env.parameters;
        sample_initial_parameters(device, env.dynamics, parameters.dynamics, rng);
        parameters.fov = env.parameters.fov + env.parameters.camera_randomization.fov_range * random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
        rl::environments::hyperdrone::randomize_camera_mount(device, env.parameters.camera_randomization, parameters.camera_mount, rng);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
