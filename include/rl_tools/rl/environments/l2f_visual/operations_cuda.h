#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H

#include "multirotor_visual.h"
#include <rl_tools/rl/environments/hyperdrone/pose.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::cuda{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_camera_for_state(
        DEVICE& device,
        const typename MultirrotorVisual<SPEC>::Parameters& parameters,
        const typename MultirrotorVisual<SPEC>::State& state,
        typename SPEC::T aspect,
        const typename SPEC::T scene_translation[3],
        typename SPEC::T scene_yaw_cos,
        typename SPEC::T scene_yaw_sin
    ){
        using T = typename SPEC::T;
        return hyperdrone::make_camera<DEVICE, T>(device, parameters.camera_mount, parameters.fov, state.orientation, state.position, aspect, scene_translation, scene_yaw_cos, scene_yaw_sin);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_target_camera(
        DEVICE& device,
        const typename MultirrotorVisual<SPEC>::Parameters& parameters,
        typename SPEC::T aspect,
        const typename SPEC::T scene_translation[3],
        typename SPEC::T scene_yaw_cos,
        typename SPEC::T scene_yaw_sin,
        typename SPEC::T target_frame_roll,
        typename SPEC::T target_frame_pitch
    ){
        using T = typename SPEC::T;
        return hyperdrone::make_target_camera<DEVICE, T>(device, parameters.camera_mount, parameters.fov, aspect, scene_translation, scene_yaw_cos, scene_yaw_sin, target_frame_roll, target_frame_pitch);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_target_camera(
        DEVICE& device,
        const typename MultirrotorVisual<SPEC>::Parameters& parameters,
        typename SPEC::T aspect,
        const typename SPEC::T scene_translation[3],
        typename SPEC::T scene_yaw_cos,
        typename SPEC::T scene_yaw_sin
    ){
        using T = typename SPEC::T;
        return hyperdrone::make_target_camera<DEVICE, T>(device, parameters.camera_mount, parameters.fov, aspect, scene_translation, scene_yaw_cos, scene_yaw_sin);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
