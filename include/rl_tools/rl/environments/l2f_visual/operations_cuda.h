#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H

#include "multirotor_visual.h"
#include <rl_tools/rl/environments/l2f/quaternion_helper.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::cuda{
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotate_scene_yaw(const T in[3], T out[3], T scene_yaw_cos, T scene_yaw_sin){
        out[0] = scene_yaw_cos * in[0] - scene_yaw_sin * in[1];
        out[1] = scene_yaw_sin * in[0] + scene_yaw_cos * in[1];
        out[2] = in[2];
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::CameraData<typename SPEC::T> make_camera_for_state(
        DEVICE&,
        const typename MultirrotorVisual<SPEC>::Parameters& parameters,
        const typename MultirrotorVisual<SPEC>::State& state,
        typename SPEC::T aspect,
        const typename SPEC::T scene_translation[3],
        typename SPEC::T scene_yaw_cos,
        typename SPEC::T scene_yaw_sin
    ){
        using T = typename SPEC::T;
        T cam_pos_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.offset_body, cam_pos_local);
        T cam_forward_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.forward_body, cam_forward_local);
        T cam_up_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.up_body, cam_up_local);

        T state_position_world[3];
        rotate_scene_yaw(state.position, state_position_world, scene_yaw_cos, scene_yaw_sin);
        T cam_pos_world[3];
        rotate_scene_yaw(cam_pos_local, cam_pos_world, scene_yaw_cos, scene_yaw_sin);
        T cam_forward_world[3];
        rotate_scene_yaw(cam_forward_local, cam_forward_world, scene_yaw_cos, scene_yaw_sin);
        T cam_up_world[3];
        rotate_scene_yaw(cam_up_local, cam_up_world, scene_yaw_cos, scene_yaw_sin);

        T position[3] = {
            state_position_world[0] + cam_pos_world[0] + scene_translation[0],
            state_position_world[1] + cam_pos_world[1] + scene_translation[1],
            state_position_world[2] + cam_pos_world[2] + scene_translation[2]
        };
        T look_at[3] = {
            position[0] + cam_forward_world[0],
            position[1] + cam_forward_world[1],
            position[2] + cam_forward_world[2]
        };
        T up[3] = {cam_up_world[0], cam_up_world[1], cam_up_world[2]};
        return make_camera_data(position, look_at, up, parameters.fov, aspect);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::CameraData<typename SPEC::T> make_target_camera(
        DEVICE&,
        const typename MultirrotorVisual<SPEC>::Parameters& parameters,
        typename SPEC::T aspect,
        const typename SPEC::T scene_translation[3],
        typename SPEC::T scene_yaw_cos,
        typename SPEC::T scene_yaw_sin
    ){
        using T = typename SPEC::T;
        T offset_world[3];
        rotate_scene_yaw(parameters.camera_mount.offset_body, offset_world, scene_yaw_cos, scene_yaw_sin);
        T forward_world[3];
        rotate_scene_yaw(parameters.camera_mount.forward_body, forward_world, scene_yaw_cos, scene_yaw_sin);
        T up_world[3];
        rotate_scene_yaw(parameters.camera_mount.up_body, up_world, scene_yaw_cos, scene_yaw_sin);

        T position[3] = {
            scene_translation[0] + offset_world[0],
            scene_translation[1] + offset_world[1],
            scene_translation[2] + offset_world[2]
        };
        T look_at[3] = {
            position[0] + forward_world[0],
            position[1] + forward_world[1],
            position[2] + forward_world[2]
        };
        T up[3] = {up_world[0], up_world[1], up_world[2]};
        return make_camera_data(position, look_at, up, parameters.fov, aspect);
    }

    __global__ void pixel_to_float_kernel(const uint32_t* __restrict__ frame_buffer, float* __restrict__ output, int num_cameras, int pixels_per_camera){
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = num_cameras * pixels_per_camera;
        if(global_idx >= total_pixels){
            return;
        }
        int cam_idx = global_idx / pixels_per_camera;
        int pixel_idx = global_idx % pixels_per_camera;
        uint32_t rgba = frame_buffer[global_idx];
        float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
        float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
        float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;
        int out_base = cam_idx * (pixels_per_camera * 3) + pixel_idx * 3;
        output[out_base + 0] = r;
        output[out_base + 1] = g;
        output[out_base + 2] = b;
    }
}
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void observe_batch_render_gpu(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const Tensor<CAMERAS_SPEC>& cameras, float* gpu_output_ptr){
        using TI = typename SPEC::TI;
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        set_cameras_async(device, *env.renderer, cameras);
        render_rgb_only(device, *env.renderer);
        const uint32_t* fb_ptr = get_framebuffer_device_ptr(device, *env.renderer);
        TI total_pixels = SPEC::NUM_ENVS * CAM_PIXELS;
        int block_size = 256;
        int grid_size = (total_pixels + block_size - 1) / block_size;
        rl::environments::l2f_visual::cuda::pixel_to_float_kernel<<<grid_size, block_size>>>(fb_ptr, gpu_output_ptr, SPEC::NUM_ENVS, CAM_PIXELS);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
