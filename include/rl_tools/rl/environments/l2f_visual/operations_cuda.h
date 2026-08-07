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
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void cross_vector(const T a[3], const T b[3], T out[3]){
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void normalize_or(DEVICE& device, T v[3], T fallback_0, T fallback_1, T fallback_2){
        T norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if(norm_sq > static_cast<T>(1e-12)){
            T inv_norm = static_cast<T>(1) / math::sqrt(device.math, norm_sq);
            v[0] *= inv_norm;
            v[1] *= inv_norm;
            v[2] *= inv_norm;
        } else {
            v[0] = fallback_0;
            v[1] = fallback_1;
            v[2] = fallback_2;
        }
    }
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotate_around_axis(DEVICE& device, const T in[3], const T axis[3], T angle, T out[3]){
        T c = math::cos(device.math, angle);
        T s = math::sin(device.math, angle);
        T axis_cross_in[3];
        cross_vector(axis, in, axis_cross_in);
        T axis_dot_in = axis[0] * in[0] + axis[1] * in[1] + axis[2] * in[2];
        T one_minus_c = static_cast<T>(1) - c;
        out[0] = in[0] * c + axis_cross_in[0] * s + axis[0] * axis_dot_in * one_minus_c;
        out[1] = in[1] * c + axis_cross_in[1] * s + axis[1] * axis_dot_in * one_minus_c;
        out[2] = in[2] * c + axis_cross_in[2] * s + axis[2] * axis_dot_in * one_minus_c;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_camera_for_state(
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
        T offset_world[3];
        rotate_scene_yaw(parameters.camera_mount.offset_body, offset_world, scene_yaw_cos, scene_yaw_sin);
        T forward_body[3] = {
            parameters.camera_mount.forward_body[0],
            parameters.camera_mount.forward_body[1],
            parameters.camera_mount.forward_body[2]
        };
        T up_body[3] = {
            parameters.camera_mount.up_body[0],
            parameters.camera_mount.up_body[1],
            parameters.camera_mount.up_body[2]
        };
        T pitch_axis[3];
        cross_vector(forward_body, up_body, pitch_axis);
        normalize_or(device, pitch_axis, static_cast<T>(0), static_cast<T>(-1), static_cast<T>(0));
        T pitched_forward_body[3];
        T pitched_up_body[3];
        rotate_around_axis(device, forward_body, pitch_axis, target_frame_pitch, pitched_forward_body);
        rotate_around_axis(device, up_body, pitch_axis, target_frame_pitch, pitched_up_body);
        T roll_axis[3] = {pitched_forward_body[0], pitched_forward_body[1], pitched_forward_body[2]};
        normalize_or(device, roll_axis, static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
        T rolled_up_body[3];
        rotate_around_axis(device, pitched_up_body, roll_axis, target_frame_roll, rolled_up_body);
        T forward_world[3];
        rotate_scene_yaw(pitched_forward_body, forward_world, scene_yaw_cos, scene_yaw_sin);
        T up_world[3];
        rotate_scene_yaw(rolled_up_body, up_world, scene_yaw_cos, scene_yaw_sin);

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
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_target_camera(
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

    __global__ void pixel_depth_to_float_kernel(const uint32_t* __restrict__ frame_buffer, const float* __restrict__ depth_buffer, float* __restrict__ output, int num_cameras, int pixels_per_camera){
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
        int out_base = cam_idx * (pixels_per_camera * 4) + pixel_idx * 4;
        output[out_base + 0] = r;
        output[out_base + 1] = g;
        output[out_base + 2] = b;
        output[out_base + 3] = depth_buffer[global_idx];
    }

    __global__ void depth_to_float_kernel(const float* __restrict__ depth_buffer, float* __restrict__ output, int num_cameras, int pixels_per_camera){
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = num_cameras * pixels_per_camera;
        if(global_idx >= total_pixels){
            return;
        }
        output[global_idx] = depth_buffer[global_idx];
    }
}
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void observe_batch_render_gpu(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const Tensor<CAMERAS_SPEC>& cameras, float* gpu_output_ptr){
        using TI = typename SPEC::TI;
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        set_cameras_async(device, *env.renderer, cameras);
        if constexpr (SPEC::HAS_RGB && SPEC::HAS_DEPTH) {
            render(device, *env.renderer);
        }
        else if constexpr (SPEC::HAS_RGB) {
            render(device, *env.renderer);
        }
        else {
            render(device, *env.renderer);
        }
        TI total_pixels = SPEC::NUM_ENVS * CAM_PIXELS;
        int block_size = 256;
        int grid_size = (total_pixels + block_size - 1) / block_size;
        if constexpr (SPEC::HAS_RGB && SPEC::HAS_DEPTH) {
            const uint32_t* fb_ptr = get_framebuffer_device_ptr(device, *env.renderer);
            const float* depth_ptr = get_depthbuffer_device_ptr(device, *env.renderer);
            rl::environments::l2f_visual::cuda::pixel_depth_to_float_kernel<<<grid_size, block_size>>>(fb_ptr, depth_ptr, gpu_output_ptr, SPEC::NUM_ENVS, CAM_PIXELS);
        }
        else if constexpr (SPEC::HAS_RGB) {
            const uint32_t* fb_ptr = get_framebuffer_device_ptr(device, *env.renderer);
            rl::environments::l2f_visual::cuda::pixel_to_float_kernel<<<grid_size, block_size>>>(fb_ptr, gpu_output_ptr, SPEC::NUM_ENVS, CAM_PIXELS);
        }
        else {
            const float* depth_ptr = get_depthbuffer_device_ptr(device, *env.renderer);
            rl::environments::l2f_visual::cuda::depth_to_float_kernel<<<grid_size, block_size>>>(depth_ptr, gpu_output_ptr, SPEC::NUM_ENVS, CAM_PIXELS);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
