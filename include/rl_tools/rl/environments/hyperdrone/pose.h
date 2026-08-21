#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_POSE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_POSE_H

#include "../l2f/quaternion_helper.h"
#include "../../../rendering/raytracing/types.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> make_camera_data(const T position[3], const T look_at[3], const T up[3], T fov, T aspect);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {

    template <typename T>
    struct CameraMount {
        T offset_body[3] = {0, 0, 0};
        T forward_body[3] = {1, 0, 0};
        T up_body[3] = {0, 0, 1};
    };

    template <typename T>
    struct CameraRandomization {
        T fov_range = 0;
        T offset_body_range[3] = {0, 0, 0};
        T rotation_body_range[3] = {0, 0, 0};
    };

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void cross_vector(const T a[3], const T b[3], T out[3]){
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T dot_vector(const T a[3], const T b[3]){
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void normalize_or(DEVICE& device, T v[3], T fallback_0, T fallback_1, T fallback_2){
        T norm_sq = dot_vector(v, v);
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
        T axis_dot_in = dot_vector(axis, in);
        T one_minus_c = static_cast<T>(1) - c;
        out[0] = in[0] * c + axis_cross_in[0] * s + axis[0] * axis_dot_in * one_minus_c;
        out[1] = in[1] * c + axis_cross_in[1] * s + axis[1] * axis_dot_in * one_minus_c;
        out[2] = in[2] * c + axis_cross_in[2] * s + axis[2] * axis_dot_in * one_minus_c;
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotate_scene_yaw(const T in[3], T out[3], T scene_yaw_cos, T scene_yaw_sin){
        out[0] = scene_yaw_cos * in[0] - scene_yaw_sin * in[1];
        out[1] = scene_yaw_sin * in[0] + scene_yaw_cos * in[1];
        out[2] = in[2];
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void orthonormalize_camera_mount(DEVICE& device, CameraMount<T>& camera_mount){
        normalize_or(device, camera_mount.forward_body, static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));

        T up_candidate[3] = {
            camera_mount.up_body[0],
            camera_mount.up_body[1],
            camera_mount.up_body[2]
        };
        T up_projection = dot_vector(camera_mount.forward_body, up_candidate);
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            up_candidate[axis_i] -= up_projection * camera_mount.forward_body[axis_i];
        }
        if(dot_vector(up_candidate, up_candidate) <= static_cast<T>(1e-12)){
            up_candidate[0] = static_cast<T>(0);
            up_candidate[1] = static_cast<T>(0);
            up_candidate[2] = static_cast<T>(1);
            if(camera_mount.forward_body[2] * camera_mount.forward_body[2] > static_cast<T>(0.81)){
                up_candidate[1] = static_cast<T>(1);
                up_candidate[2] = static_cast<T>(0);
            }
            up_projection = dot_vector(camera_mount.forward_body, up_candidate);
            for(unsigned axis_i = 0; axis_i < 3; axis_i++){
                up_candidate[axis_i] -= up_projection * camera_mount.forward_body[axis_i];
            }
        }
        normalize_or(device, up_candidate, static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));

        T side_body[3];
        cross_vector(camera_mount.forward_body, up_candidate, side_body);
        normalize_or(device, side_body, static_cast<T>(0), static_cast<T>(-1), static_cast<T>(0));
        cross_vector(side_body, camera_mount.forward_body, camera_mount.up_body);
        normalize_or(device, camera_mount.up_body, static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotate_camera_mount(DEVICE& device, CameraMount<T>& camera_mount, const T axis[3], T angle){
        T forward_body[3];
        T up_body[3];
        rotate_around_axis(device, camera_mount.forward_body, axis, angle, forward_body);
        rotate_around_axis(device, camera_mount.up_body, axis, angle, up_body);
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            camera_mount.forward_body[axis_i] = forward_body[axis_i];
            camera_mount.up_body[axis_i] = up_body[axis_i];
        }
    }

    template <typename DEVICE, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void randomize_camera_mount(DEVICE& device, const CameraRandomization<T>& randomization, CameraMount<T>& camera_mount, RNG& rng){
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            T range = randomization.offset_body_range[axis_i];
            if(range != static_cast<T>(0)){
                camera_mount.offset_body[axis_i] += range * random::uniform_real_distribution(device.random, static_cast<T>(-1), static_cast<T>(1), rng);
            }
        }

        const T axes[3][3] = {
            {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}
        };
        bool rotation_randomized = false;
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            T range = randomization.rotation_body_range[axis_i];
            if(range != static_cast<T>(0)){
                T angle = range * random::uniform_real_distribution(device.random, static_cast<T>(-1), static_cast<T>(1), rng);
                rotate_camera_mount(device, camera_mount, axes[axis_i], angle);
                rotation_randomized = true;
            }
        }
        if(rotation_randomized){
            orthonormalize_camera_mount(device, camera_mount);
        }
    }

    // scene-under-drone: the dynamics state lives near the origin; scene_translation/scene_yaw
    // place the drone's frame inside the scene at camera-production time
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> make_camera(
        DEVICE&,
        const CameraMount<T>& camera_mount,
        T fov,
        const T orientation[4],
        const T state_position[3],
        T aspect,
        const T scene_translation[3],
        T scene_yaw_cos,
        T scene_yaw_sin
    ){
        T cam_pos_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, camera_mount.offset_body, cam_pos_local);
        T cam_forward_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, camera_mount.forward_body, cam_forward_local);
        T cam_up_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, camera_mount.up_body, cam_up_local);

        T state_position_world[3];
        rotate_scene_yaw(state_position, state_position_world, scene_yaw_cos, scene_yaw_sin);
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
        return make_camera_data(position, look_at, up, fov, aspect);
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> make_target_camera(
        DEVICE& device,
        const CameraMount<T>& camera_mount,
        T fov,
        T aspect,
        const T scene_translation[3],
        T scene_yaw_cos,
        T scene_yaw_sin,
        T target_frame_roll,
        T target_frame_pitch
    ){
        T offset_world[3];
        rotate_scene_yaw(camera_mount.offset_body, offset_world, scene_yaw_cos, scene_yaw_sin);
        T forward_body[3] = {
            camera_mount.forward_body[0],
            camera_mount.forward_body[1],
            camera_mount.forward_body[2]
        };
        T up_body[3] = {
            camera_mount.up_body[0],
            camera_mount.up_body[1],
            camera_mount.up_body[2]
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
        return make_camera_data(position, look_at, up, fov, aspect);
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> make_target_camera(
        DEVICE&,
        const CameraMount<T>& camera_mount,
        T fov,
        T aspect,
        const T scene_translation[3],
        T scene_yaw_cos,
        T scene_yaw_sin
    ){
        T offset_world[3];
        rotate_scene_yaw(camera_mount.offset_body, offset_world, scene_yaw_cos, scene_yaw_sin);
        T forward_world[3];
        rotate_scene_yaw(camera_mount.forward_body, forward_world, scene_yaw_cos, scene_yaw_sin);
        T up_world[3];
        rotate_scene_yaw(camera_mount.up_body, up_world, scene_yaw_cos, scene_yaw_sin);

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
        return make_camera_data(position, look_at, up, fov, aspect);
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> interpolate_camera(const rendering::raytracing::Camera<T>& from, const rendering::raytracing::Camera<T>& to, T alpha){
        rendering::raytracing::Camera<T> out;
        for(unsigned i = 0; i < 3; i++){
            out.pos[i] = from.pos[i] + (to.pos[i] - from.pos[i]) * alpha;
            out.dir_00[i] = from.dir_00[i] + (to.dir_00[i] - from.dir_00[i]) * alpha;
            out.dir_du[i] = from.dir_du[i] + (to.dir_du[i] - from.dir_du[i]) * alpha;
            out.dir_dv[i] = from.dir_dv[i] + (to.dir_dv[i] - from.dir_dv[i]) * alpha;
        }
        return out;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
