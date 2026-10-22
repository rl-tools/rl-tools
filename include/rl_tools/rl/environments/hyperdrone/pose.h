#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_POSE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_POSE_H

#include "../l2f/quaternion_helper.h"
#include "../../../hyperlux/types.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT hyperlux::Camera<T> make_camera_data(const T position[3], const T look_at[3], const T up[3], T fov, T aspect);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {

    // orientation is a (w, x, y, z) quaternion mapping the camera's FLU axes into the body frame
    template <typename T>
    struct Pose {
        T position[3] = {0, 0, 0};
        T orientation[4] = {1, 0, 0, 0};
    };
    template <typename T>
    struct CameraMount {
        Pose<T> pose;
    };
    template <typename T>
    struct Camera {
        CameraMount<T> mount;
        T fov = 63.78166175396324; // horizontal, degrees
    };

    template <typename T>
    struct CameraRandomization {
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
    RL_TOOLS_FUNCTION_PLACEMENT void mount_axes(DEVICE&, const CameraMount<T>& mount, T forward_body[3], T up_body[3]){
        const T forward_camera[3] = {1, 0, 0};
        const T up_camera[3] = {0, 0, 1};
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(mount.pose.orientation, forward_camera, forward_body);
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(mount.pose.orientation, up_camera, up_body);
    }

    template <typename DEVICE, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void randomize_camera_mount(DEVICE& device, const CameraRandomization<T>& randomization, CameraMount<T>& mount, RNG& rng){
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            T range = randomization.offset_body_range[axis_i];
            if(range != static_cast<T>(0)){
                mount.pose.position[axis_i] += range * random::uniform_real_distribution(device.random, static_cast<T>(-1), static_cast<T>(1), rng);
            }
        }
        // rotations about the fixed body axes x, y, z (applied in that order)
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            T range = randomization.rotation_body_range[axis_i];
            if(range != static_cast<T>(0)){
                T half_angle = range * random::uniform_real_distribution(device.random, static_cast<T>(-1), static_cast<T>(1), rng) / static_cast<T>(2);
                T rotation[4] = {math::cos(device.math, half_angle), 0, 0, 0};
                rotation[1 + axis_i] = math::sin(device.math, half_angle);
                T orientation[4];
                rl::environments::l2f::quaternion_multiplication<DEVICE, T>(rotation, mount.pose.orientation, orientation);
                for(unsigned component_i = 0; component_i < 4; component_i++){
                    mount.pose.orientation[component_i] = orientation[component_i];
                }
            }
        }
    }

    // scene-under-drone: the dynamics state lives near the origin; scene_translation/scene_yaw
    // place the drone's frame inside the scene at camera-production time
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT hyperlux::Camera<T> make_camera(
        DEVICE& device,
        const Camera<T>& camera,
        const T orientation[4],
        const T state_position[3],
        T aspect,
        const T scene_translation[3],
        T scene_yaw_cos,
        T scene_yaw_sin
    ){
        T forward_body[3];
        T up_body[3];
        mount_axes(device, camera.mount, forward_body, up_body);
        T cam_pos_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, camera.mount.pose.position, cam_pos_local);
        T cam_forward_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, forward_body, cam_forward_local);
        T cam_up_local[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, up_body, cam_up_local);

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
        return make_camera_data(position, look_at, up, camera.fov, aspect);
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT hyperlux::Camera<T> make_target_camera(
        DEVICE& device,
        const Camera<T>& camera,
        T aspect,
        const T scene_translation[3],
        T scene_yaw_cos,
        T scene_yaw_sin,
        T target_frame_roll,
        T target_frame_pitch
    ){
        T offset_world[3];
        rotate_scene_yaw(camera.mount.pose.position, offset_world, scene_yaw_cos, scene_yaw_sin);
        T forward_body[3];
        T up_body[3];
        mount_axes(device, camera.mount, forward_body, up_body);
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
        return make_camera_data(position, look_at, up, camera.fov, aspect);
    }

    // the body pose (scene-under-drone frame) that carries the mount to make_target_camera's view:
    // the target perturbation turns the body rigidly about the camera
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void make_target_body_pose(DEVICE& device, const Camera<T>& camera, T target_frame_roll, T target_frame_pitch, T orientation[4], T position[3]){
        T forward_body[3];
        T up_body[3];
        mount_axes(device, camera.mount, forward_body, up_body);
        T pitch_axis[3];
        cross_vector(forward_body, up_body, pitch_axis);
        normalize_or(device, pitch_axis, static_cast<T>(0), static_cast<T>(-1), static_cast<T>(0));
        T roll_axis[3];
        rotate_around_axis(device, forward_body, pitch_axis, target_frame_pitch, roll_axis);
        normalize_or(device, roll_axis, static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
        const T pitch_sin = math::sin(device.math, target_frame_pitch / 2);
        const T roll_sin = math::sin(device.math, target_frame_roll / 2);
        const T pitch_rotation[4] = {math::cos(device.math, target_frame_pitch / 2), pitch_sin * pitch_axis[0], pitch_sin * pitch_axis[1], pitch_sin * pitch_axis[2]};
        const T roll_rotation[4] = {math::cos(device.math, target_frame_roll / 2), roll_sin * roll_axis[0], roll_sin * roll_axis[1], roll_sin * roll_axis[2]};
        rl::environments::l2f::quaternion_multiplication<DEVICE, T>(roll_rotation, pitch_rotation, orientation);
        T mount_turned[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(orientation, camera.mount.pose.position, mount_turned);
        for(unsigned axis_i = 0; axis_i < 3; axis_i++){
            position[axis_i] = camera.mount.pose.position[axis_i] - mount_turned[axis_i];
        }
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT hyperlux::Camera<T> interpolate_camera(const hyperlux::Camera<T>& from, const hyperlux::Camera<T>& to, T alpha){
        hyperlux::Camera<T> out;
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
