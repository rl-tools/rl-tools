#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_H

#include "multirotor_visual.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual {
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
}

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
        rl::environments::l2f_visual::randomize_camera_mount(device, env.parameters.camera_randomization, parameters.camera_mount, rng);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
