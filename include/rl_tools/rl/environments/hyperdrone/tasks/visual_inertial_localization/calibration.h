#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_CALIBRATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_CALIBRATION_H

#include "../../pose.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    // exact closed form of the renderer's camera: make_camera_data spans the image plane with
    // dir_du/dir_dv of length 2*tan(fov/2) (horizontal; vertical divided by aspect = W/H) and
    // the ray generator samples pixel centers at (u + 0.5)/W. That is an ideal pinhole with
    // square pixels, no distortion, no skew, and a principal point at the image center
    template <typename T>
    struct PinholeIntrinsics {
        T fx;
        T fy;
        T cx;
        T cy;
    };
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT PinholeIntrinsics<T> pinhole_intrinsics(DEVICE& device, TI width, TI height, T fov_horizontal_degrees){
        T fov = fov_horizontal_degrees / (T)180 * math::PI<T>;
        T tangent = math::sin(device.math, fov / (T)2) / math::cos(device.math, fov / (T)2);
        T focal = (T)width / ((T)2 * tangent);
        return {focal, focal, (T)width / (T)2 - (T)0.5, (T)height / (T)2 - (T)0.5};
    }
    // camera-to-body transform in computer-vision camera convention (x = right, y = down,
    // z = forward): columns of the rotation are the camera axes in the body frame, derived with
    // the same cross-product chain as make_camera_data (right = forward x up, down = forward x right)
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void camera_to_body_transform(DEVICE& device, const CameraMount<T>& mount, T rotation_out[3][3], T translation_out[3]){
        using TI = typename DEVICE::index_t;
        T forward[3], right[3], down[3];
        T forward_norm = math::sqrt(device.math, mount.forward_body[0] * mount.forward_body[0] + mount.forward_body[1] * mount.forward_body[1] + mount.forward_body[2] * mount.forward_body[2]);
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            forward[dim_i] = mount.forward_body[dim_i] / forward_norm;
        }
        right[0] = forward[1] * mount.up_body[2] - forward[2] * mount.up_body[1];
        right[1] = forward[2] * mount.up_body[0] - forward[0] * mount.up_body[2];
        right[2] = forward[0] * mount.up_body[1] - forward[1] * mount.up_body[0];
        T right_norm = math::sqrt(device.math, right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            right[dim_i] /= right_norm;
        }
        down[0] = forward[1] * right[2] - forward[2] * right[1];
        down[1] = forward[2] * right[0] - forward[0] * right[2];
        down[2] = forward[0] * right[1] - forward[1] * right[0];
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            rotation_out[dim_i][0] = right[dim_i];
            rotation_out[dim_i][1] = down[dim_i];
            rotation_out[dim_i][2] = forward[dim_i];
            translation_out[dim_i] = mount.offset_body[dim_i];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
