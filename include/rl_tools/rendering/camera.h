#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_CAMERA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_CAMERA_H

#include "../rl_tools.h"

#include <math.h>

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

// producer-side camera vocabulary: the pinhole parameterization consumed by the renderer plus the
// vector math to construct it (CUDA producers define RL_TOOLS_FUNCTION_PLACEMENT before including)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering{
        template <typename T_T>
        struct Camera {
            T_T pos[3];
            T_T dir_00[3];
            T_T dir_du[3];
            T_T dir_dv[3];
        };

        // user-facing FOV values are degrees everywhere; trig consumers convert through this —
        // the double-precision product keeps float FOVs bit-identical to the legacy radian
        // constants (e.g. 80 degrees reproduces the golden-pinned 1.3962634015954636 exactly)
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr T degrees_to_radians(T degrees){
            return (T)((double)degrees * 0.017453292519943295);
        }
    }
    namespace rendering::vec3{
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void sub(const T a[3], const T b[3], T out[3]){
            out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T dot(const T a[3], const T b[3]){
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void cross(const T a[3], const T b[3], T out[3]){
            out[0] = a[1]*b[2] - a[2]*b[1];
            out[1] = a[2]*b[0] - a[0]*b[2];
            out[2] = a[0]*b[1] - a[1]*b[0];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T length(const T v[3]){
            return sqrtf(dot(v, v));
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void normalize(const T v[3], T out[3]){
            T len = length(v);
            out[0] = v[0]/len; out[1] = v[1]/len; out[2] = v[2]/len;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void scale(const T v[3], T s, T out[3]){
            out[0] = v[0]*s; out[1] = v[1]*s; out[2] = v[2]*s;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void add(const T a[3], const T b[3], T out[3]){
            out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void cross_normalized(const T a[3], const T b[3], T out[3]){
            T tmp[3];
            cross(a, b, tmp);
            normalize(tmp, out);
        }
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::Camera<T> make_camera_data(const T position[3], const T look_at[3], const T up[3], T fov_degrees, T aspect){
        namespace v3 = rendering::vec3;
        T raw_dir[3], dir[3];
        v3::sub(look_at, position, raw_dir);
        v3::normalize(raw_dir, dir);

        T fov = rendering::degrees_to_radians(fov_degrees);
        T image_plane_scale = T{2} * tanf(fov / T{2});

        T du_dir[3], du[3];
        v3::cross_normalized(dir, up, du_dir);
        v3::scale(du_dir, image_plane_scale, du);

        T dv_dir[3], dv[3];
        v3::cross_normalized(du, dir, dv_dir);
        v3::scale(dv_dir, image_plane_scale / aspect, dv);

        T half_du[3], half_dv[3], tmp[3];
        v3::scale(du, T{-0.5}, half_du);
        v3::scale(dv, T{0.5}, half_dv);
        v3::add(dir, half_du, tmp);
        rendering::Camera<T> cam;
        v3::add(tmp, half_dv, cam.dir_00);
        cam.pos[0] = position[0]; cam.pos[1] = position[1]; cam.pos[2] = position[2];
        cam.dir_du[0] = du[0]; cam.dir_du[1] = du[1]; cam.dir_du[2] = du[2];
        v3::scale(dv, T{-1}, cam.dir_dv);
        return cam;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
