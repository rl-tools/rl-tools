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

    namespace rendering::camera_orbit{
        // upper-hemisphere-biased golden-spiral orbit for viewing architectural scenes:
        // inclination_range < 1 keeps cameras off the poles, and a camera landing more than
        // floor_margin * radius below the center is re-lifted to relift_height * radius above it
        template <typename T>
        struct Parameters {
            T inclination_range = 0.85;
            T floor_margin = 0.1;
            T relift_height = 0.3;
        };
    }
    template <typename T, typename TI>
    void generate_camera_orbit(rendering::Camera<T>* poses_out, TI num_cameras, const T center[3], T radius, const T up[3], T fov_degrees, T aspect, const rendering::camera_orbit::Parameters<T>& parameters = {}){
        const T golden_ratio = (T{1} + sqrtf(T{5})) / T{2};
        for(TI i = 0; i < num_cameras; i++){
            T theta = T{2} * (T)M_PI * i / golden_ratio;
            T cos_inclination = (T{1} - T{2} * (i + T{0.5}) / num_cameras) * parameters.inclination_range;
            T sin_inclination = sqrtf(T{1} - cos_inclination * cos_inclination);
            T position[3] = {
                center[0] + radius * sin_inclination * cosf(theta),
                center[1] + radius * sin_inclination * sinf(theta),
                center[2] + radius * cos_inclination
            };
            if(position[2] < center[2] - radius * parameters.floor_margin){
                position[2] = center[2] + radius * parameters.relift_height;
            }
            poses_out[i] = make_camera_data(position, center, up, fov_degrees, aspect);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
