#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_TRANSFORMS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_TRANSFORMS_GENERIC_H

#include "../../rl_tools.h"

#include <math.h>

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

// freestanding 3x4 row-major [R|t] transform math shared between the host verbs and device
// producers (CUDA kernels define RL_TOOLS_FUNCTION_PLACEMENT before including this header)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rendering::raytracing::detail{
        RL_TOOLS_FUNCTION_PLACEMENT inline void compose_transforms(const float a[12], const float b[12], float out[12]){
            for(int row = 0; row < 3; row++){
                for(int column = 0; column < 3; column++){
                    out[row * 4 + column] = a[row * 4] * b[column] + a[row * 4 + 1] * b[4 + column] + a[row * 4 + 2] * b[8 + column];
                }
                out[row * 4 + 3] = a[row * 4] * b[3] + a[row * 4 + 1] * b[7] + a[row * 4 + 2] * b[11] + a[row * 4 + 3];
            }
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline void invert_transform(const float transform[12], float out[12]){
            const float a = transform[0], b = transform[1], c = transform[2];
            const float d = transform[4], e = transform[5], f = transform[6];
            const float g = transform[8], h = transform[9], i = transform[10];
            const float cofactor_a = e*i - f*h;
            const float cofactor_b = f*g - d*i;
            const float cofactor_c = d*h - e*g;
            const float inv_det = 1.0f / (a*cofactor_a + b*cofactor_b + c*cofactor_c);
            out[0] = cofactor_a * inv_det; out[1] = (c*h - b*i) * inv_det; out[2]  = (b*f - c*e) * inv_det;
            out[4] = cofactor_b * inv_det; out[5] = (a*i - c*g) * inv_det; out[6]  = (c*d - a*f) * inv_det;
            out[8] = cofactor_c * inv_det; out[9] = (b*g - a*h) * inv_det; out[10] = (a*e - b*d) * inv_det;
            out[3]  = -(out[0]*transform[3] + out[1]*transform[7] + out[2] *transform[11]);
            out[7]  = -(out[4]*transform[3] + out[5]*transform[7] + out[6] *transform[11]);
            out[11] = -(out[8]*transform[3] + out[9]*transform[7] + out[10]*transform[11]);
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline void quaternion_from_transform(const float m[12], float q[4]){
            const float trace = m[0] + m[5] + m[10];
            if(trace > 0){
                const float s = sqrtf(trace + 1.0f) * 2;
                q[0] = 0.25f * s;
                q[1] = (m[9] - m[6]) / s;
                q[2] = (m[2] - m[8]) / s;
                q[3] = (m[4] - m[1]) / s;
            }
            else if(m[0] > m[5] && m[0] > m[10]){
                const float s = sqrtf(1.0f + m[0] - m[5] - m[10]) * 2;
                q[0] = (m[9] - m[6]) / s;
                q[1] = 0.25f * s;
                q[2] = (m[1] + m[4]) / s;
                q[3] = (m[2] + m[8]) / s;
            }
            else if(m[5] > m[10]){
                const float s = sqrtf(1.0f + m[5] - m[0] - m[10]) * 2;
                q[0] = (m[2] - m[8]) / s;
                q[1] = (m[1] + m[4]) / s;
                q[2] = 0.25f * s;
                q[3] = (m[6] + m[9]) / s;
            }
            else{
                const float s = sqrtf(1.0f + m[10] - m[0] - m[5]) * 2;
                q[0] = (m[4] - m[1]) / s;
                q[1] = (m[2] + m[8]) / s;
                q[2] = (m[6] + m[9]) / s;
                q[3] = 0.25f * s;
            }
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline void rotate_by_quaternion(const float q[4], const float v[3], float out[3]){
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            const float ux = y*v[2] - z*v[1], uy = z*v[0] - x*v[2], uz = x*v[1] - y*v[0];
            const float uux = y*uz - z*uy, uuy = z*ux - x*uz, uuz = x*uy - y*ux;
            out[0] = v[0] + 2*(w*ux + uux);
            out[1] = v[1] + 2*(w*uy + uuy);
            out[2] = v[2] + 2*(w*uz + uuz);
        }

        // constant-twist (screw motion) interpolation between two rigid 3x4 transforms: the
        // rotation follows the shortest-arc slerp path and the translation follows the screw of
        // the relative motion, so a fixed rotation axis line stays fixed — a translation lerp
        // instead would cut the chord and displace an off-origin pivot (e.g. a spinning prop hub)
        // at mid-shutter samples. The single interpolant behind set_transform_pair and
        // expand_motion_transforms, so every consumer derives identical per-sample matrices from
        // the same shutter pair. Exact for constant-velocity rigid motion below 180 degrees per
        // shutter interval — faster motion must supply per-sample transforms directly.
        RL_TOOLS_FUNCTION_PLACEMENT inline void slerp_transform(const float a[12], const float b[12], float t, float out[12]){
            float qa[4], qb[4];
            quaternion_from_transform(a, qa);
            quaternion_from_transform(b, qb);
            float relative[4] = {
                 qb[0]*qa[0] + qb[1]*qa[1] + qb[2]*qa[2] + qb[3]*qa[3],
                -qb[0]*qa[1] + qb[1]*qa[0] - qb[2]*qa[3] + qb[3]*qa[2],
                -qb[0]*qa[2] + qb[1]*qa[3] + qb[2]*qa[0] - qb[3]*qa[1],
                -qb[0]*qa[3] - qb[1]*qa[2] + qb[2]*qa[1] + qb[3]*qa[0]
            };
            if(relative[0] < 0){
                for(int i = 0; i < 4; i++) relative[i] = -relative[i];
            }
            {
                const float norm = sqrtf(relative[0]*relative[0] + relative[1]*relative[1] + relative[2]*relative[2] + relative[3]*relative[3]);
                for(int i = 0; i < 4; i++) relative[i] /= norm;
            }
            const float sin_half_angle = sqrtf(relative[1]*relative[1] + relative[2]*relative[2] + relative[3]*relative[3]);
            const float translation_a[3] = {a[3], a[7], a[11]};
            float translation_a_relative[3];
            rotate_by_quaternion(relative, translation_a, translation_a_relative);
            const float displacement[3] = {b[3] - translation_a_relative[0], b[7] - translation_a_relative[1], b[11] - translation_a_relative[2]};
            float relative_t[4];
            float translation_relative_t[3];
            if(sin_half_angle > 1e-3f){
                const float half_angle = atan2f(sin_half_angle, relative[0]);
                const float axis[3] = {relative[1] / sin_half_angle, relative[2] / sin_half_angle, relative[3] / sin_half_angle};
                const float sin_t_half = sinf(t * half_angle);
                relative_t[0] = cosf(t * half_angle);
                relative_t[1] = axis[0] * sin_t_half;
                relative_t[2] = axis[1] * sin_t_half;
                relative_t[3] = axis[2] * sin_t_half;
                const float displacement_along_axis = displacement[0]*axis[0] + displacement[1]*axis[1] + displacement[2]*axis[2];
                const float perpendicular[3] = {displacement[0] - displacement_along_axis*axis[0], displacement[1] - displacement_along_axis*axis[1], displacement[2] - displacement_along_axis*axis[2]};
                const float axis_cross_perpendicular[3] = {
                    axis[1]*perpendicular[2] - axis[2]*perpendicular[1],
                    axis[2]*perpendicular[0] - axis[0]*perpendicular[2],
                    axis[0]*perpendicular[1] - axis[1]*perpendicular[0]
                };
                const float cot_half_angle = relative[0] / sin_half_angle;
                float center[3], center_rotated[3];
                for(int i = 0; i < 3; i++){
                    center[i] = 0.5f * (perpendicular[i] + cot_half_angle * axis_cross_perpendicular[i]);
                }
                rotate_by_quaternion(relative_t, center, center_rotated);
                for(int i = 0; i < 3; i++){
                    translation_relative_t[i] = center[i] - center_rotated[i] + t * displacement_along_axis * axis[i];
                }
            }
            else{
                relative_t[0] = 1 + t * (relative[0] - 1);
                relative_t[1] = t * relative[1];
                relative_t[2] = t * relative[2];
                relative_t[3] = t * relative[3];
                const float norm = sqrtf(relative_t[0]*relative_t[0] + relative_t[1]*relative_t[1] + relative_t[2]*relative_t[2] + relative_t[3]*relative_t[3]);
                for(int i = 0; i < 4; i++) relative_t[i] /= norm;
                for(int i = 0; i < 3; i++){
                    translation_relative_t[i] = t * displacement[i];
                }
            }
            float q[4] = {
                relative_t[0]*qa[0] - relative_t[1]*qa[1] - relative_t[2]*qa[2] - relative_t[3]*qa[3],
                relative_t[0]*qa[1] + relative_t[1]*qa[0] + relative_t[2]*qa[3] - relative_t[3]*qa[2],
                relative_t[0]*qa[2] - relative_t[1]*qa[3] + relative_t[2]*qa[0] + relative_t[3]*qa[1],
                relative_t[0]*qa[3] + relative_t[1]*qa[2] - relative_t[2]*qa[1] + relative_t[3]*qa[0]
            };
            const float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            for(int i = 0; i < 4; i++) q[i] /= norm;
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            out[0] = 1 - 2*(y*y + z*z); out[1] = 2*(x*y - w*z);     out[2]  = 2*(x*z + w*y);
            out[4] = 2*(x*y + w*z);     out[5] = 1 - 2*(x*x + z*z); out[6]  = 2*(y*z - w*x);
            out[8] = 2*(x*z - w*y);     out[9] = 2*(y*z + w*x);     out[10] = 1 - 2*(x*x + y*y);
            float translation_a_rotated[3];
            rotate_by_quaternion(relative_t, translation_a, translation_a_rotated);
            out[3]  = translation_a_rotated[0] + translation_relative_t[0];
            out[7]  = translation_a_rotated[1] + translation_relative_t[1];
            out[11] = translation_a_rotated[2] + translation_relative_t[2];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
