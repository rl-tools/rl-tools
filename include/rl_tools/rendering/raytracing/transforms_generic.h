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

        // rotation slerp (shortest arc) + translation lerp between two rigid 3x4 transforms; the
        // single interpolant behind set_transform_pair and expand_motion_transforms, so every
        // consumer derives identical per-sample matrices from the same shutter pair. Exact for
        // single-axis rotations below 180 degrees per shutter interval — faster motion must
        // supply per-sample transforms directly.
        RL_TOOLS_FUNCTION_PLACEMENT inline void slerp_transform(const float a[12], const float b[12], float t, float out[12]){
            float qa[4], qb[4];
            quaternion_from_transform(a, qa);
            quaternion_from_transform(b, qb);
            float dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3];
            if(dot < 0){
                for(int i = 0; i < 4; i++) qb[i] = -qb[i];
                dot = -dot;
            }
            float wa, wb;
            if(dot > 0.9995f){
                wa = 1 - t;
                wb = t;
            }
            else{
                const float theta = acosf(dot);
                const float sin_theta = sinf(theta);
                wa = sinf((1 - t) * theta) / sin_theta;
                wb = sinf(t * theta) / sin_theta;
            }
            float q[4] = {wa*qa[0] + wb*qb[0], wa*qa[1] + wb*qb[1], wa*qa[2] + wb*qb[2], wa*qa[3] + wb*qb[3]};
            const float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            for(int i = 0; i < 4; i++) q[i] /= norm;
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            out[0] = 1 - 2*(y*y + z*z); out[1] = 2*(x*y - w*z);     out[2]  = 2*(x*z + w*y);
            out[4] = 2*(x*y + w*z);     out[5] = 1 - 2*(x*x + z*z); out[6]  = 2*(y*z - w*x);
            out[8] = 2*(x*z - w*y);     out[9] = 2*(y*z + w*x);     out[10] = 1 - 2*(x*x + y*y);
            out[3]  = (1 - t) * a[3]  + t * b[3];
            out[7]  = (1 - t) * a[7]  + t * b[7];
            out[11] = (1 - t) * a[11] + t * b[11];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
