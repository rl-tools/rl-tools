#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_TRANSFORMS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_TRANSFORMS_H

#include "../rl_tools.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

// freestanding 3x4 row-major [R|t] transform math on the content model, shared by the datasets
// layer (bounds, instance welding) and the renderer (uploads, device producers)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rendering{
        RL_TOOLS_FUNCTION_PLACEMENT inline void compose_transforms(const float a[12], const float b[12], float out[12]){
            for(int row = 0; row < 3; row++){
                for(int column = 0; column < 3; column++){
                    out[row * 4 + column] = a[row * 4] * b[column] + a[row * 4 + 1] * b[4 + column] + a[row * 4 + 2] * b[8 + column];
                }
                out[row * 4 + 3] = a[row * 4] * b[3] + a[row * 4 + 1] * b[7] + a[row * 4 + 2] * b[11] + a[row * 4 + 3];
            }
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline bool transform_is_identity(const float transform[12]){
            const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
            for(int element = 0; element < 12; element++){
                if(transform[element] != identity[element]){
                    return false;
                }
            }
            return true;
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline void transform_point(const float transform[12], const float point[3], float out[3]){
            for(int row = 0; row < 3; row++){
                out[row] = transform[row * 4 + 0] * point[0] + transform[row * 4 + 1] * point[1] + transform[row * 4 + 2] * point[2] + transform[row * 4 + 3];
            }
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline void transform_vector(const float transform[12], const float vector[3], float out[3]){
            for(int row = 0; row < 3; row++){
                out[row] = transform[row * 4 + 0] * vector[0] + transform[row * 4 + 1] * vector[1] + transform[row * 4 + 2] * vector[2];
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
