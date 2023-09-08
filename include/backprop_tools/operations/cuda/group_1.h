#include "../../version.h"
#if !defined(BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1_H
#ifndef BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1
    #define BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1
    #define BACKPROP_TOOLS_FUNCTION_PLACEMENT __device__ __host__
    #define BACKPROP_TOOLS_DEVICES_CUDA_CEIL(A, B) (A / B + (A % B == 0 ? 0 : 1))

    #include "../../devices/cuda.h"
    #include "../../math/operations_cuda.h"
    #include "../../random/operations_cuda.h"
    #include "../../logging/operations_cuda.h"
#else
    #error "Group 1 already imported"
#endif
#endif
