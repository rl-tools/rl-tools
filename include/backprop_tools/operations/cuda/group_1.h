#ifndef BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1
    #define BACKPROP_TOOLS_OPERATIONS_CUDA_GROUP_1
    #define BACKPROP_TOOLS_FUNCTION_PLACEMENT __device__ __host__
    #define BACKPROP_TOOLS_DEVICES_CUDA_CEIL(A, B) (A / B + (A % B == 0 ? 0 : 1))

    #include <backprop_tools/devices/cuda.h>
    #include <backprop_tools/math/operations_cuda.h>
    #include <backprop_tools/random/operations_cuda.h>
    #include <backprop_tools/logging/operations_cuda.h>
#else
    #error "Group 1 already imported"
#endif
