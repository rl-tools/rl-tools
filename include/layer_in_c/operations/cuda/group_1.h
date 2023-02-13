#ifndef LAYER_IN_C_OPERATIONS_CUDA_GROUP_1
    #define LAYER_IN_C_OPERATIONS_CUDA_GROUP_1
    #define LAYER_IN_C_FUNCTION_PLACEMENT __device__ __host__
    #define LAYER_IN_C_DEVICES_CUDA_CEIL(A, B) (A / B + (A % B == 0 ? 0 : 1))

    #include <layer_in_c/devices/cuda.h>
    #include <layer_in_c/math/operations_cuda.h>
    #include <layer_in_c/random/operations_cuda.h>
    #include <layer_in_c/logging/operations_cuda.h>
#else
    #error "Group 1 already imported"
#endif
