// ------------ Groups 1 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_1.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_1.h>
namespace layer_in_c{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = layer_in_c::devices::CPU_MKL<DEV_SPEC>;
}
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_1.h>
namespace layer_in_c{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = layer_in_c::devices::CPU_ACCELERATE<DEV_SPEC>;
}
#else
#include <layer_in_c/operations/cpu/group_1.h>
namespace layer_in_c{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = layer_in_c::devices::CPU<DEV_SPEC>;
}
#endif
#endif
#if defined(LAYER_IN_C_BACKEND_ENABLE_CUDA) && defined(LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <layer_in_c/operations/cuda/group_1.h>
namespace layer_in_c {
    template<typename DEV_SPEC>
    using DEVICE_FACTORY_GPU = layer_in_c::devices::CUDA<DEV_SPEC>;
}
#endif
// ------------ Groups 2 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_2.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_2.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_2.h>
#else
#include <layer_in_c/operations/cpu/group_2.h>
#endif
#endif
#if defined(LAYER_IN_C_BACKEND_ENABLE_CUDA) && defined(LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <layer_in_c/operations/cuda/group_2.h>
#endif
// ------------ Groups 3 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_3.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_3.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_3.h>
#else
#include <layer_in_c/operations/cpu/group_3.h>
#endif
#endif
#if defined(LAYER_IN_C_BACKEND_ENABLE_CUDA) && defined(LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <layer_in_c/operations/cuda/group_3.h>
#endif
