// ------------ Groups 1 ------------
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
#endif
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_MKL<DEV_SPEC>;
}
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_ACCELERATE<DEV_SPEC>;
}
#else
#include <backprop_tools/operations/cpu/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU<DEV_SPEC>;
}
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/operations/cuda/group_1.h>
namespace backprop_tools {
    template<typename DEV_SPEC>
    using DEVICE_FACTORY_GPU = backprop_tools::devices::CUDA<DEV_SPEC>;
}
#endif
// ------------ Groups 2 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_2.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_2.h>
#else
#include <backprop_tools/operations/cpu/group_2.h>
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/operations/cuda/group_2.h>
#endif
// ------------ Groups 3 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_3.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_3.h>
#else
#include <backprop_tools/operations/cpu/group_3.h>
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/operations/cuda/group_3.h>
#endif
