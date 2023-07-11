// ------------ Groups 1 ------------
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_mkl/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_MKL<DEV_SPEC>;
}
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_accelerate/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_ACCELERATE<DEV_SPEC>;
}
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_openblas/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU_OPENBLAS<DEV_SPEC>;
}
#else
#include <backprop_tools/operations/cpu/group_1.h>
namespace backprop_tools{
    template <typename DEV_SPEC>
    using DEVICE_FACTORY = backprop_tools::devices::CPU<DEV_SPEC>;
}
#endif
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
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_2.h>
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_accelerate/group_2.h>
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_openblas/group_2.h>
#else
#include <backprop_tools/operations/cpu/group_2.h>
#endif
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/operations/cuda/group_2.h>
#endif
// ------------ Groups 3 ------------
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_3.h>
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_accelerate/group_3.h>
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/operations/cpu_openblas/group_3.h>
#else
#include <backprop_tools/operations/cpu/group_3.h>
#endif
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/operations/cuda/group_3.h>
#endif
