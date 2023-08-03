#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/rl/algorithms/td3/operations_cpu_mkl.h>
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include <backprop_tools/rl/algorithms/td3/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>
#endif
#endif
