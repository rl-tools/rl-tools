#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/rl/algorithms/td3/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/algorithms/td3/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>
#endif
#endif
