#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/nn/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/nn/operations_cpu_accelerate.h>
#else
#include <backprop_tools/nn/operations_generic.h>
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include <backprop_tools/nn/operations_cuda.h>
#endif
