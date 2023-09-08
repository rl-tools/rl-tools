#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include "../nn/operations_cpu_mkl.h"
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include "../nn/operations_cpu_accelerate.h"
#else
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
#include "../nn/operations_cpu_openblas.h"
#else
#include "../nn/operations_generic.h"
#endif
#endif
#endif
#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA) && defined(BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include "../nn/operations_cuda.h"
#endif
