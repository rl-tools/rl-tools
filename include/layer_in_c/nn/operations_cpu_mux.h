#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/nn/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/nn/operations_cpu_accelerate.h>
#else
#include <layer_in_c/nn/operations_generic.h>
#endif
#endif
