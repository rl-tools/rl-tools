#ifndef LAYER_IN_C_OPERATIONS_CUDA_H
#define LAYER_IN_C_OPERATIONS_CUDA_H

namespace layer_in_c{
    constexpr bool compile_time_redefinition_detector = true; // When importing different devices don't import the full header. The operations need to be imporeted interleaved (e.g. include cpu group 1 -> include cuda group 1 -> include cpu group 2 -> include cuda group 2 -> ...)
}

// Group 1
#include <layer_in_c/devices/cuda.h>
#include <layer_in_c/math/operations_cuda.h>
#include <layer_in_c/random/operations_cuda.h>
#include <layer_in_c/logging/operations_cuda.h>

// Group 2: depends on logging
#include <layer_in_c/utils/assert/operations_cuda.h>
// Group 3: dependent on assert
#include <layer_in_c/containers/operations_generic.h>

#endif