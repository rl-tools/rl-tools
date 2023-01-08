#ifndef LAYER_IN_C_CONTEXT
#define LAYER_IN_C_CONTEXT

#include <cstddef>
namespace layer_in_c{
    using index_t = size_t;
}

#include <layer_in_c/math/operations_cpu.h>
#include <layer_in_c/utils/random/operations_cpu.h>
#include <layer_in_c/logging/operations_cpu.h>
#include <layer_in_c/utils/assert/operations_cpu.h>
#else
#pragma message "Can't include CPU context, some context has already been included"
#endif
