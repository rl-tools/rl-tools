#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_CPU_MKL_GROUP_1_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_OPERATIONS_CPU_MKL_GROUP_1_H
#if defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_CPU_MKL_GROUP_1)
    #define BACKPROP_TOOLS_OPERATIONS_CPU_MKL_GROUP_1
    #include <mkl.h>
    #include "../../devices/cpu_mkl.h"
    #include "../../utils/assert/declarations_cpu.h"
    #include "../../math/operations_cpu.h"
    #include "../../random/operations_cpu.h"
    #include "../../logging/operations_cpu.h"
#else
    #error "Group 1 already imported"
#endif
#endif
