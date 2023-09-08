#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_CPU_TENSORBOARD_GROUP_1_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_OPERATIONS_CPU_TENSORBOARD_GROUP_1_H
#if defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_CPU_TENSORBOARD_GROUP_1)
    #define BACKPROP_TOOLS_OPERATIONS_CPU_TENSORBOARD_GROUP_1
    #include "../../devices/cpu_tensorboard.h"
    #include "../../utils/assert/declarations_cpu.h"
    #include "../../math/operations_cpu.h"
    #include "../../random/operations_cpu.h"
    #include "../../logging/operations_cpu_tensorboard.h"
#else
    #error "Group 1 already imported"
#endif
#endif
