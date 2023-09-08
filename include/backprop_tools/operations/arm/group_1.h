#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_ARM_GROUP_1_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_OPERATIONS_ARM_GROUP_1_H

#if defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_OPERATIONS_ARM_GROUP_1)
    #define BACKPROP_TOOLS_OPERATIONS_ARM_GROUP_1
    #include "../../devices/arm.h"
//    #include "../../utils/assert/declarations_arm.h"
    #include "../../math/operations_arm.h"
    #include "../../random/operations_arm.h"
//    #include "../../logging/operations_arm.h"
#else
    #error "Group 1 already imported"
#endif

#endif