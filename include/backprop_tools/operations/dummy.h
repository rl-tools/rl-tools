#include "../version.h"
#if !defined(BACKPROP_TOOLS_OPERATIONS_DUMMY_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_OPERATIONS_DUMMY_H

#include <backprop_tools/backprop_tools.h>
#ifndef BACKPROP_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    constexpr bool compile_time_redefinition_detector = true; // When importing different devices don't import the full header. The operations need to be imporeted interleaved (e.g. include cpu group 1 -> include cuda group 1 -> include cpu group 2 -> include cuda group 2 -> ...)
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif

#include "dummy/group_1.h"
#include "dummy/group_2.h"
#include "dummy/group_3.h"

#endif