#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

/*
 * Generic operations can run on the CPU or GPU depending on the setting of the FUNCTION_PLACEMENT macro.
 */
#include "layers/operations_generic.h"

namespace layer_in_c {
    using namespace layer_in_c::device::generic;
}