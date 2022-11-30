#ifndef LAYER_IN_C_NN_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_OPERATIONS_GENERIC_H
#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

/*
 * Generic operations can run on the CPU or GPU depending on the setting of the FUNCTION_PLACEMENT macro.
 */
namespace layer_in_c {
    namespace devices {
        struct Generic{};
    }
}
#include "layers/operations_generic.h"


#endif