#ifndef LAYER_IN_C_CONTEXT
#define LAYER_IN_C_CONTEXT

namespace layer_in_c{
    using index_t = unsigned;
}

#include <layer_in_c/math/operations_dummy.h>
#include <layer_in_c/utils/random/operations_dummy.h>
#include <layer_in_c/logging/operations_dummy.h>
#include <layer_in_c/utils/assert/operations_dummy.h>
#else
#pragma message "Can't include CPU context, some context has already been included"
#endif
