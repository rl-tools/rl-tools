#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H
//#include "pendulum/operations_generic.h"


BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template <typename DEVICE, typename ENV>
    void init(DEVICE&, ENV&, bool){};
    template <typename DEVICE, typename ENV>
    void render(DEVICE&, ENV&, bool){};
    template <typename DEVICE, typename ENV, typename STATE>
    void set_state(DEVICE&, ENV&, bool, STATE&){};
    template <typename DEVICE, typename ENV, typename ACTION>
    void set_action(DEVICE&, ENV&, bool, ACTION&){};
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif