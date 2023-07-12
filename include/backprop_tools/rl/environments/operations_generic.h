#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H
#include "pendulum/operations_generic.h"


namespace backprop_tools{
    template <typename DEVICE, typename ENV>
    void render(DEVICE, ENV, bool){};
    template <typename DEVICE, typename ENV, typename STATE>
    void set_state(DEVICE, ENV, bool, STATE){};
    template <typename DEVICE, typename ENV, typename ACTION>
    void set_action(DEVICE, ENV, bool, ACTION){};
}
#endif