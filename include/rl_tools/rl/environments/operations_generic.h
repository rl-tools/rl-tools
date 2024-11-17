#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_H
//#include "pendulum/operations_generic.h"
#include "environments.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename ENV, typename PARAMS>
    void init(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI){};
    template <typename DEVICE, typename ENV, typename PARAMS>
    void render(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI){};
    template <typename DEVICE, typename ENV, typename PARAMS, typename STATE>
    void set_state(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI, STATE&){};
    template <typename DEVICE, typename ENV, typename PARAMS, typename STATE, typename ACTION>
    void set_state(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI, STATE&, ACTION&){};
    template <typename DEVICE, typename ENV, typename PARAMS, typename ACTION>
    void set_action(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI, ACTION&){};
    template <typename DEVICE, typename ENV, typename PARAMS, typename STATE>
    void set_truncated(DEVICE&, ENV&, PARAMS&, rl::environments::DummyUI, STATE&){};
    template <typename DEVICE, typename ENV>
    auto get_ui(DEVICE&, ENV&){return "";}
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif