#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_OPERATIONS_CPU_H

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE>
    constexpr rl::environments::hyperdrone::controllers::Bounds<typename SPEC::T> command_bounds(const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, typename SPEC::TI index){
        return {(typename SPEC::T)-1, (typename SPEC::T)1};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
