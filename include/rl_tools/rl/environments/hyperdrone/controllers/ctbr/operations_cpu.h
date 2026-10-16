#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_CTBR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_CTBR_OPERATIONS_CPU_H

#include "operations_generic.h"
#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename SPEC>
    constexpr rl::environments::hyperdrone::controllers::Bounds<typename SPEC::T> command_bounds(const rl::environments::hyperdrone::controllers::CTBR<SPEC>&, typename SPEC::TI index){
        using T = typename SPEC::T;
        constexpr rl::environments::l2f::parameters::IdealCTBR<T> limits{};
        return index == 0 ? rl::environments::hyperdrone::controllers::Bounds<T>{0, std::numeric_limits<T>::infinity()} : rl::environments::hyperdrone::controllers::Bounds<T>{-limits.rate_limit[index - 1], limits.rate_limit[index - 1]};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
