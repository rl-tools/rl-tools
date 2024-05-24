#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_CPU_H

#include "pendulum.h"
#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    std::string serialize_json(DEVICE&, rl::environments::Pendulum<SPEC>& env, typename rl::environments::Pendulum<SPEC>::State& state){
        std::string json = "{";
        json += "\"theta\":" + std::to_string(state.theta) + ",";
        json += "\"theta_dot\":" + std::to_string(state.theta_dot);
        json += "}";
        return json;
    }
}

#endif
