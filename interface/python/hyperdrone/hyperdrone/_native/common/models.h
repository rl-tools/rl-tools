#pragma once
#include <rl_tools/rl/environments/l2f/parameters/registry.h>

#include <string>

namespace hyperdrone {
    // runtime lookup of the l2f dynamics registry by the names in hyperdrone.dynamics.MODELS
    template <typename PARAMETERS_SPEC, typename DYNAMICS>
    bool model_by_name(const std::string& name, DYNAMICS& out){
        namespace pd = rl_tools::rl::environments::l2f::parameters::dynamics;
        if(name == "crazyflie"){ out = pd::registry<pd::REGISTRY::crazyflie, PARAMETERS_SPEC>; return true; }
        if(name == "crazyflie_openmv"){ out = pd::registry<pd::REGISTRY::crazyflie_openmv, PARAMETERS_SPEC>; return true; }
        if(name == "mrs"){ out = pd::registry<pd::REGISTRY::mrs, PARAMETERS_SPEC>; return true; }
        if(name == "x500"){ out = pd::registry<pd::REGISTRY::x500, PARAMETERS_SPEC>; return true; }
        if(name == "x500_real"){ out = pd::registry<pd::REGISTRY::x500_real, PARAMETERS_SPEC>; return true; }
        if(name == "x500_sim"){ out = pd::registry<pd::REGISTRY::x500_sim, PARAMETERS_SPEC>; return true; }
        if(name == "arpl"){ out = pd::registry<pd::REGISTRY::arpl, PARAMETERS_SPEC>; return true; }
        if(name == "fs_base"){ out = pd::registry<pd::REGISTRY::fs_base, PARAMETERS_SPEC>; return true; }
        if(name == "flightmare"){ out = pd::registry<pd::REGISTRY::flightmare, PARAMETERS_SPEC>; return true; }
        if(name == "soft"){ out = pd::registry<pd::REGISTRY::soft, PARAMETERS_SPEC>; return true; }
        if(name == "soft_rigid"){ out = pd::registry<pd::REGISTRY::soft_rigid, PARAMETERS_SPEC>; return true; }
        if(name == "crazyflie_brushless"){ out = pd::registry<pd::REGISTRY::crazyflie_brushless, PARAMETERS_SPEC>; return true; }
        if(name == "savagebee_pusher"){ out = pd::registry<pd::REGISTRY::savagebee_pusher, PARAMETERS_SPEC>; return true; }
        return false;
    }
}
