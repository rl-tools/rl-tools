#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REGISTRY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REGISTRY_H

#include "dynamics/crazyflie.h"
#include "dynamics/mrs.h"
#include "dynamics/race.h"
#include "dynamics/x500_real.h"
#include "dynamics/x500_sim.h"
#include "dynamics/fs.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters{
    namespace dynamics{
        enum class REGISTRY{
            crazyflie,
            mrs,
            x500_real,
            x500_sim,
            fs_base
        };
        template <typename SPEC>
        constexpr auto registry = [](){
            if constexpr (SPEC::MODEL == REGISTRY::crazyflie){
                return dynamics::crazy_flie<SPEC>;
            }else if constexpr (SPEC::MODEL == REGISTRY::mrs){
                return dynamics::mrs<SPEC>;
            }else if constexpr (SPEC::MODEL == REGISTRY::x500_real){
                return dynamics::x500::real<SPEC>;
            }else if constexpr (SPEC::MODEL == REGISTRY::x500_sim){
                return dynamics::x500::sim<SPEC>;
            }else if constexpr (SPEC::MODEL == REGISTRY::fs_base){
                return dynamics::fs::base<SPEC>;
            }else{
                static_assert(rl_tools::utils::typing::dependent_false<SPEC>, "Unknown model");
            }
        }();

        template <auto THING>
        struct Dependent{};

        template <REGISTRY MODEL>
        constexpr auto registry_name = [](){
            if constexpr (MODEL == REGISTRY::crazyflie){
                return "crazyflie";
            }else if constexpr (MODEL == REGISTRY::mrs){
                return "mrs";
            }else if constexpr (MODEL == REGISTRY::x500_real){
                return "x500_real";
            }else if constexpr (MODEL == REGISTRY::x500_sim){
                return "x500_sim";
            }else if constexpr (MODEL == REGISTRY::fs_base){
                return "fs_base";
            }else{
                static_assert(rl_tools::utils::typing::dependent_false<Dependent<MODEL>>, "Unknown model");
            }
        }();
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
