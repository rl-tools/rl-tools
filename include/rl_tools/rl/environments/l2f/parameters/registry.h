#include "dynamics/crazy_flie.h"
#include "dynamics/mrs.h"
#include "dynamics/race.h"
#include "dynamics/x500_real.h"
#include "dynamics/x500_sim.h"
#include "dynamics/fs.h"
namespace rl_tools::rl::environments::multirotor::parameters{
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
                static_assert(utils::typing::dependent_false<SPEC>, "Unknown model");
            }
        }();

        template <typename SPEC>
        constexpr auto registry_name = [](){
            if constexpr (SPEC::MODEL == REGISTRY::crazyflie){
                return "crazyflie";
            }else if constexpr (SPEC::MODEL == REGISTRY::mrs){
                return "mrs";
            }else if constexpr (SPEC::MODEL == REGISTRY::x500_real){
                return "x500_real";
            }else if constexpr (SPEC::MODEL == REGISTRY::x500_sim){
                return "x500_sim";
            }else if constexpr (SPEC::MODEL == REGISTRY::fs_base){
                return "fs_base";
            }else{
                static_assert(utils::typing::dependent_false<SPEC>, "Unknown model");
            }
        }();
    }
}
