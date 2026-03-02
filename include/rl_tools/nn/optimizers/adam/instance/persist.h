#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_PERSIST_H

#include "../adam.h"
#include "../../../parameters/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename CONTAINER, typename GROUP>
    void save(DEVICE& device, nn::parameters::Adam::Instance<CONTAINER>& parameter, GROUP& group) {
        save(device, (nn::parameters::Gradient::Instance<CONTAINER>&)parameter, group);
        save(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment");
        save(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment");
        if constexpr(nn::parameters::Adam::Instance<CONTAINER>::USE_MASTER_PARAMETERS){
            save(device, parameter.master_parameters, group, "master_parameters");
        }
    }
    template<typename DEVICE, typename CONTAINER, typename GROUP>
    bool load(DEVICE& device, nn::parameters::Adam::Instance<CONTAINER>& parameter, GROUP& group) {
        bool success = load(device, (nn::parameters::Gradient::Instance<CONTAINER>&)parameter, group);
        success &= load(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment");
        success &= load(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment");
        if constexpr(nn::parameters::Adam::Instance<CONTAINER>::USE_MASTER_PARAMETERS){
            if(group_exists(device, group, "master_parameters")){
                success &= load(device, parameter.master_parameters, group, "master_parameters");
            }
            else{
                // Backward compatibility with checkpoints generated before master parameters existed.
                copy(device, device, parameter.parameters, parameter.master_parameters);
            }
        }
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
