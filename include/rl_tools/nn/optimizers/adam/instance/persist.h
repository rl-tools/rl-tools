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
    }
    template<typename DEVICE, typename CONTAINER, typename GROUP>
    void load(DEVICE& device, nn::parameters::Adam::Instance<CONTAINER>& parameter, GROUP& group) {
        load(device, (nn::parameters::Gradient::Instance<CONTAINER>&)parameter, group);
        load(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment");
        load(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
