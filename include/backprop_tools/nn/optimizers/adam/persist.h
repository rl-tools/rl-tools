#ifndef BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_PERSIST_H
#define BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_PERSIST_H

#include <highfive/H5Group.hpp>

#include "adam.h"

namespace backprop_tools{
    template<typename DEVICE, typename CONTAINER>
    void save(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& parameter, HighFive::Group group) {
        save(device, (nn::parameters::Gradient::instance<CONTAINER>&)parameter, group);
        save(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment");
        save(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment");
    }
    template<typename DEVICE, typename CONTAINER>
    void load(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, (nn::parameters::Gradient::instance<CONTAINER>&)parameter, group);
        load(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment", true);
        load(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment", true);
    }
}

#endif
