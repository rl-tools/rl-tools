#ifndef LAYER_IN_C_NN_PARAMETERS_PERSIST_H
#define LAYER_IN_C_NN_PARAMETERS_PERSIST_H

#include <layer_in_c/nn/parameters/parameters.h>

#include <highfive/H5Group.hpp>
namespace layer_in_c{
    template<typename DEVICE, typename CONTAINER>
    void save(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& parameter, HighFive::Group group) {
        save(device, parameter.parameters, group, "parameters");
    }
    template<typename DEVICE, typename CONTAINER>
    void save(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& parameter, HighFive::Group group) {
        save(device, (nn::parameters::Plain::instance<CONTAINER>&)parameter, group);
        save(device, parameter.gradient, group, "gradient");
    }
    template<typename DEVICE, typename CONTAINER>
    void save(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& parameter, HighFive::Group group) {
        save(device, (nn::parameters::Gradient::instance<CONTAINER>&)parameter, group);
        save(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment");
        save(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment");
    }
    template<typename DEVICE, typename CONTAINER>
    void load(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, parameter.parameters, group, "parameters");
    }
    template<typename DEVICE, typename CONTAINER>
    void load(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, (nn::parameters::Plain::instance<CONTAINER>&)parameter, group);
        load(device, parameter.gradient, group, "gradient");
    }
    template<typename DEVICE, typename CONTAINER>
    void load(DEVICE& device, nn::parameters::Adam::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, (nn::parameters::Gradient::instance<CONTAINER>&)parameter, group);
        load(device, parameter.gradient_first_order_moment, group, "gradient_first_order_moment", true);
        load(device, parameter.gradient_second_order_moment, group, "gradient_second_order_moment", true);
    }
}
#endif
