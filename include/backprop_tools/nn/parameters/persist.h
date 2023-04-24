#ifndef BACKPROP_TOOLS_NN_PARAMETERS_PERSIST_H
#define BACKPROP_TOOLS_NN_PARAMETERS_PERSIST_H

#include <backprop_tools/nn/parameters/parameters.h>

#include <highfive/H5Group.hpp>
namespace backprop_tools{
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
    void load(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, parameter.parameters, group, "parameters");
    }
    template<typename DEVICE, typename CONTAINER>
    void load(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& parameter, HighFive::Group group) {
        load(device, (nn::parameters::Plain::instance<CONTAINER>&)parameter, group);
        load(device, parameter.gradient, group, "gradient");
    }
}
#endif
