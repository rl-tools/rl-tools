#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_PARAMETERS_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_PARAMETERS_PERSIST_H

#include "../../nn/parameters/parameters.h"

#include <highfive/H5Group.hpp>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
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
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
