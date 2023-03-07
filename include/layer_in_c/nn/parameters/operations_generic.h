#ifndef LAYER_IN_C_NN_PARAMETERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_PARAMETERS_OPERATIONS_GENERIC_H

#include "parameters.h"

namespace layer_in_c{
    template <typename DEVICE, typename CONTAINER>
    void malloc(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& p){
        malloc(device, p.parameters);
    }
    template <typename DEVICE, typename CONTAINER>
    void free(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& p){
        free(device, p.parameters);
    }
    template <typename DEVICE, typename CONTAINER>
    void malloc(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& p){
        malloc(device, (nn::parameters::Plain::instance<CONTAINER>&) p);
        malloc(device, p.gradient);
    }
    template <typename DEVICE, typename CONTAINER>
    void free(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& p){
        free(device, (nn::parameters::Plain::instance<CONTAINER>&) p);
        free(device, p.gradient);
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn::parameters::Plain::instance<TARGET_SPEC>& target, const nn::parameters::Plain::instance<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.parameters, source.parameters);
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn::parameters::Gradient::instance<TARGET_SPEC>& target, const nn::parameters::Gradient::instance<SOURCE_SPEC>& source){
        copy(target_device, source_device, (nn::parameters::Plain::instance<TARGET_SPEC>&) target, (nn::parameters::Plain::instance<SOURCE_SPEC>&) source);
        copy(target_device, source_device, target.gradient, source.gradient);
    }
}
#endif
