#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_GENERIC_H

#include "adam.h"
#include "../../../nn/layers/dense/layer.h"
#include "../../../nn/parameters/operations_generic.h"
#include "../../../utils/polyak/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename MODEL>
    void reset_optimizer_state(DEVICE& device, nn::optimizers::Adam<SPEC>& optimizer, MODEL& model) {
        optimizer.age = 1;
        _reset_optimizer_state(device, model, optimizer);
    }

    template<typename DEVICE, typename SPEC, typename MODEL>
    void step(DEVICE& device, nn::optimizers::Adam<SPEC>& optimizer, MODEL& model) {
        using T = typename SPEC::T;
        optimizer.first_order_moment_bias_correction  = 1/(1 - math::pow(device.math, optimizer.parameters.beta_1, (T)optimizer.age));
        optimizer.second_order_moment_bias_correction = 1/(1 - math::pow(device.math, optimizer.parameters.beta_2, (T)optimizer.age));
        optimizer.age += 1;
        update(device, model, optimizer);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::optimizers::Adam<SOURCE_SPEC>& source, nn::optimizers::Adam<TARGET_SPEC>& target){
        target.parameters = source.parameters;
        target.age = source.age;
    }

    template<typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const nn::parameters::Adam::instance<SPEC>& p){
        bool param_nan = is_nan(device, (nn::parameters::Gradient::instance<SPEC>&) p);
        return param_nan || is_nan(device, p.gradient_first_order_moment) || is_nan(device, p.gradient_second_order_moment);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
