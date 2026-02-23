#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_LAMB_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_LAMB_OPERATIONS_GENERIC_H

#include "lamb.h"
#include "../adam/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::optimizers::Lamb<SPEC>& optimizer){
        malloc(device, static_cast<nn::optimizers::Adam<SPEC>&>(optimizer));
        malloc(device, optimizer.lamb_parameters);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::optimizers::Lamb<SPEC>& optimizer){
        free(device, static_cast<nn::optimizers::Adam<SPEC>&>(optimizer));
        free(device, optimizer.lamb_parameters);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, nn::optimizers::Lamb<SPEC>& optimizer){
        init(device, static_cast<nn::optimizers::Adam<SPEC>&>(optimizer));
        typename nn::optimizers::Lamb<SPEC>::LAMB_PARAMETERS lamb_params = {
            SPEC::LAMB_DEFAULT_PARAMETERS::UPPER_BOUND_TRUST_RATIO,
            SPEC::LAMB_DEFAULT_PARAMETERS::LOWER_BOUND_TRUST_RATIO,
        };
        set(device, optimizer.lamb_parameters, lamb_params, 0);
    }
    template<typename DEVICE, typename SPEC, typename MODEL>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_optimizer_state(DEVICE& device, nn::optimizers::Lamb<SPEC>& optimizer, MODEL& model){
        set(device, optimizer.age, 1, 0);
        _reset_optimizer_state(device, model, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename MODEL>
    RL_TOOLS_FUNCTION_PLACEMENT void step(DEVICE& device, nn::optimizers::Lamb<SPEC>& optimizer, MODEL& model){
        _step(device, static_cast<nn::optimizers::Adam<SPEC>&>(optimizer));
        update(device, model, optimizer);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::optimizers::Lamb<SOURCE_SPEC>& source, nn::optimizers::Lamb<TARGET_SPEC>& target){
        copy(source_device, target_device, static_cast<const nn::optimizers::Adam<SOURCE_SPEC>&>(source), static_cast<nn::optimizers::Adam<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.lamb_parameters, target.lamb_parameters);
    }
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T abs_diff(DEVICE& device, const nn::optimizers::lamb::Parameters<T>& p1, const nn::optimizers::lamb::Parameters<T>& p2){
        T acc = 0;
        acc += math::abs(device.math, p1.upper_bound_trust_ratio - p2.upper_bound_trust_ratio);
        acc += math::abs(device.math, p1.lower_bound_trust_ratio - p2.lower_bound_trust_ratio);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::T abs_diff(DEVICE& device, nn::optimizers::Lamb<SPEC_1>& o1, nn::optimizers::Lamb<SPEC_2>& o2){
        using T = typename SPEC_1::T;
        T acc = abs_diff(device, static_cast<nn::optimizers::Adam<SPEC_1>&>(o1), static_cast<nn::optimizers::Adam<SPEC_2>&>(o2));
        acc += abs_diff(device, get(device, o1.lamb_parameters, 0), get(device, o2.lamb_parameters, 0));
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
