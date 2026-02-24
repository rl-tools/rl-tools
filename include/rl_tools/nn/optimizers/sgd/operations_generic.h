#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_OPERATIONS_GENERIC_H

#include "sgd.h"
#include "../../../nn/parameters/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::optimizers::SGD<SPEC>& optimizer){
        malloc(device, optimizer.parameters);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::optimizers::SGD<SPEC>& optimizer){
        free(device, optimizer.parameters);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, nn::optimizers::SGD<SPEC>& optimizer){
        typename nn::optimizers::SGD<SPEC>::PARAMETERS parameters = {
            SPEC::DEFAULT_PARAMETERS::LEARNING_RATE,
            SPEC::DEFAULT_PARAMETERS::MOMENTUM,
            SPEC::DEFAULT_PARAMETERS::WEIGHT_DECAY,
            SPEC::DEFAULT_PARAMETERS::NESTEROV
        };
        set(device, optimizer.parameters, parameters, 0);
    }
    template<typename DEVICE, typename SPEC, typename MODEL>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_optimizer_state(DEVICE& device, nn::optimizers::SGD<SPEC>& optimizer, MODEL& model) {
        _reset_optimizer_state(device, model, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename MODEL>
    RL_TOOLS_FUNCTION_PLACEMENT void step(DEVICE& device, nn::optimizers::SGD<SPEC>& optimizer, MODEL& model) {
        update(device, model, optimizer);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::optimizers::SGD<SOURCE_SPEC>& source, nn::optimizers::SGD<TARGET_SPEC>& target){
        copy(source_device, target_device, source.parameters, target.parameters);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
