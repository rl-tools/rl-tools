#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_OPERATIONS_CUDA_H
#include "sgd.h"
#include "../../../nn/parameters/operations_generic.h"
#include "operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::optimizers::sgd::kernels{
        template<typename DEVICE, typename SPEC>
        __global__
        void init(DEVICE device, nn::optimizers::SGD<SPEC> optimizer) {
            typename nn::optimizers::SGD<SPEC>::PARAMETERS parameters = {
                SPEC::DEFAULT_PARAMETERS::LEARNING_RATE,
                SPEC::DEFAULT_PARAMETERS::MOMENTUM,
                SPEC::DEFAULT_PARAMETERS::WEIGHT_DECAY,
                SPEC::DEFAULT_PARAMETERS::NESTEROV
            };
            set(device, optimizer.parameters, parameters, 0);
        }
    }
    template<typename DEV_SPEC, typename SPEC>
    void init(devices::CUDA<DEV_SPEC>& device, nn::optimizers::SGD<SPEC>& optimizer) {
        dim3 grid(1);
        dim3 block(1);
        nn::optimizers::sgd::kernels::init<<<grid, block, 0, device.stream>>>(device, optimizer);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename MODEL>
    void reset_optimizer_state(devices::CUDA<DEV_SPEC>& device, nn::optimizers::SGD<SPEC>& optimizer, MODEL& model) {
        _reset_optimizer_state(device, model, optimizer);
    }
    template<typename DEV_SPEC, typename SPEC, typename MODEL>
    void step(devices::CUDA<DEV_SPEC>& device, nn::optimizers::SGD<SPEC>& optimizer, MODEL& model){
        update(device, model, optimizer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
