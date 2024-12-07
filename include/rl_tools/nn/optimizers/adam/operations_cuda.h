#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_OPERATIONS_CUDA_H

#include "adam.h"
#include "../../../nn/layers/dense/layer.h"
#include "../../../nn/parameters/operations_generic.h"
#include "../../../utils/polyak/operations_generic.h"

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::optimizers::adam::kernels{
        template<typename DEVICE, typename SPEC>
        __global__
        void reset_optimizer_state(DEVICE device, nn::optimizers::Adam<SPEC> optimizer) {
            set(device, optimizer.age, 1, 0);
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename MODEL>
    void reset_optimizer_state(devices::CUDA<DEV_SPEC>& device, nn::optimizers::Adam<SPEC>& optimizer, MODEL& model) {
        dim3 activation_grid(1);
        dim3 activation_block(1);
        nn::optimizers::adam::kernels::reset_optimizer_state<<<activation_grid, activation_block, 0, device.stream>>>(device, optimizer);
        check_status(device);
        _reset_optimizer_state(device, model, optimizer);
    }
    namespace nn::optimizers::adam::kernels{
        template<typename DEVICE, typename SPEC>
        __global__
        void step(DEVICE device, nn::optimizers::Adam<SPEC> optimizer) {
            _step(device, optimizer);
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename MODEL>
    void step(devices::CUDA<DEV_SPEC>& device, nn::optimizers::Adam<SPEC>& optimizer, MODEL& model){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        dim3 activation_grid(1);
        dim3 activation_block(1);
        nn::optimizers::adam::kernels::step<<<activation_grid, activation_block, 0, device.stream>>>(device, optimizer);
        check_status(device);
        update(device, model, optimizer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
