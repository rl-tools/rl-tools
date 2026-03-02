#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_OPERATIONS_CUDA_H

#include "../sgd.h"
#include "operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace nn::optimizers::sgd::cuda {
        template<typename DEV_SPEC, typename PARAMETER_SPEC, typename SPEC>
        __global__
        void update_kernel(devices::CUDA<DEV_SPEC>& device, nn::parameters::SGD::Instance<PARAMETER_SPEC> parameter, nn::optimizers::SGD<SPEC> optimizer) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;

            const auto& optimizer_parameters = get_ref(device, optimizer.parameters, 0);

            using PARAMETER_INSTANCE = nn::parameters::SGD::Instance<PARAMETER_SPEC>;
            static constexpr bool USE_MASTER_PARAMETERS = PARAMETER_INSTANCE::USE_MASTER_PARAMETERS;
            using T_VELOCITY = typename PARAMETER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
            using T_PARAMETER = typename decltype(parameter.parameters)::T;
            using T_MASTER_PARAMETER = typename PARAMETER_INSTANCE::T_MASTER_PARAMETER;
            auto params = matrix_view(device, parameter.parameters);
            auto grad = matrix_view(device, parameter.gradient);
            auto vel = matrix_view(device, parameter.velocity);
            constexpr TI ROWS = decltype(params)::ROWS;
            constexpr TI COLS = decltype(params)::COLS;

            TI col_i = blockIdx.x * blockDim.x + threadIdx.x;
            TI row_i = blockIdx.y * blockDim.y + threadIdx.y;
            if(col_i < COLS && row_i < ROWS){
                T_VELOCITY lr = (T_VELOCITY)optimizer_parameters.learning_rate;
                T_VELOCITY mom = (T_VELOCITY)optimizer_parameters.momentum;
                T_VELOCITY wd = (T_VELOCITY)optimizer_parameters.weight_decay;
                T_VELOCITY value;
                if constexpr(USE_MASTER_PARAMETERS){
                    auto master_params = matrix_view(device, parameter.master_parameters);
                    value = (T_VELOCITY)get(master_params, row_i, col_i);
                }
                else{
                    value = (T_VELOCITY)get(params, row_i, col_i);
                }
                T_VELOCITY g = get(grad, row_i, col_i);
                if constexpr(SPEC::ENABLE_WEIGHT_DECAY){
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                        g += value * wd;
                    }
                }
                T_VELOCITY v = mom * get(vel, row_i, col_i) + g;
                set(vel, row_i, col_i, v);
                T_VELOCITY param_update = optimizer_parameters.nesterov ? mom * v + g : v;
                value -= lr * param_update;
                if constexpr(USE_MASTER_PARAMETERS){
                    auto master_params = matrix_view(device, parameter.master_parameters);
                    set(master_params, row_i, col_i, (T_MASTER_PARAMETER)value);
                }
                set(params, row_i, col_i, (T_PARAMETER)value);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
    void update(devices::CUDA<DEV_SPEC>& device, nn::parameters::SGD::Instance<SPEC>& p, nn::optimizers::SGD<PARAMETERS>& optimizer) {
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ROWS = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_COLS = 32;
        using MATRIX_SPEC = typename decltype(matrix_view(device, p.parameters))::SPEC;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ROWS = RL_TOOLS_DEVICES_CUDA_CEIL(MATRIX_SPEC::ROWS, BLOCKSIZE_ROWS);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(MATRIX_SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS, N_BLOCKS_ROWS);
        dim3 block(BLOCKSIZE_COLS, BLOCKSIZE_ROWS);
        nn::optimizers::sgd::cuda::update_kernel<<<grid, block, 0, device.stream>>>(device, p, optimizer);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
