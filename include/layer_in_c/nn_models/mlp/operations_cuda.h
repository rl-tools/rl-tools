#ifndef LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include <layer_in_c/devices/cuda.h>
#include <array>
namespace layer_in_c{
    namespace nn_models::mlp::cuda{
        template<typename DEV_SPEC, typename SPEC>
        __global__ void
        evaluate_kernel(const nn_models::mlp::NeuralNetwork<devices::CUDA<DEV_SPEC>, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], typename SPEC::T output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM]){
            if(blockIdx.x == 0 && threadIdx.x == 0){
                layer_in_c::evaluate((nn_models::mlp::NeuralNetwork<devices::CUDA_GENERIC<DEV_SPEC>, SPEC>&) network, input, output);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC>
    void evaluate(const nn_models::mlp::NeuralNetwork<devices::CUDA<DEV_SPEC>, SPEC>& network, const typename SPEC::T* input, typename SPEC::T* output){
        dim3 grid(1);
        dim3 block(1);
        cudaDeviceSynchronize();
        nn_models::mlp::cuda::evaluate_kernel<<<grid, block>>>(network, input, output);
        cudaDeviceSynchronize();
    }
}
#endif
