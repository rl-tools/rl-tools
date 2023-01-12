#ifndef LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include <layer_in_c/devices/cuda.h>

//namespace layer_in_c{
//    namespace nn_models::mlp::cuda{
//        template<typename DEV_SPEC, typename SPEC>
//        __global__ void
//        evaluate_kernel(devices::CUDA<DEV_SPEC>& device, const nn_models::mlp::NeuralNetwork<SPEC>& p_network, const typename SPEC::T input[utils::typing::remove_reference<decltype(p_network)>::type::INPUT_DIM], typename SPEC::T output[utils::typing::remove_reference<decltype(p_network)>::type::OUTPUT_DIM]){
//            __shared__ nn_models::mlp::NeuralNetwork<SPEC> network;
//            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
////            if(blockIdx.x == 0 && threadIdx.x == 0){
////                network = p_network;
////            }
////            if(blockIdx.x == 0 && threadIdx.x == 0){
////                for(TI i = 0; i < 1000; i++){
////                    layer_in_c::evaluate((devices::CUDA_GENERIC<DEV_SPEC>&)device, (nn_models::mlp::NeuralNetwork<SPEC>&) network, input, output);
////                }
////            }
//            output[0] = 10;
//        }
//    }
//    template<typename DEV_SPEC_CPU, typename DEV_SPEC, typename SPEC>
//    void evaluate(devices::CUDA<DEV_SPEC>& device_cuda, devices::CPU<DEV_SPEC_CPU> device_cpu, const nn_models::mlp::NeuralNetwork<SPEC>& network, const typename SPEC::T* input, typename SPEC::T* output){
//        dim3 grid(1);
//        dim3 block(1);
//        // measure cuda kernel execution time
//        cudaEvent_t start, stop;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start);
//        nn_models::mlp::cuda::evaluate_kernel<<<grid, block>>>(device_cuda, network, input, output);
//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
//        float milliseconds = 0;
//        cudaEventElapsedTime(&milliseconds, start, stop);
//        logging::text(device_cpu.logger, "CUDA kernel execution time: " + std::to_string(milliseconds) + " ms");
//    }
//}
#endif
