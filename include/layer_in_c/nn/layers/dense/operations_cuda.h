#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H

//#include "operations_generic.h"
#include <layer_in_c/devices/cuda.h>
#include <layer_in_c/nn/nn.h>

namespace layer_in_c{
    namespace nn::dense::cuda{
        template<typename DEV_SPEC, typename SPEC>
        __global__ void
        evaluate_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]){
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            TI thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            if(thread_id < OUTPUT_DIM){
                output[thread_id] = layer.biases[thread_id];
                for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                    output[thread_id] += layer.weights[thread_id][input_i] * input[input_i];
                }
                output[thread_id] = nn::activation_functions::activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(output[thread_id]);
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]) {
        dim3 grid(1);
        dim3 block(SPEC::OUTPUT_DIM);
        // measure cuda kernel execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        nn::dense::cuda::evaluate_kernel<<<grid, block>>>(device, layer, input, output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "layer evaluate elapsed: " << milliseconds << " ms" << std::endl;
    }
}
#endif
