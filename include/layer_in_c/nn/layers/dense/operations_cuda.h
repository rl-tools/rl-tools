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
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            TI thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            if(thread_id < OUTPUT_DIM){
                T acc = layer.biases[thread_id];
                for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                    acc += layer.weights[thread_id][input_i] * input[input_i];
                }
                acc = activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(acc);
                output[thread_id] = acc;
            }
        }

        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        __global__ void
        evaluate_batch_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& p_layer, const typename SPEC::T* p_input, typename SPEC::T* output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            using TX_TYPE = uint8_t;
            constexpr TI NUM_TX_OPS = sizeof(nn::layers::dense::Layer<SPEC>);
            TI NUM_TX_ITERATIONS = NUM_TX_OPS/blockDim.x + (NUM_TX_OPS % blockDim.x == 0 ? 0 : 1);
            __shared__ nn::layers::dense::Layer<SPEC> layer;

            for(TI sm_i = 0; sm_i < NUM_TX_ITERATIONS; sm_i++) {
                TI sm_offset = sm_i * blockDim.x + threadIdx.x;
                if(sm_offset < NUM_TX_OPS) {
                    ((TX_TYPE*)&layer)[sm_offset] = ((TX_TYPE*)&p_layer)[sm_offset];
                }
            }
            __syncthreads();
            TI thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            T input[INPUT_DIM];
            for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                input[input_i] = p_input[thread_id * INPUT_DIM + input_i];
            }
            if(thread_id < BATCH_SIZE){
                for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++){
                    auto batch_output_i = thread_id * OUTPUT_DIM + output_i;
                    output[batch_output_i] = layer.biases[output_i];
                    for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                        output[batch_output_i] += layer.weights[output_i][input_i] * input[input_i];
//                        output[batch_output_i] += layer.weights[output_i][input_i] * p_input[thread_id * INPUT_DIM + input_i];
                    }
                    output[batch_output_i] = activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(output[batch_output_i]);
                }
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]) {
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS = SPEC::OUTPUT_DIM / BLOCKSIZE + (SPEC::OUTPUT_DIM % BLOCKSIZE == 0 ? 0 : 1);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        nn::dense::cuda::evaluate_kernel<<<grid, block>>>(device, layer, input, output);
        auto error = cudaGetLastError();
        if(error != cudaSuccess) {
            printf("CUDA error: %s", cudaGetErrorString(error));
        }
    }

    template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
    void evaluate_batch(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, const typename SPEC::T* input, typename SPEC::T* output) {
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE = 1024;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS = BATCH_SIZE / BLOCKSIZE + (BATCH_SIZE % BLOCKSIZE == 0 ? 0 : 1);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        nn::dense::cuda::evaluate_batch_kernel<DEV_SPEC, SPEC, BATCH_SIZE><<<grid, block>>>(device, layer, input, output);
    }
}
#endif
