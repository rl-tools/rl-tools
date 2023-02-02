#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H

//#include "operations_generic.h"
#include <layer_in_c/devices/cuda.h>
#include <layer_in_c/nn/nn.h>

namespace layer_in_c{
    namespace nn::dense::cuda{
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        __global__ void
        evaluate_batch_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, const typename SPEC::T* input, typename SPEC::T* output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

            TI thread_id = blockIdx.x * blockDim.x + threadIdx.x;
            if(thread_id < BATCH_SIZE){
                for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++){
                    TI batch_i = thread_id;
                    auto batch_output_i = batch_i * OUTPUT_DIM + output_i;
                    output[batch_output_i] = layer.biases.data[output_i];
                    for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                        output[batch_output_i] += layer.weights.data[output_i * INPUT_DIM + input_i] * input[batch_i * INPUT_DIM + input_i];
                    }
                    output[batch_output_i] = activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(output[batch_output_i]);
                }
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS = BATCH_SIZE / BLOCKSIZE + (BATCH_SIZE % BLOCKSIZE == 0 ? 0 : 1);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        nn::dense::cuda::evaluate_batch_kernel<DEV_SPEC, LAYER_SPEC, BATCH_SIZE><<<grid, block>>>(device, layer, input.data, output.data);
        // handle cuda error
//        cudaDeviceSynchronize();
//        auto err = cudaGetLastError();
//        if(err != cudaSuccess){
//            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//
//        }
    }
}
#endif
