#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CUDA_H

//#include "operations_generic.h"
#include <layer_in_c/devices/cuda.h>
#include <layer_in_c/nn/nn.h>

#define LAYER_IN_C_CEIL(A, B) (A / B + (A % B == 0 ? 0 : 1))

#include <cublas_v2.h>

namespace layer_in_c{
    namespace nn::dense::cuda{
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        __global__ void
        set_biases_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

            TI output_pos = blockIdx.x * blockDim.x + threadIdx.x;
            if(output_pos < OUTPUT_DIM){
                T bias = layer.biases.data[output_pos];
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    output[batch_i * OUTPUT_DIM + output_pos] = bias;
                }
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        void set_biases(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* output) {
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_BIAS = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_BIAS = LAYER_IN_C_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_BIAS);
            dim3 bias_grid(N_BLOCKS_BIAS);
            dim3 bias_block(BLOCKSIZE_BIAS);
            nn::dense::cuda::set_biases_kernel<DEV_SPEC, SPEC, BATCH_SIZE><<<bias_grid, bias_block>>>(device, layer, output);
        }
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        __global__ void
        activate_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* pre_activations, typename SPEC::T* output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

            TI output_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
            TI output_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
            if(output_pos_x < OUTPUT_DIM && output_pos_y < BATCH_SIZE){
                output[output_pos_y * OUTPUT_DIM + output_pos_x] = pre_activations[output_pos_y * OUTPUT_DIM + output_pos_x];
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE>
        void activate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* pre_activations, typename SPEC::T* output) {
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_BATCH = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_BATCH = LAYER_IN_C_CEIL(BATCH_SIZE, BLOCKSIZE_ACTIVATION_BATCH);
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = LAYER_IN_C_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
            dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT, N_BLOCKS_ACTIVATION_BATCH);
            dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT, BLOCKSIZE_ACTIVATION_BATCH);
            nn::dense::cuda::activate_kernel<DEV_SPEC, SPEC, BATCH_SIZE><<<activation_grid, activation_block>>>(device, layer, pre_activations, output);
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;
        {
            nn::dense::cuda::set_biases<DEV_SPEC, LAYER_SPEC, BATCH_SIZE>(device, layer, output.data);

            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = input^T   (B x I)
            // op(B) k x n = weights   (I x O)
            // op(C) m x n = OUTPUT    (B x O)
            constexpr auto m = BATCH_SIZE;
            constexpr auto k = LAYER_SPEC::INPUT_DIM;
            constexpr auto n = LAYER_SPEC::OUTPUT_DIM;
            cublasStatus_t stat;
            if constexpr(utils::typing::is_same_v<T, float>){
                stat = cublasSgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (float*)layer.weights.data, k, (float*)input.data, k, &beta, (float*)output.data, n);
            }
            else{
                stat = cublasDgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (double*)layer.weights.data, k, (double*)input.data, k, &beta, (double*)output.data, n);
            }

            nn::dense::cuda::activate<DEV_SPEC, LAYER_SPEC, BATCH_SIZE>(device, layer, output.data, output.data);
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = input     (B x I)
        // op(B) k x n = weights^T (I x O)
        // op(C) m x n = OUTPUT    (B x O)
        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;

        nn::dense::cuda::set_biases<DEV_SPEC, LAYER_SPEC, BATCH_SIZE>(device, layer, output.data);

        cublasStatus_t stat;
        if constexpr(utils::typing::is_same_v<T, float>){
            stat = cublasSgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (float*)layer.weights.data, k, (float*)input.data, k, &beta, (float*)output.data, n);
        }
        else{
            stat = cublasDgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (double*)layer.weights.data, k, (double*)input.data, k, &beta, (double*)output.data, n);
        }

        copy(device, device, layer.pre_activations, output);

        nn::dense::cuda::activate<DEV_SPEC, LAYER_SPEC, BATCH_SIZE>(device, layer, output.data, output.data);
    }
}
#endif
