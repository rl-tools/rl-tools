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
        set_biases(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* output) {
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
        __global__ void
        activate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, typename SPEC::T* pre_activations, typename SPEC::T* output) {
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
        template<typename DEV_SPEC, typename SPEC, typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE, typename devices::CUDA<DEV_SPEC>::index_t BLOCK_SIZE>
        __global__ void
        evaluate_batch_kernel(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC> layer, const typename SPEC::T* input, typename SPEC::T* output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

            __shared__ T shared_input[BLOCK_SIZE * BLOCK_SIZE];
            __shared__ T shared_weights[BLOCK_SIZE * BLOCK_SIZE];

            assert(BLOCK_SIZE == blockDim.x);
            assert(BLOCK_SIZE == blockDim.y);

            TI block_output_pos = blockIdx.x * BLOCK_SIZE;
            TI block_batch_pos = blockIdx.y * BLOCK_SIZE;
            TI thread_output_pos = block_output_pos + threadIdx.y;
            TI thread_batch_pos = block_batch_pos + threadIdx.y;

            TI thread_block_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

            TI print_block_idx = 0;
            TI print_block_idy = 1;
            TI print_thread_idx = 4;
            TI print_thread_idy = 0;

            T acc = 0;
            for(TI block_reduction_i = 0; block_reduction_i < LAYER_IN_C_CEIL(INPUT_DIM, BLOCK_SIZE) * BLOCK_SIZE; block_reduction_i += BLOCK_SIZE){
                TI thread_input_pos = block_reduction_i + threadIdx.x;
                if(thread_input_pos < INPUT_DIM && thread_batch_pos < BATCH_SIZE){
                    shared_input[thread_block_index] = input[thread_batch_pos * INPUT_DIM + thread_input_pos];
                }
                else{
                    shared_input[thread_block_index] = 0;
                }
                if(thread_input_pos < INPUT_DIM && thread_output_pos < OUTPUT_DIM){
                    shared_weights[thread_block_index] = layer.weights.data[thread_output_pos * INPUT_DIM + thread_input_pos];
                }
                else{
                    shared_weights[thread_block_index] = 0;
                }
                __syncthreads();
//                if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                    printf("input:\n");
//                    for(TI i = 0; i < BLOCK_SIZE; i++){
//                        for(TI j = 0; j < BLOCK_SIZE; j++){
//                            printf(" %f", shared_input[i * BLOCK_SIZE + j]);
//                        }
//                        printf("\n");
//                    }
//                    printf("weights:\n");
//                    for(TI i = 0; i < BLOCK_SIZE; i++){
//                        for(TI j = 0; j < BLOCK_SIZE; j++){
//                            printf(" %f", shared_weights[i * BLOCK_SIZE + j]);
//                        }
//                        printf("\n");
//                    }
//                }
                // x: output, y: batch
                for(TI reduction_i = 0; reduction_i < BLOCK_SIZE; reduction_i++){
                    T a = shared_weights[threadIdx.x * BLOCK_SIZE + reduction_i];
                    T b = shared_input[threadIdx.y * BLOCK_SIZE + reduction_i];
                    acc += a * b;
//                    if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                        printf("a: %f, b: %f\n", a, b);
//                    }
                }
            }
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }
            T b = layer.biases.data[blockIdx.x * BLOCK_SIZE + threadIdx.x];
            acc += b;
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("bias: %f\n",  b);
//            }
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }
            acc = activation<typename devices::CUDA<DEV_SPEC>::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(acc);
//            if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                printf("result: %f\n",  acc);
//            }

            if(blockIdx.y * BLOCK_SIZE + threadIdx.y < BATCH_SIZE && blockIdx.x * BLOCK_SIZE + threadIdx.x < OUTPUT_DIM){
//                if(blockIdx.x == print_block_idx && blockIdx.y == print_block_idy && threadIdx.x == print_thread_idx && threadIdx.y == print_thread_idy){
//                    printf("writing %f to: %d %d\n", acc, blockIdx.y * BLOCK_SIZE + threadIdx.y, blockIdx.x * BLOCK_SIZE + threadIdx.x);
//                }
                output[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * OUTPUT_DIM + blockIdx.x * BLOCK_SIZE + threadIdx.x] = acc;
            }
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
//        {
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_BATCH = 32;
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_BATCH = LAYER_IN_C_CEIL(BATCH_SIZE, BLOCKSIZE_BATCH);
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_OUTPUT = 32;
//            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_OUTPUT = LAYER_IN_C_CEIL(BATCH_SIZE, BLOCKSIZE_OUTPUT);
//            dim3 grid(N_BLOCKS_OUTPUT, N_BLOCKS_BATCH);
//            dim3 block(BLOCKSIZE_OUTPUT, BLOCKSIZE_BATCH);
//            nn::dense::cuda::evaluate_batch_kernel<DEV_SPEC, LAYER_SPEC, BATCH_SIZE, BLOCKSIZE_BATCH><<<grid, block>>>(device, layer, input.data, output.data);
////          handle cuda error
//            cudaDeviceSynchronize();
//            auto err = cudaGetLastError();
//            if(err != cudaSuccess){
//                std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//
//            }
//        }
        {
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_BIAS = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_BIAS = LAYER_IN_C_CEIL(LAYER_SPEC::OUTPUT_DIM, BLOCKSIZE_BIAS);
            dim3 bias_grid(N_BLOCKS_BIAS);
            dim3 bias_block(BLOCKSIZE_BIAS);
            nn::dense::cuda::set_biases<DEV_SPEC, LAYER_SPEC, BATCH_SIZE><<<bias_grid, bias_block>>>(device, layer, output.data);

            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = input     (B x I)
            // op(B) k x n = weights^T (I x O)
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

            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_BATCH = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_BATCH = LAYER_IN_C_CEIL(BATCH_SIZE, BLOCKSIZE_ACTIVATION_BATCH);
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = LAYER_IN_C_CEIL(LAYER_SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
            dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT, N_BLOCKS_ACTIVATION_BATCH);
            dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT, BLOCKSIZE_ACTIVATION_BATCH);
            nn::dense::cuda::activate<DEV_SPEC, LAYER_SPEC, BATCH_SIZE><<<activation_grid, activation_block>>>(device, layer, output.data, output.data);
        }
    }
}
#endif
