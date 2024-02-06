#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CUDA_H

//#include "operations_generic.h"
#include "../../../devices/cuda.h"
#include "../../../nn/parameters/operations_cuda.h"
#include "../../../nn/nn.h"

#include <cublas_v2.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::dense::cuda{
        template<typename DEV_SPEC, typename SPEC, typename OUTPUT_SPEC>
        __global__ void
        set_biases_kernel(devices::CUDA<DEV_SPEC> device, const nn::layers::dense::Layer<SPEC> layer, Matrix<OUTPUT_SPEC> output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            constexpr TI BATCH_SIZE = OUTPUT_SPEC::ROWS;

            TI output_pos = blockIdx.x * blockDim.x + threadIdx.x;
            if(output_pos < OUTPUT_DIM){
                T bias = get(layer.biases.parameters, 0, output_pos);
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    set(output, batch_i, output_pos, bias);
                }
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename OUTPUT_SPEC>
        void set_biases(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, Matrix<OUTPUT_SPEC>& output) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_BIAS = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_BIAS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_BIAS);
            dim3 bias_grid(N_BLOCKS_BIAS);
            dim3 bias_block(BLOCKSIZE_BIAS);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            nn::dense::cuda::set_biases_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, layer, output);
            check_status(device);
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename OUTPUT_SPEC>
        __global__ void
        activation_kernel(devices::CUDA<DEV_SPEC> device, const nn::layers::dense::Layer<SPEC>& layer, Matrix<PRE_ACTIVATIONS_SPEC> pre_activations, Matrix<OUTPUT_SPEC> output) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            static_assert(PRE_ACTIVATIONS_SPEC::ROWS == OUTPUT_SPEC::ROWS);
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            constexpr TI BATCH_SIZE = PRE_ACTIVATIONS_SPEC::ROWS;

            TI output_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
            TI output_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
            if(output_pos_x < OUTPUT_DIM && output_pos_y < BATCH_SIZE){
                set(output, output_pos_y, output_pos_x, rl_tools::activation<typename DEV_SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(get(pre_activations, output_pos_y, output_pos_x)));
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename OUTPUT_SPEC>
        void activation(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, Matrix<PRE_ACTIVATIONS_SPEC>& pre_activations, Matrix<OUTPUT_SPEC>& output){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            static_assert(PRE_ACTIVATIONS_SPEC::ROWS == OUTPUT_SPEC::ROWS);
            constexpr TI BATCH_SIZE = PRE_ACTIVATIONS_SPEC::ROWS;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_BATCH = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_BATCH = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_ACTIVATION_BATCH);
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
            dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT, N_BLOCKS_ACTIVATION_BATCH);
            dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT, BLOCKSIZE_ACTIVATION_BATCH);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            nn::dense::cuda::activation_kernel<<<activation_grid, activation_block, 0, device.stream>>>(tag_device, layer, pre_activations, output);
            check_status(device);
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename D_OUTPUT_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
        __global__ void
        d_activation_kernel(devices::CUDA<DEV_SPEC> device, const nn::layers::dense::Layer<SPEC> layer, Matrix<PRE_ACTIVATIONS_SPEC> pre_activations, Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_PRE_ACTIVATIONS_SPEC> d_pre_activations) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            static_assert(containers::check_structure<PRE_ACTIVATIONS_SPEC, D_OUTPUT_SPEC>);
            static_assert(containers::check_structure<D_OUTPUT_SPEC, D_PRE_ACTIVATIONS_SPEC>);
            constexpr TI BATCH_SIZE = PRE_ACTIVATIONS_SPEC::ROWS;

            TI output_i = blockIdx.x * blockDim.x + threadIdx.x;
            if(output_i < OUTPUT_DIM){
                T acc = 0;
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    T d_pre_activation_temp = d_activation_d_x<typename DEV_SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(get(pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                    set(d_pre_activations, batch_i, output_i, d_pre_activation_temp);
                    acc += d_pre_activation_temp;
                }
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename D_OUTPUT_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
        void d_activation(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, Matrix<PRE_ACTIVATIONS_SPEC>& pre_activations, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_PRE_ACTIVATIONS_SPEC>& d_pre_activations) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
            dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT);
            dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            nn::dense::cuda::d_activation_kernel<<<activation_grid, activation_block, 0, device.stream>>>(tag_device, layer, pre_activations, d_output, d_pre_activations);
            check_status(device);
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename D_OUTPUT_SPEC, typename D_BIASES_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
        __global__ void
        d_activation_accumulate_bias_gradient_kernel(devices::CUDA<DEV_SPEC> device, const nn::layers::dense::Layer<SPEC> layer, Matrix<PRE_ACTIVATIONS_SPEC> pre_activations, Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_BIASES_SPEC> d_biases, Matrix<D_PRE_ACTIVATIONS_SPEC> d_pre_activations) {
            using T = typename SPEC::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
            static_assert(containers::check_structure<PRE_ACTIVATIONS_SPEC, D_OUTPUT_SPEC>);
            static_assert(containers::check_structure<D_OUTPUT_SPEC, D_PRE_ACTIVATIONS_SPEC>);
            constexpr TI BATCH_SIZE = PRE_ACTIVATIONS_SPEC::ROWS;
            static_assert(PRE_ACTIVATIONS_SPEC::COLS == D_BIASES_SPEC::COLS);

            TI output_i = blockIdx.x * blockDim.x + threadIdx.x;
            if(output_i < OUTPUT_DIM){
                T acc = 0;
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    T d_pre_activation_temp = d_activation_d_x<typename DEV_SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(get(pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                    set(d_pre_activations, batch_i, output_i, d_pre_activation_temp);
                    acc += d_pre_activation_temp;
                }
                increment(d_biases, 0, output_i, acc);
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename PRE_ACTIVATIONS_SPEC, typename D_OUTPUT_SPEC, typename D_BIASES_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
        void d_activation_accumulate_bias_gradient(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<SPEC>& layer, Matrix<PRE_ACTIVATIONS_SPEC>& pre_activations, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_BIASES_SPEC>& d_biases, Matrix<D_PRE_ACTIVATIONS_SPEC>& d_pre_activations) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
            constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
            dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT);
            dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            nn::dense::cuda::d_activation_accumulate_bias_gradient_kernel<<<activation_grid, activation_block, 0, device.stream>>>(tag_device, layer, pre_activations, d_output, d_biases, d_pre_activations);
            check_status(device);
        }
        template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
        __global__
        void update_kernel(devices::CUDA<DEV_SPEC> device, nn::layers::dense::LayerBackwardGradient<SPEC> layer, nn::optimizers::Adam<PARAMETERS> optimizer) {
            // fully fused adam update
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
            constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

            TI input_i = blockIdx.x * blockDim.x + threadIdx.x;
            TI output_i = blockIdx.y * blockDim.y + threadIdx.y;
            if(input_i < INPUT_DIM && output_i < OUTPUT_DIM){
                if(input_i == 0){
                    T d_bias = get(layer.biases.gradient, 0, output_i);
                    T d_bias_first_order_moment = optimizer.parameters.beta_1 * get(layer.biases.gradient_first_order_moment, 0, output_i) + (1 - optimizer.parameters.beta_1) * d_bias;
                    set(layer.biases.gradient_first_order_moment, 0, output_i, d_bias_first_order_moment);
                    T d_bias_second_order_moment = optimizer.parameters.beta_2 * get(layer.biases.gradient_second_order_moment, 0, output_i) + (1 - optimizer.parameters.beta_2) * d_bias * d_bias;
                    set(layer.biases.gradient_second_order_moment, 0, output_i, d_bias_second_order_moment);
                    T bias_update = optimizer.parameters.alpha * optimizer.first_order_moment_bias_correction * d_bias_first_order_moment / (math::sqrt(typename DEVICE::SPEC::MATH_DEVICE_ACCURATE(), d_bias_second_order_moment * optimizer.second_order_moment_bias_correction) + optimizer.parameters.epsilon);
                    increment(layer.biases.parameters, 0, output_i, -bias_update);
                }
                T d_weight = get(layer.weights.gradient, output_i, input_i);
                T d_weight_first_order_moment = optimizer.parameters.beta_1 * get(layer.weights.gradient_first_order_moment, output_i, input_i) + (1 - optimizer.parameters.beta_1) * d_weight;
                set(layer.weights.gradient_first_order_moment, output_i, input_i, d_weight_first_order_moment);
                T d_weight_second_order_moment = optimizer.parameters.beta_2 * get(layer.weights.gradient_second_order_moment, output_i, input_i) + (1 - optimizer.parameters.beta_2) * d_weight * d_weight;
                set(layer.weights.gradient_second_order_moment, output_i, input_i, d_weight_second_order_moment);
                T weight_update = optimizer.parameters.alpha * optimizer.first_order_moment_bias_correction * d_weight_first_order_moment / (math::sqrt(typename DEVICE::SPEC::MATH_DEVICE_ACCURATE(), d_weight_second_order_moment * optimizer.second_order_moment_bias_correction) + optimizer.parameters.epsilon);
                increment(layer.weights.parameters, output_i, input_i, -weight_update);
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(INPUT_SPEC::COL_PITCH == 1);
        static_assert(OUTPUT_SPEC::COL_PITCH == 1);
        static_assert(decltype(layer.weights.parameters)::COL_PITCH == 1);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;
        {
            nn::dense::cuda::set_biases(device, layer, output);

            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = WEIGHTS^T^T (O x I)
            // op(B) k x n = INPUT^T     (I x B)
            // op(C) m x n = OUTPUT^T    (O x B)
            constexpr auto m = LAYER_SPEC::OUTPUT_DIM;
            constexpr auto k = LAYER_SPEC::INPUT_DIM;
            constexpr auto n = BATCH_SIZE;
            cublasStatus_t stat;
            if constexpr(utils::typing::is_same_v<T, float>){
                stat = cublasSgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)input._data, row_pitch(input), &beta, (T*)output._data, row_pitch(output));
            }
            else{
                stat = cublasDgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)input._data, row_pitch(input), &beta, (T*)output._data, row_pitch(output));
            }
            if(stat != CUBLAS_STATUS_SUCCESS){
                std::cout << "CUBLAS ERROR: " << cublasGetStatusString(stat) << std::endl;
            }
            nn::dense::cuda::activation(device, layer, output, output);
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(INPUT_SPEC::COL_PITCH == 1);
        static_assert(OUTPUT_SPEC::COL_PITCH == 1);
        static_assert(decltype(layer.weights.parameters)::COL_PITCH == 1);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = weights^T^T (O x I)
        // op(B) k x n = input^T     (I x B)
        // op(C) m x n = output^T    (O x B)
        constexpr auto m = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = BATCH_SIZE;

        nn::dense::cuda::set_biases(device, layer, output);

        cublasStatus_t stat;
        if constexpr(utils::typing::is_same_v<T, float>){
            stat = cublasSgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)input._data, row_pitch(input), &beta, (T*)output._data, row_pitch(output));
        }
        else{
            stat = cublasDgemm(device.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)input._data, row_pitch(input), &beta, (T*)output._data, row_pitch(output));
        }
        if(stat != CUBLAS_STATUS_SUCCESS){
            std::cout << "CUBLAS ERROR: " << cublasGetStatusString(stat) << std::endl;
        }

        copy(device, device, output, layer.pre_activations);

        nn::dense::cuda::activation(device, layer, output, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward_input_additional(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(D_OUTPUT_SPEC::COL_PITCH == 1);
        static_assert(D_INPUT_SPEC::COL_PITCH == 1);
        static_assert(decltype(layer.weights.gradient)::COL_PITCH == 1);

        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        {
            // d_input
            constexpr T alpha = 1;
            constexpr T beta = 0;
            // op(A) m x k = weights^T  (I x O)
            // op(B) k x n = d_output^T (O x B)
            // op(C) m x n = d_input^T  (I x B)

            constexpr auto m = LAYER_SPEC::INPUT_DIM;
            constexpr auto n = BATCH_SIZE;
            constexpr auto k = LAYER_SPEC::OUTPUT_DIM;

            cublasStatus_t stat;
            if constexpr(utils::typing::is_same_v<T, float>){
                stat = cublasSgemm(device.handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)d_output._data, row_pitch(d_output), &beta, (T*)d_input._data, row_pitch(d_input));
            }
            else{
                stat = cublasDgemm(device.handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), (T*)d_output._data, row_pitch(d_output), &beta, (T*)d_input._data, row_pitch(d_input));
            }
            if(stat != CUBLAS_STATUS_SUCCESS){
                std::cout << "CUBLAS ERROR: " << cublasGetStatusString(stat) << std::endl;
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
    void backward_param(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output) {
        // Warning do not reuse d_output as d_output is used as a temporary buffer
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(INPUT_SPEC::COL_PITCH == 1);
        static_assert(D_OUTPUT_SPEC::COL_PITCH == 1);
        static_assert(decltype(layer.weights.gradient)::COL_PITCH == 1);

        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        {
            // weights.gradient
            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = input^T       (I x B)
            // op(B) k x n = d_output^T^T  (B x O)
            // op(C) m x n = d_weights^T   (I x O)

            constexpr auto m = LAYER_SPEC::INPUT_DIM;
            constexpr auto n = LAYER_SPEC::OUTPUT_DIM;
            constexpr auto k = BATCH_SIZE;

            nn::dense::cuda::d_activation_accumulate_bias_gradient(device, layer, layer.pre_activations, d_output, layer.biases.gradient, d_output);

            cublasStatus_t stat;
            if constexpr(utils::typing::is_same_v<T, float>){
                stat = cublasSgemm(device.handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, (T*)input._data, row_pitch(input), (T*)d_output._data, row_pitch(d_output), &beta, (T*)layer.weights.gradient._data, row_pitch(layer.weights.gradient));
            }
            else{
                stat = cublasDgemm(device.handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, (T*)input._data, row_pitch(input), (T*)d_output._data, row_pitch(d_output), &beta, (T*)layer.weights.gradient._data, row_pitch(layer.weights.gradient));
            }
            if(stat != CUBLAS_STATUS_SUCCESS){
                std::cout << "CUBLAS ERROR: " << cublasGetStatusString(stat) << std::endl;
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        backward_param(device, layer, input, d_output);
        backward_input_additional(device, layer, d_output, d_input);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward_input(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        nn::dense::cuda::d_activation(device, layer, layer.pre_activations, d_output, d_output);
        backward_input_additional(device, layer, d_output, d_input);
    }

    template<typename DEV_SPEC, typename SPEC>
    void zero_gradient(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        cudaMemset(layer.weights.gradient._data, 0, decltype(layer.weights.gradient)::SPEC::SIZE_BYTES);
        check_status(device);
        cudaMemset(layer.biases.gradient._data, 0, decltype(layer.biases.gradient)::SPEC::SIZE_BYTES);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
    void _reset_optimizer_state(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, nn::optimizers::Adam<PARAMETERS>& optimizer) {
        cudaMemset(layer.weights.gradient_first_order_moment._data, 0, decltype(layer.weights.gradient_first_order_moment)::SPEC::SIZE_BYTES);
        check_status(device);
        cudaMemset(layer.weights.gradient_second_order_moment._data, 0, decltype(layer.weights.gradient_second_order_moment)::SPEC::SIZE_BYTES);
        check_status(device);
        cudaMemset(layer.biases.gradient_first_order_moment._data, 0, decltype(layer.biases.gradient_first_order_moment)::SPEC::SIZE_BYTES);
        check_status(device);
        cudaMemset(layer.biases.gradient_second_order_moment._data, 0, decltype(layer.biases.gradient_second_order_moment)::SPEC::SIZE_BYTES);
        check_status(device);
    }

    template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
    void update(devices::CUDA<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, nn::optimizers::Adam<PARAMETERS>& optimizer) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_INPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_INPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INPUT_DIM, BLOCKSIZE_ACTIVATION_INPUT);
        dim3 activation_grid(N_BLOCKS_ACTIVATION_INPUT, N_BLOCKS_ACTIVATION_OUTPUT);
        dim3 activation_block(BLOCKSIZE_ACTIVATION_INPUT, BLOCKSIZE_ACTIVATION_OUTPUT);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::dense::cuda::update_kernel<<<activation_grid, activation_block, 0, device.stream>>>(tag_device, layer, optimizer);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
