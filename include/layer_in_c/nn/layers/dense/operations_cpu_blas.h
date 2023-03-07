#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_BLAS_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_BLAS_H

#include "operations_cpu.h"
#include <layer_in_c/utils/generic/memcpy.h>
#include <layer_in_c/devices/cpu_blas.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);

        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CPU_BLAS<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = input     (B x I)
        // op(B) k x n = weights^T (I x O)
        // op(C) m x n = OUTPUT    (B x O)
        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;

        set_broadcast(device, output, layer.biases.parameters);

        if constexpr(utils::typing::is_same_v<T, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)output._data, row_pitch(output));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)output._data, row_pitch(output));
        }
        for(TI i = 0; i < BATCH_SIZE; i++){
            for(TI j = 0; j < LAYER_SPEC::OUTPUT_DIM; j++){
                set(output, i, j, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, i, j)));
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = input     (B x I)
        // op(B) k x n = weights^T (I x O)
        // op(C) m x n = OUTPUT    (B x O)
        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;


        set_broadcast(device, output, layer.biases.parameters);

        if constexpr(utils::typing::is_same_v<T, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)output._data, row_pitch(output));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)output._data, row_pitch(output));
        }
        copy(device, device, layer.pre_activations, output);
        for(TI i = 0; i < BATCH_SIZE; i++){
            for(TI j = 0; j < LAYER_SPEC::OUTPUT_DIM; j++){
                set(output, i, j, activation<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, i, j)));
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        // Warning do not reuse d_output as d_output is used as a temporary buffer
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;

        {
            // d_weights
            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = d_output^T (O x B)
            // op(B) k x n = input      (B x I)
            // op(C) m x n = d_weights  (O x I)

            constexpr auto m = LAYER_SPEC::OUTPUT_DIM;
            constexpr auto k = BATCH_SIZE;
            constexpr auto n = LAYER_SPEC::INPUT_DIM;

            // calculating pre-activation
            for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
                for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                    T d_pre_activation = d_activation_d_x<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                    increment(layer.biases.gradient, 0, output_i, d_pre_activation);
                    set(d_output, batch_i, output_i, d_pre_activation);
                }
            }
            if constexpr(utils::typing::is_same_v<T, float>){
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)input._data, row_pitch(input), beta, (T*)layer.weights.gradient._data, row_pitch(layer.weights.gradient));
            }
            else{
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)input._data, row_pitch(input), beta, (T*)layer.weights.gradient._data, row_pitch(layer.weights.gradient));
            }
        }
        {
            // d_input
            constexpr T alpha = 1;
            constexpr T beta = 0;
            // op(A) m x k = d_output   (B x O)
            // op(B) k x n = weights    (O x I)
            // op(C) m x n = d_input    (B x I)

            constexpr auto m = BATCH_SIZE;
            constexpr auto k = LAYER_SPEC::OUTPUT_DIM;
            constexpr auto n = LAYER_SPEC::INPUT_DIM;

            if constexpr(utils::typing::is_same_v<T, float>){
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)d_input._data, row_pitch(d_input));
            }
            else{
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)layer.weights.parameters._data, row_pitch(layer.weights.parameters), beta, (T*)d_input._data, row_pitch(d_input));
            }
        }
    }
}

#endif