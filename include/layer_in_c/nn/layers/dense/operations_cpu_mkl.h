#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H

#include "operations_generic.h"
#include <layer_in_c/devices/cpu_mkl.h>

#include "mkl.h"

namespace layer_in_c{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    FUNCTION_PLACEMENT void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CPU_MKL<DEV_SPEC>;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;

        T alpha, beta;
        alpha = 1.0; beta = 0.0;

        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;

        // A m x k
        // B k x n
        // C m x n

        set(output, 0);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, input.data, k, layer.weights.data, n, beta, output.data, n);
    }
}

