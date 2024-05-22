#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H

#include "operations_cpu_blas.h"
#include "../../../devices/cpu_mkl.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::dense::Buffer& buffer, RNG& rng){
        evaluate((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output, buffer, rng);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, RNG& rng){
        forward((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output, rng);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer&) {
        backward_input((devices::CPU_BLAS<DEV_SPEC> &) device, layer, d_output, d_input);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        nn::layers::dense::Buffer buffer;
        backward_input(device, layer, d_output, d_input, buffer);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::dense::Buffer&) {
        backward((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output) {
        nn::layers::dense::Buffer buffer;
        backward(device, layer, input, d_output, buffer);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer&) {
        backward_full((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output, d_input);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        nn::layers::dense::Buffer buffer;
        backward_full(device, layer, input, d_output, d_input, buffer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif