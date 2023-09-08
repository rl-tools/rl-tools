#include "../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_MKL_H

#include "operations_cpu_blas.h"
#include "../../../devices/cpu_mkl.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        evaluate((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        forward((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void backward_input(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        backward_input((devices::CPU_BLAS<DEV_SPEC> &) device, layer, d_output, d_input);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void backward_param(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output) {
        backward_param((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        backward((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output, d_input);
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif