#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_ACCELERATE_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_ACCELERATE_H

#include <Accelerate/Accelerate.h>
#include "operations_cpu_blas.h"
#include <layer_in_c/devices/cpu_accelerate.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT void evaluate(devices::CPU_ACCELERATE<DEV_SPEC>& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        evaluate((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT void forward(devices::CPU_ACCELERATE<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        forward((devices::CPU_BLAS<DEV_SPEC>&) device, layer, input, output);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT void backward(devices::CPU_ACCELERATE<DEV_SPEC>& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        backward((devices::CPU_BLAS<DEV_SPEC> &) device, layer, input, d_output, d_input);
    }
}

#endif