#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_OPERATIONS_GENERIC_H

#include "layer.h"
#include <rl_tools/nn/parameters/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::gru::Layer<SPEC>& layer){
        malloc(device, layer.weights_input);
        using VIEW_SPEC = tensor::ViewSpec<0, SPEC::HIDDEN_DIM>;
        layer.W_ir = view_range(device, layer.weights_input.parameters, 0*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.W_iz = view_range(device, layer.weights_input.parameters, 1*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.W_in = view_range(device, layer.weights_input.parameters, 2*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        malloc(device, layer.biases_input);
        layer.b_ir = view_range(device, layer.biases_input.parameters, 0*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.b_iz = view_range(device, layer.biases_input.parameters, 1*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.b_in = view_range(device, layer.biases_input.parameters, 2*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        malloc(device, layer.weights_hidden);
        layer.W_hr = view_range(device, layer.weights_hidden.parameters, 0*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.W_hz = view_range(device, layer.weights_hidden.parameters, 1*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.W_hn = view_range(device, layer.weights_hidden.parameters, 2*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        malloc(device, layer.biases_hidden);
        layer.b_hr = view_range(device, layer.biases_hidden.parameters, 0*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.b_hz = view_range(device, layer.biases_hidden.parameters, 1*SPEC::HIDDEN_DIM, VIEW_SPEC{});
        layer.b_hn = view_range(device, layer.biases_hidden.parameters, 2*SPEC::HIDDEN_DIM, VIEW_SPEC{});

        malloc(device, layer.initial_hidden_state.parameters);
        set_all(device, layer.initial_hidden_state.parameters, 0);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void forward(DEVICE& device, const nn::layers::gru::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output){
        static_assert(nn::layers::gru::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>, "Input and output spec not matching");
        using TI = typename DEVICE::index_t;

        for(TI step_i=0; step_i < LAYER_SPEC::SEQUENCE_LENGTH; ++step_i){
            auto input_step = view(device, input, step_i, tensor::ViewSpec<0>{});
            std::cout << "Input step " << std::endl;
            print(device, input_step);
        }
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::gru::Layer<SPEC>& layer){
        malloc(device, layer.weights_input);
        malloc(device, layer.biases_input);
        malloc(device, layer.weights_hidden);
        malloc(device, layer.biases_hidden);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif