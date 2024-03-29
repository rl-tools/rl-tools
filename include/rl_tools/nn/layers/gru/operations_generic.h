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

        malloc(device, layer.initial_hidden_state);
        set_all(device, layer.initial_hidden_state.parameters, 0);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::gru::LayerBackward<SPEC>& layer){
        malloc(device, static_cast<nn::layers::gru::Layer<SPEC>&>(layer));
        malloc(device, layer.output);
        malloc(device, layer.pre_activation);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void forward(DEVICE& device, nn::layers::gru::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output){
        static_assert(nn::layers::gru::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>, "Input and output spec not matching");
        using TI = typename DEVICE::index_t;

        for(TI step_i=0; step_i < LAYER_SPEC::SEQUENCE_LENGTH; ++step_i){
            auto input_step = view(device, input, step_i);
            auto output_step = view(device, layer.output, step_i);
            auto pre_activation_step = view(device, layer.pre_activation, step_i);
            decltype(view(device, output, step_i-1)) output_previous_step;
            if(step_i == 0){
                output_previous_step = view(device, output, 0);
                set_all(device, output_previous_step, 0);
            }
            else{
                output_previous_step = view(device, output, step_i-1);
            }
            multiply_transpose_bias(device, layer.weights_hidden.parameters, output_previous_step, layer.biases_input.parameters, output_step);

            std::cout << "Input step " << std::endl;
            print(device, input_step);
            multiply_transpose_bias(device, layer.weights_input.parameters, input_step, layer.biases_input.parameters, pre_activation_step);
            auto rz_preactivation_input = view_range(device, pre_activation_step, 0, tensor::ViewSpec<1, 2*LAYER_SPEC::HIDDEN_DIM>{});
            auto rz_preactivation_hidden = view_range(device, output_step, 0, tensor::ViewSpec<1, 2*LAYER_SPEC::HIDDEN_DIM>{});
            add(device, rz_preactivation_input, rz_preactivation_hidden);
            sigmoid(device, rz_preactivation_input);
            auto r_postactivation = view_range(device, pre_activation_step, 0, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            auto n_preactivation_hidden = view_range(device, output_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            multiply(device, n_preactivation_hidden, rz_preactivation_hidden);
            auto n_preactivation_input = view_range(device, pre_activation_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            add(device, n_preactivation_input, n_preactivation_hidden);
            tanh(device, n_preactivation_input);

        }
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::gru::Layer<SPEC>& layer){
        free(device, layer.weights_input);
        free(device, layer.biases_input);
        free(device, layer.weights_hidden);
        free(device, layer.biases_hidden);
        free(device, layer.initial_hidden_state);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::gru::LayerBackward<SPEC>& layer){
        free(device, static_cast<nn::layers::gru::Layer<SPEC>&>(layer));
        free(device, layer.output);
        free(device, layer.pre_activation);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif