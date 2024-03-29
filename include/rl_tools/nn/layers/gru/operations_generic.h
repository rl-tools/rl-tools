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
        malloc(device, layer.input_pre_activation);
        malloc(device, layer.hidden_pre_activation);
        malloc(device, layer.post_activation);
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC>
    void forward(DEVICE& device, nn::layers::gru::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input){
        static_assert(nn::layers::gru::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>, "Input and output spec not matching");
        using TI = typename DEVICE::index_t;

        for(TI step_i=0; step_i < LAYER_SPEC::SEQUENCE_LENGTH; ++step_i){
            auto input_step = view(device, input, step_i);
            auto pre_activation_hidden_step = view(device, layer.hidden_pre_activation, step_i);
            auto pre_activation_input_step = view(device, layer.input_pre_activation, step_i);
            auto post_activation_step = view(device, layer.post_activation, step_i);
            auto output_step = view(device, layer.output, step_i);
            decltype(view(device, layer.output, step_i-1)) output_previous_step;
            if(step_i == 0){
                output_previous_step = view(device, layer.output, 0);
                set_all(device, output_previous_step, 0);
            }
            else{
                output_previous_step = view(device, layer.output, step_i-1);
            }
            matrix_multiply_transpose_bias(device, layer.weights_hidden.parameters, output_previous_step, layer.biases_input.parameters, pre_activation_hidden_step);

            matrix_multiply_transpose_bias(device, layer.weights_input.parameters, input_step, layer.biases_input.parameters, pre_activation_input_step);
            auto rz_pre_activation_input = view_range(device, pre_activation_input_step, 0, tensor::ViewSpec<1, 2*LAYER_SPEC::HIDDEN_DIM>{});
            auto rz_pre_activation_hidden = view_range(device, pre_activation_hidden_step, 0, tensor::ViewSpec<1, 2*LAYER_SPEC::HIDDEN_DIM>{});
            add(device, rz_pre_activation_input, rz_pre_activation_hidden);
            auto rz_post_activation = view_range(device, post_activation_step, 0, tensor::ViewSpec<1,2* LAYER_SPEC::HIDDEN_DIM>{});
            sigmoid(device, rz_pre_activation_input, rz_post_activation);
            auto r_post_activation = view_range(device, post_activation_step, 0, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            auto n_pre_activation_hidden = view_range(device, pre_activation_hidden_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            auto n_pre_activation_input = view_range(device, pre_activation_input_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            multiply_accumulate(device, n_pre_activation_hidden, r_post_activation, n_pre_activation_input);
            auto n_post_activation = view_range(device, post_activation_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            tanh(device, n_pre_activation_input, n_post_activation);
            auto z_post_activation = view_range(device, post_activation_step, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
            one_minus(device, z_post_activation, output_step);
            multiply(device, output_step, n_post_activation);
            multiply_accumulate(device, z_post_activation, output_previous_step, output_step);
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
        free(device, layer.input_pre_activation);
        free(device, layer.hidden_pre_activation);
        free(device, layer.post_activation);
        free(device, layer.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif