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
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::gru::BuffersBackward<SPEC>& buffers){
        malloc(device, buffers.dr_dr_pa);
        malloc(device, buffers.dh_dr);
        malloc(device, buffers.dh_dr_pa);
        malloc(device, buffers.dh_dz);
        malloc(device, buffers.dz_dz_pa);
        malloc(device, buffers.dh_dn);
        malloc(device, buffers.dn_dn_pa);
        malloc(device, buffers.dn_dn_pa_pa);
        malloc(device, buffers.dh_dz_pa);
        malloc(device, buffers.dh_dn_pa);
        malloc(device, buffers.dh_dn_pa_pa);
        malloc(device, buffers.dr_pa);
        malloc(device, buffers.dz_pa);
        malloc(device, buffers.dn_pa);
        malloc(device, buffers.dn_pa_pa);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
    void multiply_broadcast_accumulate(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_OUTPUT::SHAPE{}));
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{}));
        static_assert(get<1>(typename SPEC_OUTPUT::SHAPE{}) == get<1>(typename SPEC_1::SHAPE{}));
        using TI = typename DEVICE::index_t;
        using T = typename SPEC_1::T;
        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
            for(TI j=0; j < get<1>(typename SPEC_1::SHAPE{}); j++){
                T t1_value = get(device, t1, i, j);
                T t2_value = get(device, t2, j);
                T t_output_value = get(device, t_output, i, j);
                set(device, t_output, t1_value * t2_value + t_output_value, i, j);
            }
        }
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
            if(step_i == 0){
                matrix_multiply_broadcast_transpose_bias(device, layer.weights_hidden.parameters, layer.initial_hidden_state.parameters, layer.biases_hidden.parameters, pre_activation_hidden_step);
            }
            else{
                auto output_previous_step = view(device, layer.output, step_i-1);
                matrix_multiply_transpose_bias(device, layer.weights_hidden.parameters, output_previous_step, layer.biases_hidden.parameters, pre_activation_hidden_step);
            }

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
            if(step_i == 0){
                multiply_broadcast_accumulate(device, z_post_activation, layer.initial_hidden_state.parameters, output_step);
            }
            else{
                auto output_previous_step = view(device, layer.output, step_i-1);
                multiply_accumulate(device, z_post_activation, output_previous_step, output_step);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
    void subtract_broadcast(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output) {
        // broadcast t1 along first dimension
        static_assert(length(typename SPEC_1::SHAPE{}) == 1);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUTPUT::SHAPE{}));
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{}));
        static_assert(get<1>(typename SPEC_OUTPUT::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{}));
        using TI = typename DEVICE::index_t;
        using T = typename SPEC_1::T;
        for(TI i=0; i < get<0>(typename SPEC_2::SHAPE{}); i++){
            for(TI j=0; j < get<1>(typename SPEC_2::SHAPE{}); j++){
                T t1_value = get(device, t1, j);
                T t2_value = get(device, t2, i, j);
                set(device, t_output, t1_value - t2_value, i, j);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void matrix_multiply_broadcast_accumulate(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 1);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{}));
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI row_i=0; row_i < get<0>(typename SPEC_1::SHAPE{}); ++row_i){
            for(TI col_j=0; col_j < get<0>(typename SPEC_2::SHAPE{}); ++col_j){
                T acc = get(device, result, row_i, col_j);
                T t2_value = get(device, t2, col_j);
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, row_i, k) * t2_value;
                }
                set(device, result, acc, row_i, col_j);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void matrix_multiply_accumulate_reduce(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 1);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{}));
        static_assert(get<1>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI row_i=0; row_i < get<0>(typename SPEC_1::SHAPE{}); ++row_i){
            for(TI col_j=0; col_j < get<1>(typename SPEC_2::SHAPE{}); ++col_j){
                T acc = get(device, result, col_j);
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, row_i, k) * get(device, t2, k, col_j);
                }
                set(device, result, acc, col_j);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
    void multiply_accumulate_reduce(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUTPUT::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{}));
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{}));
        static_assert(get<1>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUTPUT::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI row_i=0; row_i < get<0>(typename SPEC_1::SHAPE{}); ++row_i){
            for(TI col_j=0; col_j < get<1>(typename SPEC_1::SHAPE{}); ++col_j){
                T increment = get(device, t1, row_i, col_j) * get(device, t2, row_i, col_j);
                set(device, t_output, get(device, t_output, col_j) + increment, col_j);
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn::layers::gru::LayerBackwardGradient<SPEC>& layer) {
        zero_gradient(device, layer.weights_input);
        zero_gradient(device, layer.biases_input);
        zero_gradient(device, layer.weights_hidden);
        zero_gradient(device, layer.biases_hidden);
        zero_gradient(device, layer.initial_hidden_state);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward(DEVICE& device, nn::layers::gru::LayerBackwardGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::gru::BuffersBackward<LAYER_SPEC>& buffers, typename DEVICE::index_t step_i){
        // warning this modifies d_output!
        static_assert(nn::layers::gru::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>, "Input and output spec not matching");
        using TI = typename DEVICE::index_t;
        auto input_step = view(device, input, step_i);
        auto pre_activation_hidden_step = view(device, layer.hidden_pre_activation, step_i);
        auto pre_activation_input_step = view(device, layer.input_pre_activation, step_i);
        auto post_activation_step = view(device, layer.post_activation, step_i);
        auto output_step = view(device, layer.output, step_i);
        auto doutput_step = view(device, d_output, step_i);

        auto z_post_activation = view_range(device, post_activation_step, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        one_minus(device, z_post_activation, buffers.dh_dn);

        // dh_dz = h_{t-1} - n_t
        auto n_post_activation = view_range(device, post_activation_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        if(step_i == 0){
            subtract_broadcast(device, layer.initial_hidden_state.parameters, n_post_activation, buffers.dh_dz);
        }
        else{
            auto output_previous_step = view(device, layer.output, step_i-1);
            subtract(device, output_previous_step, n_post_activation, buffers.dh_dz);
        }
        auto rz_pre_activation = view_range(device, pre_activation_input_step, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, 2*LAYER_SPEC::HIDDEN_DIM>{});
        auto r_pre_activation = view_range(device, pre_activation_input_step, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        auto z_pre_activation = view_range(device, pre_activation_input_step, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        d_sigmoid(device, r_pre_activation, buffers.dr_dr_pa); // todo: feed post activation to prevent recalc of sigmoid
        d_sigmoid(device, z_pre_activation, buffers.dz_dz_pa);
        multiply(device, buffers.dh_dz, buffers.dz_dz_pa, buffers.dh_dz_pa);
        multiply(device, doutput_step, buffers.dh_dz_pa, buffers.dz_pa);
        auto n_pre_activation = view_range(device, pre_activation_input_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        d_tanh(device, n_pre_activation, buffers.dn_dn_pa);
        multiply(device, buffers.dh_dn, buffers.dn_dn_pa, buffers.dh_dn_pa);
        auto r_post_activation = view_range(device, post_activation_step, 0, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        multiply(device, buffers.dh_dn_pa, r_post_activation, buffers.dh_dn_pa_pa);
        auto n_pre_activation_hidden = view_range(device, pre_activation_hidden_step, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<1, LAYER_SPEC::HIDDEN_DIM>{});
        multiply(device, buffers.dh_dn_pa, n_pre_activation_hidden, buffers.dh_dr);
        multiply(device, buffers.dh_dr, buffers.dr_dr_pa, buffers.dh_dr_pa);
        multiply(device, doutput_step, buffers.dh_dr_pa, buffers.dr_pa);
        auto dr_pa_transpose = permute(device, buffers.dr_pa, tensor::PermutationSpec<1, 0>{});
        static_assert(decltype(dr_pa_transpose)::SPEC::SIZE == decltype(buffers.dh_dr_pa)::SPEC::SIZE);
        auto W_ir_grad = view_range(device, layer.weights_input.gradient, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        matrix_multiply_accumulate(device, dr_pa_transpose, input_step, W_ir_grad);
//            print(device, W_ir_grad);
        auto W_hr_grad = view_range(device, layer.weights_hidden.gradient, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        if(step_i == 0){
            matrix_multiply_broadcast_accumulate(device, dr_pa_transpose, layer.initial_hidden_state.parameters, W_hr_grad);
            auto doutput_previous_step = layer.initial_hidden_state.gradient;
            matrix_multiply_accumulate_reduce(device, buffers.dr_pa, layer.W_hr, doutput_previous_step);
        }
        else{
            auto output_previous_step = view(device, layer.output, step_i-1);
            auto doutput_previous_step = view(device, d_output, step_i-1);
            matrix_multiply_accumulate(device, dr_pa_transpose, output_previous_step, W_hr_grad);
            matrix_multiply_accumulate(device, buffers.dr_pa, layer.W_hr, doutput_previous_step);
        }
        // W_hr_grad good
        auto b_ir_grad = view_range(device, layer.biases_input.gradient, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        auto b_hr_grad = view_range(device, layer.biases_hidden.gradient, 0*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        reduce_sum<true>(device, dr_pa_transpose, b_ir_grad);
        reduce_sum<true>(device, dr_pa_transpose, b_hr_grad);

        auto W_iz_grad = view_range(device, layer.weights_input.gradient, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        auto dz_pa_transpose = permute(device, buffers.dz_pa, tensor::PermutationSpec<1, 0>{});
        static_assert(decltype(dr_pa_transpose)::SPEC::SIZE == decltype(buffers.dh_dr_pa)::SPEC::SIZE);
        matrix_multiply_accumulate(device, dz_pa_transpose, input_step, W_iz_grad);

        auto W_hz_grad = view_range(device, layer.weights_hidden.gradient, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        if(step_i == 0){
            matrix_multiply_broadcast_accumulate(device, dz_pa_transpose, layer.initial_hidden_state.parameters, W_hz_grad);
            auto doutput_previous_step = layer.initial_hidden_state.gradient;
            matrix_multiply_accumulate_reduce(device, buffers.dz_pa, layer.W_hz, doutput_previous_step);
        }
        else{
            auto output_previous_step = view(device, layer.output, step_i-1);
            auto doutput_previous_step = view(device, d_output, step_i-1);
            matrix_multiply_accumulate(device, dz_pa_transpose, output_previous_step, W_hz_grad);
            matrix_multiply_accumulate(device, buffers.dz_pa, layer.W_hz, doutput_previous_step);
        }

        auto b_iz_grad = view_range(device, layer.biases_input.gradient, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        auto b_hz_grad = view_range(device, layer.biases_hidden.gradient, 1*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        reduce_sum<true>(device, dz_pa_transpose, b_iz_grad);
        reduce_sum<true>(device, dz_pa_transpose, b_hz_grad); // todo reduce into buffer first and then accumulate into the two biases

        auto W_in_grad = view_range(device, layer.weights_input.gradient, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        multiply(device, doutput_step, buffers.dh_dn_pa, buffers.dn_pa);
        auto dn_pa_transpose = permute(device, buffers.dn_pa, tensor::PermutationSpec<1, 0>{});
        matrix_multiply_accumulate(device, dn_pa_transpose, input_step, W_in_grad);

        auto W_hn_grad = view_range(device, layer.weights_hidden.gradient, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        multiply(device, buffers.dn_pa, r_post_activation, buffers.dn_pa_pa);
        auto dn_pa_pa_transpose = permute(device, buffers.dn_pa_pa, tensor::PermutationSpec<1, 0>{});
        if(step_i == 0){
            matrix_multiply_broadcast_accumulate(device, dn_pa_pa_transpose, layer.initial_hidden_state.parameters, W_hn_grad);
            auto doutput_previous_step = layer.initial_hidden_state.gradient;
            matrix_multiply_accumulate_reduce(device, buffers.dn_pa_pa, layer.W_hn, doutput_previous_step);
            multiply_accumulate_reduce(device, doutput_step, z_post_activation, doutput_previous_step);
        }
        else{
            auto output_previous_step = view(device, layer.output, step_i-1);
            auto doutput_previous_step = view(device, d_output, step_i-1);
            matrix_multiply_accumulate(device, dn_pa_pa_transpose, output_previous_step, W_hn_grad);
            matrix_multiply_accumulate(device, buffers.dn_pa_pa, layer.W_hn, doutput_previous_step);
            multiply_accumulate(device, doutput_step, z_post_activation, doutput_previous_step);
        }
        auto b_in_grad = view_range(device, layer.biases_input.gradient, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        reduce_sum<true>(device, dn_pa_transpose, b_in_grad);
        auto b_hn_grad = view_range(device, layer.biases_hidden.gradient, 2*LAYER_SPEC::HIDDEN_DIM, tensor::ViewSpec<0, LAYER_SPEC::HIDDEN_DIM>{});
        reduce_sum<true>(device, dn_pa_pa_transpose, b_hn_grad);

    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward(DEVICE& device, nn::layers::gru::LayerBackwardGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::gru::BuffersBackward<LAYER_SPEC>& buffers){
        std::cout << "Sequence length: " << LAYER_SPEC::SEQUENCE_LENGTH << std::endl;
        using TI = typename DEVICE::index_t;
        for(TI step_i=LAYER_SPEC::SEQUENCE_LENGTH-1; step_i >= 0; --step_i) {
            backward(device, layer, input, d_output, d_input, buffers, step_i);
            if(step_i == 0){
                break;
            }
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
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::gru::BuffersBackward<SPEC>& layer){
        free(device, layer.dh_dz);
        free(device, layer.dz_dz_pa);
        free(device, layer.dh_dn);
        free(device, layer.dn_dn_pa);
        free(device, layer.dn_dn_pa_pa);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif