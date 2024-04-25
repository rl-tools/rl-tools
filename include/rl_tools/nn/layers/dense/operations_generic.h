#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H

#include "../../../containers.h"
#include "../../../nn/parameters/operations_generic.h"

#include "../../../nn/layers/dense/layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer) {
        malloc(device, layer.weights);
        malloc(device, layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer) {
        free(device, layer.weights);
        free(device, layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer) {
        malloc(device, (nn::layers::dense::Layer<SPEC>&) layer);
        malloc(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer) {
        free(device, (nn::layers::dense::Layer<SPEC>&) layer);
        free(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        malloc(device, (nn::layers::dense::LayerBackward<SPEC>&) layer);
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        free(device, (nn::layers::dense::LayerBackward<SPEC>&) layer);
        free(device, layer.output);
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    void init_kaiming(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T negative_slope = math::sqrt(device.math, (T)5);
        T gain = math::sqrt(device.math, (T)2.0 / (1 + negative_slope * negative_slope));
        T fan = SPEC::INPUT_DIM;
        T std = gain / math::sqrt(device.math, fan);
        T weight_bound = math::sqrt(device.math, (T)3.0) * std;
        T bias_bound = 1/math::sqrt(device.math, (T)SPEC::INPUT_DIM);
        for(TI i = 0; i < SPEC::OUTPUT_DIM; i++) {
            set(layer.biases.parameters, 0, i, (T)random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -bias_bound, bias_bound, rng));
            for(TI j = 0; j < SPEC::INPUT_DIM; j++) {
                set(layer.weights.parameters, i, j, random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -weight_bound, weight_bound, rng));
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, RNG& rng) {
        init_kaiming(device, layer, rng);
    }

#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    void evaluate(DEVICE& device, const nn::layers::dense::Layer<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, RNG& rng) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using TI = typename DEVICE::index_t;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < LAYER_SPEC::OUTPUT_DIM; output_i++) {
                set(output, batch_i, output_i, get(layer.biases.parameters, 0, output_i));
                for(TI input_i = 0; input_i < LAYER_SPEC::INPUT_DIM; input_i++) {
                    increment(output, batch_i, output_i, get(layer.weights.parameters, output_i, input_i) * get(input, batch_i, input_i));
                }
                set(output, batch_i, output_i, activation<typename DEVICE::SPEC::MATH, typename LAYER_SPEC::T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, batch_i, output_i)));
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    void forward(DEVICE& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, RNG& rng){
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;

        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI i = 0; i < LAYER_SPEC::OUTPUT_DIM; i++) {
                set(layer.pre_activations, batch_i, i, get(layer.biases.parameters, 0, i));
                for(TI j = 0; j < LAYER_SPEC::INPUT_DIM; j++) {
                    increment(layer.pre_activations, batch_i, i, get(layer.weights.parameters, i, j) * get(input, batch_i, j));
                }
                set(output, batch_i, i, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, i)));
            }
        }
    }
#endif

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG>
    void forward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, RNG& rng) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>);
        forward(device, (nn::layers::dense::LayerBackward<LAYER_SPEC>&)layer, input, layer.output, rng);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    void forward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, RNG& rng) {
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        // compile time warning if used
        forward(device, layer, input, rng);
        copy(device, device, layer.output, output);
    }

#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward_input(DEVICE& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input){
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        using SPEC = LAYER_SPEC;
        constexpr auto BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < SPEC::OUTPUT_DIM; output_i++) {
                typename SPEC::T d_pre_activation = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                for(TI input_j = 0; input_j < SPEC::INPUT_DIM; input_j++) {
                    if(output_i == 0){
                        set(d_input, batch_i, input_j, 0);
                    }
                    increment(d_input, batch_i, input_j, get(layer.weights.parameters, output_i, input_j) * d_pre_activation);
                }
            }
        }
    }


    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
    void backward_param(DEVICE& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;

        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                T d_pre_activation = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                increment(layer.biases.gradient, 0, output_i, d_pre_activation);
                for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                    increment(layer.weights.gradient, output_i, input_i, d_pre_activation * get(input, batch_i, input_i));
                }
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
    void backward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::T;
        using TI = typename DEVICE::index_t;

        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                T d_pre_activation = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                increment(layer.biases.gradient, 0, output_i, d_pre_activation);
                for(TI input_i = 0; input_i < INPUT_DIM; input_i++){
                    if(output_i == 0){
                        set(d_input, batch_i, input_i, 0);
                    }
                    increment(d_input, batch_i, input_i, get(layer.weights.parameters, output_i, input_i) * d_pre_activation);
                    increment(layer.weights.gradient, output_i, input_i, d_pre_activation * get(input, batch_i, input_i));
                }
            }
        }
    }

#endif
    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        zero_gradient(device, layer.weights);
        zero_gradient(device, layer.biases);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, OPTIMIZER& optimizer){
        update(device, layer.weights, optimizer);
        update(device, layer.biases, optimizer);
    }

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, layer.weights, optimizer);
        _reset_optimizer_state(device, layer.biases, optimizer);
    }

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::layers::dense::Layer<SOURCE_SPEC>& source, nn::layers::dense::Layer<TARGET_SPEC>& target){
        static_assert(nn::layers::dense::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.weights, target.weights);
        copy(source_device, target_device, source.biases, target.biases);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::dense::LayerBackward<SOURCE_SPEC>& source, nn::layers::dense::LayerBackward<TARGET_SPEC>& target){
        static_assert(nn::layers::dense::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, (nn::layers::dense::Layer<SOURCE_SPEC>&) source, (nn::layers::dense::Layer<TARGET_SPEC>&) target);
        copy(source_device, target_device, source.pre_activations, target.pre_activations);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::dense::LayerBackwardGradient<SOURCE_SPEC>& source, nn::layers::dense::LayerBackwardGradient<TARGET_SPEC>& target){
        static_assert(nn::layers::dense::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, (nn::layers::dense::LayerBackward<SOURCE_SPEC>&)source, (nn::layers::dense::LayerBackward<TARGET_SPEC>&)target);
        copy(source_device, target_device, source.output, target.output);

    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::Layer<SPEC_1>* l1, const rl_tools::nn::layers::dense::Layer<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = 0;
        acc += abs_diff(device, l1->weights, l2->weights);
        acc += abs_diff(device, l1->biases, l2->biases);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::Layer<SPEC_1>& l1, const rl_tools::nn::layers::dense::Layer<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(device, &l1, &l2);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackward<SPEC_1>* l1, const rl_tools::nn::layers::dense::LayerBackward<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = abs_diff(device, (rl_tools::nn::layers::dense::Layer<SPEC_1>*) l1, (rl_tools::nn::layers::dense::Layer<SPEC_2>*) l2);
        acc += abs_diff(device, l1->pre_activations, l2->pre_activations);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackward<SPEC_1>& l1, const rl_tools::nn::layers::dense::LayerBackward<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(device, &l1, &l2);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC_1>* l1, const rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = abs_diff(device, (rl_tools::nn::layers::dense::LayerBackward<SPEC_1>*) l1, (rl_tools::nn::layers::dense::LayerBackward<SPEC_2>*) l2);
        acc += abs_diff(device, l1->output, l2->output);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC_1>& l1, const rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(device, &l1, &l2);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::dense::LayerBackward<SPEC>* l) {
        set_all(device, l->pre_activations, 0);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::dense::LayerBackward<SPEC>& l) {
        reset_forward_state(device, (rl_tools::nn::layers::dense::Layer<SPEC>*) l);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC>* l) {
        reset_forward_state(device, (rl_tools::nn::layers::dense::LayerBackward<SPEC>*) l);
        set_all(device, l->output, 0);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC>& l) {
        reset_forward_state(device, &l);
    }
    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const rl_tools::nn::layers::dense::Layer<SPEC>& l) {
        return is_nan(device, l.weights) || is_nan(device, l.biases);
    }
    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackward<SPEC>& l) {
        return
                is_nan(device, (rl_tools::nn::layers::dense::Layer<SPEC>&) l) ||
                is_nan(device, l.pre_activations);
    }
    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const rl_tools::nn::layers::dense::LayerBackwardGradient<SPEC>& l) {
        return
            is_nan(device, (rl_tools::nn::layers::dense::LayerBackward<SPEC>&) l) ||
            is_nan(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
