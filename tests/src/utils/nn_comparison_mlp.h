#ifndef LAYER_IN_C_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H
#define LAYER_IN_C_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H

#include "nn_comparison.h"

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff(const layer_in_c::nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += abs_diff(n1.input_layer, n2.input_layer);
    for(lic::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += abs_diff(n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += abs_diff(n1.output_layer, n2.output_layer);
    return acc;
}
template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_grad(const layer_in_c::nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += abs_diff(n1.input_layer, n2.input_layer);
    for(lic::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += abs_diff(n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += abs_diff(n1.output_layer, n2.output_layer);
    return acc;
}

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_adam(const layer_in_c::nn_models::mlp::NeuralNetworkAdam<DEVICE, SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetworkAdam<DEVICE, SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += abs_diff(n1.input_layer, n2.input_layer);
    for(lic::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += abs_diff(n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += abs_diff(n1.output_layer, n2.output_layer);
    return acc;
}

#endif
