#ifndef LAYER_IN_C_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H
#define LAYER_IN_C_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H

#include "nn_comparison.h"

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff(DEVICE& device, const layer_in_c::nn_models::mlp::NeuralNetwork<SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetwork<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += lic::abs_diff(device, n1.input_layer, n2.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += lic::abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += lic::abs_diff(device, n1.output_layer, n2.output_layer);
    return acc;
}
template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_grad(DEVICE& device, const layer_in_c::nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    using GradNetworkSpec = layer_in_c::nn_models::mlp::BackwardGradientSpecification<typename SPEC::STRUCTURE_SPEC>;
    using GradNetworkType = layer_in_c::nn_models::mlp::NeuralNetworkBackwardGradient<GradNetworkSpec>;
    GradNetworkType n1g;
    lic::malloc(device, n1g);
    lic::copy(device, n1g, n1);
    GradNetworkType n2g;
    lic::malloc(device, n2g);
    lic::copy(device, n2g, n2);
    T acc = 0;
    acc += lic::abs_diff(device, n1g.input_layer, n2g.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += lic::abs_diff(device, n1g.hidden_layers[layer_i], n2g.hidden_layers[layer_i]);
    }
    acc += lic::abs_diff(device, n1g.output_layer, n2g.output_layer);
    lic::free(device, n1g);
    lic::free(device, n2g);
    return acc;
}

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_adam(DEVICE& device, const layer_in_c::nn_models::mlp::NeuralNetworkAdam<SPEC>& n1, const layer_in_c::nn_models::mlp::NeuralNetworkAdam<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += lic::abs_diff(device, n1.input_layer, n2.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += lic::abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += lic::abs_diff(device, n1.output_layer, n2.output_layer);
    return acc;
}

#endif
