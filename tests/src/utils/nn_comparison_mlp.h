#ifndef BACKPROP_TOOLS_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H
#define BACKPROP_TOOLS_TESTS_SRC_UTILS_NN_COMPARISON_MLP_H

#include "nn_comparison.h"

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff(DEVICE& device, const backprop_tools::nn_models::mlp::NeuralNetwork<SPEC>& n1, const backprop_tools::nn_models::mlp::NeuralNetwork<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += bpt::abs_diff(device, n1.input_layer, n2.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += bpt::abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += bpt::abs_diff(device, n1.output_layer, n2.output_layer);
    return acc;
}
template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_grad(DEVICE& device, const backprop_tools::nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& n1, const backprop_tools::nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    using GradNetworkSpec = backprop_tools::nn_models::mlp::BackwardGradientSpecification<typename SPEC::STRUCTURE_SPEC>;
    using GradNetworkType = backprop_tools::nn_models::mlp::NeuralNetworkBackwardGradient<GradNetworkSpec>;
    GradNetworkType n1g;
    bpt::malloc(device, n1g);
    bpt::copy(device, device, n1g, n1);
    bpt::reset_forward_state(device, n1g);
    GradNetworkType n2g;
    bpt::malloc(device, n2g);
    bpt::copy(device, device, n2g, n2);
    bpt::reset_forward_state(device, n2g);
    T acc = 0;
    acc += bpt::abs_diff(device, n1g.input_layer, n2g.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += bpt::abs_diff(device, n1g.hidden_layers[layer_i], n2g.hidden_layers[layer_i]);
    }
    acc += bpt::abs_diff(device, n1g.output_layer, n2g.output_layer);
    bpt::free(device, n1g);
    bpt::free(device, n2g);
    return acc;
}

template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff_adam(DEVICE& device, const backprop_tools::nn_models::mlp::NeuralNetworkAdam<SPEC>& n1, const backprop_tools::nn_models::mlp::NeuralNetworkAdam<SPEC>& n2) {
    using NetworkType = typename std::remove_reference<decltype(n1)>::type;
    typedef typename SPEC::T T;
    T acc = 0;
    acc += bpt::abs_diff(device, n1.input_layer, n2.input_layer);
    for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
        acc += bpt::abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
    }
    acc += bpt::abs_diff(device, n1.output_layer, n2.output_layer);
    return acc;
}

#endif
