#ifndef LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CPU_H
#define LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_CPU_H

#include <layer_in_c/nn_models/mlp/network.h>

#include <layer_in_c/nn/operations_cpu.h>
#include "operations_generic.h"
// Specializations
namespace layer_in_c{
    template<typename SPEC, typename RNG>
    FUNCTION_PLACEMENT void init_weights(nn_models::mlp::NeuralNetwork<devices::CPU, SPEC>& network, RNG& rng) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        init_kaiming(network.input_layer, rng);
        for (index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            init_kaiming(network.hidden_layers[layer_i], rng);
        }
        init_kaiming(network.output_layer, rng);
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetwork<devices::CPU, TARGET_SPEC>& target, nn_models::mlp::NeuralNetwork<devices::CPU, SOURCE_SPEC>& source) {
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "Cannot copy networks with different structure.");
        target = source;
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetwork<devices::CPU, TARGET_SPEC>& target, nn_models::mlp::NeuralNetwork<typename devices::Generic, SOURCE_SPEC>& source) {
        target = source;
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetwork<typename devices::Generic, TARGET_SPEC>& target, nn_models::mlp::NeuralNetwork<devices::CPU, SOURCE_SPEC>& source) {
        target = source;
    }
}

// Fallback to generic implementations
#endif
