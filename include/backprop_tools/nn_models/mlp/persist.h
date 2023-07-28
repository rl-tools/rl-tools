#ifndef BACKPROP_TOOLS_NN_MODELS_MLP_PERSIST_H
#define BACKPROP_TOOLS_NN_MODELS_MLP_PERSIST_H
#include <backprop_tools/nn/parameters/persist.h>
#include <backprop_tools/nn/optimizers/adam/persist.h>
#include <backprop_tools/nn/persist.h>
#include "network.h"

#include <highfive/H5Group.hpp>

namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, HighFive::Group group) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        save(device, network.input_layer, group.createGroup("input_layer"));
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            save(device, network.hidden_layers[layer_i], group.createGroup("hidden_layer_" + std::to_string(layer_i)));
        }
        save(device, network.output_layer, group.createGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn_models::mlp::NeuralNetworkAdam<SPEC>& network, HighFive::Group group) {
        save(device, (nn_models::mlp::NeuralNetwork<SPEC>&)network, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, HighFive::Group group){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        load(device, network.input_layer, group.getGroup("input_layer"));
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            load(device, network.hidden_layers[layer_i], group.getGroup("hidden_layer_" + std::to_string(layer_i)));
        }
        load(device, network.output_layer, group.getGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        load(device, network, file.getGroup("mlp"));
    }
}
#endif
