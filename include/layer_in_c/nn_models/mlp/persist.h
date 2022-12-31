#ifndef LAYER_IN_C_NN_MODELS_MLP_PERSIST_H
#define LAYER_IN_C_NN_MODELS_MLP_PERSIST_H
#include <layer_in_c/nn/persist.h>
#include "network.h"

#include <highfive/H5Group.hpp>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void save(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network, HighFive::Group group) {
        using NetworkType = typename std::remove_reference<decltype(network)>::type;
        save(network.input_layer, group.createGroup("input_layer"));
        for(size_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            save(network.hidden_layers[layer_i], group.createGroup("hidden_layer_" + std::to_string(layer_i)));
        }
        save(network.output_layer, group.createGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void save(nn_models::mlp::NeuralNetworkAdam<DEVICE, SPEC>& network, HighFive::Group group) {
        save(static_cast<nn_models::mlp::NeuralNetwork<DEVICE, SPEC>&>(network), group);
        std::vector<typeof(network.age)> age;
        age.push_back(network.age);
        group.createDataSet("age", age);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network, HighFive::Group group){
        using NetworkType = typename std::remove_reference<decltype(network)>::type;
        load(network.input_layer, group.getGroup("input_layer"));
        for(size_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            load(network.hidden_layers[layer_i], group.getGroup("hidden_layer_" + std::to_string(layer_i)));
        }
        load(network.output_layer, group.getGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void load(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        load(network, file.getGroup("mlp"));
    }
}
#endif
