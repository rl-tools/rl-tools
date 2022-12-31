#ifndef LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#define LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#include <layer_in_c/nn/persist.h>
#include "network.h"

#include <highfive/H5Group.hpp>
namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void save(nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>& network, HighFive::Group group) {
        save(network.input_layer, group.createGroup("input_layer"));
        save(network.hidden_layer_0, group.createGroup("hidden_layer_0"));
        save(network.output_layer, group.createGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void save(nn_models::three_layer_fc::NeuralNetworkAdam<DEVICE, SPEC>& network, HighFive::Group group) {
        save(static_cast<nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>&>(network), group);
        std::vector<typeof(network.age)> age;
        age.push_back(network.age);
        group.createDataSet("age", age);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>& network, HighFive::Group group){
        load(network.input_layer, group.getGroup("input_layer"));
        load(network.hidden_layer_0, group.getGroup("hidden_layer_0"));
        load(network.output_layer, group.getGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void load(nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        load(network, file.getGroup("three_layer_fc"));
    }
}
#endif