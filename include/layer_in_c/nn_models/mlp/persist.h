#ifndef LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#define LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#include <highfive/H5Group.hpp>
#include "network.h"
#include <layer_in_c/nn/persist.h>
namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void save(nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>& network, HighFive::Group group) {
        save(network.layer_1, group.createGroup("layer_1"));
        save(network.layer_2, group.createGroup("layer_2"));
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
        load(network.layer_1, group.getGroup("layer_1"));
        load(network.layer_2, group.getGroup("layer_2"));
        load(network.output_layer, group.getGroup("output_layer"));
    }
    template<typename DEVICE, typename SPEC>
    void load(nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        load(network, file.getGroup("three_layer_fc"));
    }
}
#endif