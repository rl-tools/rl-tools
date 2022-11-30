#ifndef LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#define LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_PERSIST
#include <highfive/H5File.hpp>
namespace layer_in_c::nn_models::three_layer_fc {
    template<typename SPEC>
    void save(three_layer_fc::NeuralNetwork<SPEC>& network, std::string file_path) {
        auto file = HighFive::File(file_path, HighFive::File::Overwrite);
        save(network, file.createGroup("three_layer_fc"));
    }
    template<typename SPEC>
    void save(three_layer_fc::NeuralNetwork<SPEC>& network, HighFive::Group group) {

    }
    template<typename SPEC>
    void load(three_layer_fc::NeuralNetwork<SPEC>& network) {
    }
}
#endif