#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include "layer.h"
#include <layer_in_c/utils/persist.h>
namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        HighFive::DataSpace weight_data_space(SPEC::OUTPUT_DIM, SPEC::INPUT_DIM);
        auto weights = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.weights);
        auto biases = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.biases);
        group.createDataSet("weights", weights);
        group.createDataSet("biases" , biases);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        std::vector<std::vector<typename SPEC::T>> weights;
        std::vector<typename SPEC::T> biases;
        group.getDataSet("weights").read(weights);
        group.getDataSet("biases").read(biases);
        utils::persist::array_conversion::std_vector_to_matrix<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.weights, weights);
        utils::persist::array_conversion::std_vector_to_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.biases, biases);
    }
}
#endif
