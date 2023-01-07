#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include "layer.h"
#include <layer_in_c/utils/persist.h>
#include <iostream>
namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        auto weights = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.weights);
        auto biases = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.biases);
        group.createDataSet("weights", weights);
        group.createDataSet("biases" , biases);
    }
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, HighFive::Group group) {
        save((nn::layers::dense::Layer<DEVICE, SPEC>&)layer, group);
        auto weights = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.pre_activations);
        group.createDataSet("pre_activations", weights);
    }
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackward<DEVICE, SPEC>&)layer, group);
        auto output = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.output);
        auto d_weights = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights);
        auto d_biases = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases);
        group.createDataSet("output", output);
        group.createDataSet("d_weights", d_weights);
        group.createDataSet("d_biases" , d_biases);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(nn::layers::dense::LayerBackwardSGD<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
        auto d_weights_first_order_moment  = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights_first_order_moment);
        auto d_weights_second_order_moment = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights_second_order_moment);
        auto d_biases_first_order_moment   = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases_first_order_moment);
        auto d_biases_second_order_moment  = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases_second_order_moment);
        group.createDataSet("d_weights_first_order_moment", d_weights_first_order_moment);
        group.createDataSet("d_weights_second_order_moment", d_weights_second_order_moment);
        group.createDataSet("d_biases_first_order_moment" , d_biases_first_order_moment);
        group.createDataSet("d_biases_second_order_moment" , d_biases_second_order_moment);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        auto weights_dataset = group.getDataSet("weights");
        auto weights_dims = weights_dataset.getDimensions();
        assert(weights_dims[0] == SPEC::OUTPUT_DIM);
        assert(weights_dims[1] == SPEC::INPUT_DIM);
        weights_dataset.read(layer.weights);

        auto biases_dataset = group.getDataSet("biases");
        auto biases_dims = biases_dataset.getDimensions();
        assert(biases_dims[0] == SPEC::OUTPUT_DIM);
        biases_dataset.read(layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, HighFive::Group group) {
        load((nn::layers::dense::Layer<DEVICE, SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackward<DEVICE, SPEC>&)layer, group);
        auto d_weights_dataset = group.getDataSet("d_weights");
        auto d_weights_dims = d_weights_dataset.getDimensions();
        assert(d_weights_dims[0] == SPEC::OUTPUT_DIM);
        assert(d_weights_dims[1] == SPEC::INPUT_DIM);
        d_weights_dataset.read(layer.d_weights);

        auto d_biases_dataset = group.getDataSet("d_biases");
        auto d_biases_dims = d_biases_dataset.getDimensions();
        assert(d_biases_dims[0] == SPEC::OUTPUT_DIM);
        d_biases_dataset.read(layer.d_biases);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(nn::layers::dense::LayerBackwardSGD<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)(layer), group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
        if(group.exist("d_biases_first_order_moment")) {
            auto d_weights_first_order_moment_dataset = group.getDataSet("d_weights_first_order_moment");
            auto d_weights_first_order_moment_dims = d_weights_first_order_moment_dataset.getDimensions();
            assert(d_weights_first_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            assert(d_weights_first_order_moment_dims[1] == SPEC::INPUT_DIM);
            d_weights_first_order_moment_dataset.read(layer.d_weights_first_order_moment);

            auto d_weights_second_order_moment_dataset = group.getDataSet("d_weights_second_order_moment");
            auto d_weights_second_order_moment_dims = d_weights_second_order_moment_dataset.getDimensions();
            assert(d_weights_second_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            assert(d_weights_second_order_moment_dims[1] == SPEC::INPUT_DIM);
            d_weights_second_order_moment_dataset.read(layer.d_weights_second_order_moment);

            auto d_biases_first_order_moment_dataset = group.getDataSet("d_biases_first_order_moment");
            auto d_biases_first_order_moment_dims = d_biases_first_order_moment_dataset.getDimensions();
            assert(d_biases_first_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            d_biases_first_order_moment_dataset.read(layer.d_biases_first_order_moment);

            auto d_biases_second_order_moment_dataset = group.getDataSet("d_biases_second_order_moment");
            auto d_biases_second_order_moment_dims = d_biases_second_order_moment_dataset.getDimensions();
            assert(d_biases_second_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            d_biases_second_order_moment_dataset.read(layer.d_biases_second_order_moment);
        }
        else{
            std::cout << "Warning: Adam state not found. Initializing with zeros." << std::endl;
            for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
                for(index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                    layer.d_weights_first_order_moment[i][j] = 0;
                    layer.d_weights_second_order_moment[i][j] = 0;
                }
                layer.d_biases_first_order_moment[i] = 0;
                layer.d_biases_second_order_moment[i] = 0;
            }
        }
    }
}
#endif
