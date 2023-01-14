#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include "layer.h"
#include <layer_in_c/utils/persist.h>
#include <iostream>
namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        auto weights = utils::persist::array_conversion::matrix_to_std_vector(device, layer.weights);
        auto biases  = utils::persist::array_conversion::matrix_to_std_vector(device, layer.biases);
        group.createDataSet("weights", weights);
        group.createDataSet("biases" , biases);
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::Layer<SPEC>&)layer, group);
        auto weights = utils::persist::array_conversion::matrix_to_std_vector(device, layer.pre_activations);
        group.createDataSet("pre_activations", weights);
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        auto output    = utils::persist::array_conversion::matrix_to_std_vector(device, layer.output);
        auto d_weights = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_weights);
        auto d_biases  = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_biases);
        group.createDataSet("output", output);
        group.createDataSet("d_weights", d_weights);
        group.createDataSet("d_biases" , d_biases);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(DEVICE& device, nn::layers::dense::LayerBackwardSGD<SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerBackwardGradient<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerBackwardGradient<SPEC>&)layer, group);
        auto d_weights_first_order_moment  = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_weights_first_order_moment);
        auto d_weights_second_order_moment = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_weights_second_order_moment);
        auto d_biases_first_order_moment   = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_biases_first_order_moment);
        auto d_biases_second_order_moment  = utils::persist::array_conversion::matrix_to_std_vector(device, layer.d_biases_second_order_moment);
        group.createDataSet("d_weights_first_order_moment", d_weights_first_order_moment);
        group.createDataSet("d_weights_second_order_moment", d_weights_second_order_moment);
        group.createDataSet("d_biases_first_order_moment" , d_biases_first_order_moment);
        group.createDataSet("d_biases_second_order_moment" , d_biases_second_order_moment);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        auto weights_dataset = group.getDataSet("weights");
        auto weights_dims = weights_dataset.getDimensions();
        assert(weights_dims[0] == SPEC::OUTPUT_DIM);
        assert(weights_dims[1] == SPEC::INPUT_DIM);
        weights_dataset.read(layer.weights.data);

        auto biases_dataset = group.getDataSet("biases");
        auto biases_dims = biases_dataset.getDimensions();
        assert(biases_dims[0] == SPEC::OUTPUT_DIM);
        biases_dataset.read(layer.biases.data);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::Layer<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        auto d_weights_dataset = group.getDataSet("d_weights");
        auto d_weights_dims = d_weights_dataset.getDimensions();
        assert(d_weights_dims[0] == SPEC::OUTPUT_DIM);
        assert(d_weights_dims[1] == SPEC::INPUT_DIM);
        d_weights_dataset.read(layer.d_weights.data);

        auto d_biases_dataset = group.getDataSet("d_biases");
        auto d_biases_dims = d_biases_dataset.getDimensions();
        assert(d_biases_dims[0] == SPEC::OUTPUT_DIM);
        d_biases_dataset.read(layer.d_biases.data);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(DEVICE& device, nn::layers::dense::LayerBackwardSGD<SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackwardGradient<SPEC>&)(layer), group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackwardGradient<SPEC>&)layer, group);
        if(group.exist("d_biases_first_order_moment")) {
            auto d_weights_first_order_moment_dataset = group.getDataSet("d_weights_first_order_moment");
            auto d_weights_first_order_moment_dims = d_weights_first_order_moment_dataset.getDimensions();
            assert(d_weights_first_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            assert(d_weights_first_order_moment_dims[1] == SPEC::INPUT_DIM);
            d_weights_first_order_moment_dataset.read(layer.d_weights_first_order_moment.data);

            auto d_weights_second_order_moment_dataset = group.getDataSet("d_weights_second_order_moment");
            auto d_weights_second_order_moment_dims = d_weights_second_order_moment_dataset.getDimensions();
            assert(d_weights_second_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            assert(d_weights_second_order_moment_dims[1] == SPEC::INPUT_DIM);
            d_weights_second_order_moment_dataset.read(layer.d_weights_second_order_moment.data);

            auto d_biases_first_order_moment_dataset = group.getDataSet("d_biases_first_order_moment");
            auto d_biases_first_order_moment_dims = d_biases_first_order_moment_dataset.getDimensions();
            assert(d_biases_first_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            d_biases_first_order_moment_dataset.read(layer.d_biases_first_order_moment.data);

            auto d_biases_second_order_moment_dataset = group.getDataSet("d_biases_second_order_moment");
            auto d_biases_second_order_moment_dims = d_biases_second_order_moment_dataset.getDimensions();
            assert(d_biases_second_order_moment_dims[0] == SPEC::OUTPUT_DIM);
            d_biases_second_order_moment_dataset.read(layer.d_biases_second_order_moment.data);
        }
        else{
            std::cout << "Warning: Adam state not found. Initializing with zeros." << std::endl;
            for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
                for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                    layer.d_weights_first_order_moment.data[i * SPEC::INPUT_DIM + j] = 0;
                    layer.d_weights_second_order_moment.data[i * SPEC::INPUT_DIM + j] = 0;
                }
                layer.d_biases_first_order_moment.data[i] = 0;
                layer.d_biases_second_order_moment.data[i] = 0;
            }
        }
    }
}
#endif
