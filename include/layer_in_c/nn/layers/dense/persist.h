#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#include <layer_in_c/containers/persist.h>
#include "layer.h"
#include <layer_in_c/utils/persist.h>
#include <iostream>
namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        save(device, layer.weights, group, "weights");
        save(device, layer.biases, group, "biases");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::Layer<SPEC>&)layer, group);
        save(device, layer.pre_activations, group, "pre_activations");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
        save(device, layer.d_weights, group, "d_weights");
        save(device, layer.d_biases, group, "d_biases");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        load(device, layer.weights, group, "weights");
        load(device, layer.biases, group, "biases");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::Layer<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        load(device, layer.d_weights, group, "d_weights");
        load(device, layer.d_biases, group, "d_biases");
    }
}
#endif
