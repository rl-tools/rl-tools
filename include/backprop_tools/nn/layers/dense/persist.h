#include "../../../version.h"
#if !defined(BACKPROP_TOOLS_NN_LAYERS_DENSE_PERSIST_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_NN_LAYERS_DENSE_PERSIST_H
#include "../../../containers/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        // todo: forward implementation to Parameter struct
        save(device, layer.weights, group.createGroup("weights"));
        save(device, layer.biases, group.createGroup("biases"));
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
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, HighFive::Group group) {
        load(device, layer.weights, group.getGroup("weights"));
        load(device, layer.biases, group.getGroup("biases"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::Layer<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
