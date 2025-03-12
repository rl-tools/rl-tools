#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_PERSIST_H
#include "../../../version.h"
#include "../../../containers/matrix/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
#include "persist_common.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerForward<SPEC>& layer, HighFive::Group group) {
        // todo: forward implementation to Parameter struct
        save(device, layer.weights, group.createGroup("weights"));
        save(device, layer.biases, group.createGroup("biases"));
        group.createAttribute<std::string>("activation_function", nn::layers::dense::persist::get_activation_function_string_short<SPEC::CONFIG::ACTIVATION_FUNCTION>());
        group.createAttribute<std::string>("type", "dense");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerForward<SPEC>&)layer, group);
        save(device, layer.pre_activations, group, "pre_activations");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::dense::LayerGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerForward<SPEC>& layer, HighFive::Group group) {
        load(device, layer.weights, group.getGroup("weights"));
        load(device, layer.biases, group.getGroup("biases"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerForward<SPEC>&)layer, group);
        if(group.exist("pre_activations")){
            load(device, layer.pre_activations, group, "pre_activations");
        }
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::dense::LayerGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        if(group.exist("output")){
            load(device, layer.output, group, "output");
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
