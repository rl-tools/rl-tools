#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_PERSIST_H
#include "../../../version.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
#include "persist_common.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dense::LayerForward<SPEC>& layer, GROUP& group) {
        // todo: forward implementation to Parameter struct
        auto weights_group = create_group(device, group, "weights");
        auto biases_group = create_group(device, group, "biases");
        save(device, layer.weights, weights_group);
        save(device, layer.biases, biases_group);
        set_attribute(device, group, "activation_function", nn::layers::dense::persist::get_activation_function_string_short<SPEC::CONFIG::ACTIVATION_FUNCTION>());
        set_attribute(device, group, "type", "dense");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::dense::LayerForward<SPEC>&)layer, group);
        save(device, layer.pre_activations, group, "pre_activations");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dense::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, nn::layers::dense::LayerForward<SPEC>& layer, GROUP& group) {
        auto weights_group = get_group(device, group, "weights");
        auto biases_group = get_group(device, group, "biases");
        load(device, layer.weights, weights_group);
        load(device, layer.biases, biases_group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, GROUP& group) {
        load(device, (nn::layers::dense::LayerForward<SPEC>&)layer, group);
        if(group_exists(device, group, "pre_activations")){
            load(device, layer.pre_activations, group, "pre_activations");
        }
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, nn::layers::dense::LayerGradient<SPEC>& layer, GROUP& group) {
        load(device, (nn::layers::dense::LayerBackward<SPEC>&)layer, group);
        if(group_exists(device, group, "output")){
            load(device, layer.output, group, "output");
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
