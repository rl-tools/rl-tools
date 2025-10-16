#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_EMBEDDING_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_EMBEDDING_PERSIST_H

#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::embedding::LayerForward<SPEC>& layer, HighFive::Group group) {
        // todo: forward implementation to Parameter struct
        save(device, layer.weights, group.createGroup("weights"));
        group.createAttribute<std::string>("type", "embedding");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::embedding::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::embedding::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::embedding::LayerGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::embedding::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::embedding::LayerForward<SPEC>& layer, HighFive::Group group) {
        load(device, layer.weights, group.getGroup("weights"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::embedding::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::embedding::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::embedding::LayerGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::embedding::LayerBackward<SPEC>&)layer, group);
        if(group.exist("output")){
            load(device, layer.output, group, "output");
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
