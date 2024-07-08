#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_STANDARDIZE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_STANDARDIZE_PERSIST_H
#include "../../../containers/matrix/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer, HighFive::Group group) {
        // todo: forward implementation to Parameter struct
        save(device, layer.mean, group.createGroup("mean"));
        save(device, layer.precision, group.createGroup("precision"));
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::standardize::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::standardize::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::standardize::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer, HighFive::Group group) {
        load(device, layer.mean, group.getGroup("mean"));
        load(device, layer.precision, group.getGroup("precision"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::standardize::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::standardize::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::standardize::LayerBackward<SPEC>&)layer, group);
        if(group.exist("output")){
            load(device, layer.output, group, "output");
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
