#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_PERSIST_H
#include "layer.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dynamic_conv2d::LayerForward<SPEC>& layer, GROUP& group) {
        set_attribute(device, group, "type", "dynamic_conv2d");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::dynamic_conv2d::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::dynamic_conv2d::LayerBackward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::dynamic_conv2d::LayerForward<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename GROUP>
    void save(DEVICE& device, nn::layers::dynamic_conv2d::State& state, GROUP& group) {}
    template<typename DEVICE, typename GROUP>
    bool load(DEVICE& device, nn::layers::dynamic_conv2d::State& state, GROUP& group) { return true; }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
