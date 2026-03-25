#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_MAX_POOL2D_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_MAX_POOL2D_PERSIST_H
#include "../../../version.h"
#include "layer.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::max_pool2d::LayerForward<SPEC>& layer, GROUP& group) {
        set_attribute(device, group, "type", "max_pool2d");
        set_attribute(device, group, "kernel_height", std::to_string(SPEC::CONFIG::KERNEL_HEIGHT).c_str());
        set_attribute(device, group, "kernel_width", std::to_string(SPEC::CONFIG::KERNEL_WIDTH).c_str());
        set_attribute(device, group, "stride_h", std::to_string(SPEC::CONFIG::STRIDE_H).c_str());
        set_attribute(device, group, "stride_w", std::to_string(SPEC::CONFIG::STRIDE_W).c_str());
        set_attribute(device, group, "padding_h", std::to_string(SPEC::CONFIG::PADDING_H).c_str());
        set_attribute(device, group, "padding_w", std::to_string(SPEC::CONFIG::PADDING_W).c_str());
        write_attributes(device, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::max_pool2d::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::max_pool2d::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::max_pool2d::LayerBackward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::max_pool2d::LayerForward<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::max_pool2d::LayerBackward<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer, GROUP& group) { return true; }
    template<typename DEVICE, typename GROUP>
    void save(DEVICE& device, nn::layers::max_pool2d::State& state, GROUP& group) {}
    template<typename DEVICE, typename GROUP>
    bool load(DEVICE& device, nn::layers::max_pool2d::State& state, GROUP& group) { return true; }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
