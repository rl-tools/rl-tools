#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONV2D_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONV2D_PERSIST_H
#include "../../../version.h"
#include "layer.h"
#include "../../parameters/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer, GROUP& group) {
        auto weights_group = create_group(device, group, "weights");
        auto biases_group = create_group(device, group, "biases");
        save(device, layer.weights, weights_group);
        save(device, layer.biases, biases_group);
        set_attribute(device, group, "type", "conv2d");
        set_attribute(device, group, "output_channels", std::to_string(SPEC::OUTPUT_CHANNELS));
        set_attribute(device, group, "input_channels", std::to_string(SPEC::INPUT_CHANNELS));
        set_attribute(device, group, "kernel_height", std::to_string(SPEC::KERNEL_HEIGHT));
        set_attribute(device, group, "kernel_width", std::to_string(SPEC::KERNEL_WIDTH));
        set_attribute(device, group, "stride_h", std::to_string(SPEC::STRIDE_H));
        set_attribute(device, group, "stride_w", std::to_string(SPEC::STRIDE_W));
        set_attribute(device, group, "padding_h", std::to_string(SPEC::PADDING_H));
        set_attribute(device, group, "padding_w", std::to_string(SPEC::PADDING_W));
        write_attributes(device, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::conv2d::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::conv2d::LayerForward<SPEC>&)layer, group);
        save(device, layer.pre_activations, group, "pre_activations");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::conv2d::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer, GROUP& group) {
        auto weights_group = get_group(device, group, "weights");
        auto biases_group = get_group(device, group, "biases");
        bool success = load(device, layer.weights, weights_group);
        success &= load(device, layer.biases, biases_group);
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::conv2d::LayerBackward<SPEC>& layer, GROUP& group) {
        bool success = load(device, (nn::layers::conv2d::LayerForward<SPEC>&)layer, group);
        if(group_exists(device, group, "pre_activations")){
            success &= load(device, layer.pre_activations, group, "pre_activations");
        }
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer, GROUP& group) {
        bool success = load(device, (nn::layers::conv2d::LayerBackward<SPEC>&)layer, group);
        if(group_exists(device, group, "output")){
            success &= load(device, layer.output, group, "output");
        }
        return success;
    }
    template<typename DEVICE, typename GROUP>
    void save(DEVICE& device, nn::layers::conv2d::State& state, GROUP& group) {}
    template<typename DEVICE, typename GROUP>
    bool load(DEVICE& device, nn::layers::conv2d::State& state, GROUP& group) { return true; }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
