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
        if constexpr(SPEC::NORMALIZATION != nn::layers::conv2d::Normalization::NONE) {
            auto gamma_group = create_group(device, group, "gamma");
            auto beta_group = create_group(device, group, "beta");
            save(device, layer.norm.gamma, gamma_group);
            save(device, layer.norm.beta, beta_group);
            if constexpr(SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                save(device, layer.norm.running_mean.parameters, group, "running_mean");
                save(device, layer.norm.running_var.parameters, group, "running_var");
            }
        }
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
        if constexpr(SPEC::NORMALIZATION != nn::layers::conv2d::Normalization::NONE) {
            if(group_exists(device, group, "gamma") && group_exists(device, group, "beta")) {
                auto gamma_group = get_group(device, group, "gamma");
                auto beta_group = get_group(device, group, "beta");
                success &= load(device, layer.norm.gamma, gamma_group);
                success &= load(device, layer.norm.beta, beta_group);
            }
            if constexpr(SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                if(group_exists(device, group, "running_mean") || dataset_exists(device, group, "running_mean")) {
                    success &= load(device, layer.norm.running_mean.parameters, group, "running_mean");
                    success &= load(device, layer.norm.running_var.parameters, group, "running_var");
                }
            }
        }
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
