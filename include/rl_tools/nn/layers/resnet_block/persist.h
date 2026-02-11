#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_RESNET_BLOCK_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_RESNET_BLOCK_PERSIST_H
#include "../../../version.h"
#include "layer.h"
#include "../conv2d/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // ======================== DownsampleStorage save / load ========================
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, GROUP& group) {}
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, GROUP& group) { return true; }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, GROUP& group) {
        auto ds_group = create_group(device, group, "downsample");
        save(device, ds.conv, ds_group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, GROUP& group) {
        if(group_exists(device, group, "downsample")){
            auto ds_group = get_group(device, group, "downsample");
            return load(device, ds.conv, ds_group);
        }
        return true;
    }

    // ======================== Layer save / load ========================
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::LayerForward<SPEC>& layer, GROUP& group) {
        set_attribute(device, group, "type", "resnet_block");
        auto conv1_group = create_group(device, group, "conv1");
        save(device, layer.conv1, conv1_group);
        auto conv2_group = create_group(device, group, "conv2");
        save(device, layer.conv2, conv2_group);
        save(device, layer.downsample, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::resnet_block::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::resnet_block::LayerBackward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::LayerForward<SPEC>& layer, GROUP& group) {
        auto conv1_group = get_group(device, group, "conv1");
        bool success = load(device, layer.conv1, conv1_group);
        auto conv2_group = get_group(device, group, "conv2");
        success &= load(device, layer.conv2, conv2_group);
        success &= load(device, layer.downsample, group);
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::LayerBackward<SPEC>& layer, GROUP& group) {
        return load(device, (nn::layers::resnet_block::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer, GROUP& group) {
        return load(device, (nn::layers::resnet_block::LayerBackward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename GROUP>
    void save(DEVICE& device, nn::layers::resnet_block::State& state, GROUP& group) {}
    template<typename DEVICE, typename GROUP>
    bool load(DEVICE& device, nn::layers::resnet_block::State& state, GROUP& group) { return true; }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
