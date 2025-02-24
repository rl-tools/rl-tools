#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_PERSIST_H
#include "../../../containers/matrix/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer, HighFive::Group group){
        group.createAttribute<std::string>("type", "sample_and_squash");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::sample_and_squash::LayerForward<SPEC>&)layer, group);
        save(device, layer.pre_squashing, group, "pre_squashing");
        save(device, layer.noise, group, "noise");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::sample_and_squash::LayerBackward<SPEC>&)layer, group);
        save(device, layer.log_probabilities, group, "log_probabilities");
        save(device, layer.log_alpha, group.createGroup("log_alpha"));
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer, HighFive::Group group) {
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::sample_and_squash::LayerForward<SPEC>&)layer, group);
        load(device, layer.pre_squashing, group, "pre_squashing");
        load(device, layer.noise, group, "noise");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::sample_and_squash::LayerBackward<SPEC>&)layer, group);
        load(device, layer.log_probabilities, group, "log_probabilities");
        load(device, layer.log_alpha, group.getGroup("output"));
        load(device, layer.output, group, "output");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
