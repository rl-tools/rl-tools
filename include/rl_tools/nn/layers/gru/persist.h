#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_PERSIST_H
#include "../../../containers/matrix/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::gru::LayerForward<SPEC>& layer, HighFive::Group group) {
        // todo: forward implementation to Parameter struct
        save(device, layer.weights_input, group.createGroup("weights_input"));
        save(device, layer.biases_input, group.createGroup("biases_input"));
        save(device, layer.weights_hidden, group.createGroup("weights_hidden"));
        save(device, layer.biases_hidden, group.createGroup("biases_hidden"));
        save(device, layer.initial_hidden_state, group.createGroup("initial_hidden_state"));
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::gru::LayerBackward<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::gru::LayerForward<SPEC>&)layer, group);
        save(device, layer.post_activation, group, "post_activation");
        save(device, layer.n_pre_pre_activation, group, "n_pre_pre_activation");
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::gru::LayerGradient<SPEC>& layer, HighFive::Group group) {
        save(device, (nn::layers::gru::LayerBackward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::gru::LayerForward<SPEC>& layer, HighFive::Group group) {
        load(device, layer.weights_input, group.getGroup("weights_input"));
        load(device, layer.biases_input, group.getGroup("biases_input"));
        load(device, layer.weights_hidden, group.getGroup("weights_hidden"));
        load(device, layer.biases_hidden, group.getGroup("biases_hidden"));
        load(device, layer.initial_hidden_state, group.getGroup("initial_hidden_state"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::gru::LayerBackward<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::gru::LayerForward<SPEC>&)layer, group);
        load(device, layer.post_activation, group, "post_activation");
        load(device, layer.n_pre_pre_activation, group, "n_pre_pre_activation");
        load(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::gru::LayerGradient<SPEC>& layer, HighFive::Group group) {
        load(device, (nn::layers::gru::LayerBackward<SPEC>&)layer, group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
