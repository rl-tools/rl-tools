#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "network.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& network, GROUP& group) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        set_attribute(device, group, "type", "mlp");
        set_attribute(device, group, "num_layers", std::to_string(SPEC::NUM_LAYERS).c_str());
        auto input_layer_group = create_group(device, group, "input_layer");
        save(device, network.input_layer, input_layer_group);
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            auto hidden_layer_group = create_group(device, group, "hidden_layer_" + std::to_string(layer_i));
            save(device, network.hidden_layers[layer_i], hidden_layer_group);
        }
        auto output_layer_group = create_group(device, group, "output_layer");
        save(device, network.output_layer, output_layer_group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& network, GROUP& group){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        auto input_layer_group = get_group(device, group, "input_layer");
        load(device, network.input_layer, input_layer_group);
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++) {
            auto hidden_layer_group = get_group(device, group, "hidden_layer_" + std::to_string(layer_i));
            load(device, network.hidden_layers[layer_i], hidden_layer_group);
        }
        auto output_layer_group = get_group(device, group, "output_layer");
        load(device, network.output_layer, output_layer_group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
