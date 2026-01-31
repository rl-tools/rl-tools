#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "network.h"
#include "../mlp/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    void save(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& network, GROUP& group) {
        save(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        auto log_std_group = create_group(device, group, "log_std");
        save(device, network.log_std, log_std_group);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    void save(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>& network, GROUP& group){
        save(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>&>(network), group);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    void save(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network, GROUP& group){
        save(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>&>(network), group);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    bool load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& network, GROUP& group){
        bool success = load(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        auto log_std_group = get_group(device, group, "log_std");
        success &= load(device, network.log_std, log_std_group);
        return success;
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    bool load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>& network, GROUP& group){
        return load(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>&>(network), group);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename GROUP>
    bool load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network, GROUP& group){
        return load(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>&>(network), group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
