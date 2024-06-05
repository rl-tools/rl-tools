#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "network.h"

#include <highfive/H5Group.hpp>

#include "../mlp/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, HighFive::Group group) {
        save(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        save(device, network.log_std, group.createGroup("log_std"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, HighFive::Group group){
        load(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        load(device, network.log_std, group.createGroup("log_std"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        load(device, network, file.getGroup("mlp"));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
