#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "network.h"
#include "../../utils/persist.h"
#include "../../persist/backends/hdf5/operations_cpu.h"
#include <highfive/H5File.hpp>

#include "../mlp/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, GROUP& group) {
        save(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        auto log_std_group = create_group(device, group, "log_std");
        save(device, network.log_std, log_std_group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, GROUP& group){
        load(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), group);
        auto log_std_group = get_group(device, group, "log_std");
        load(device, network.log_std, log_std_group);
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, std::string file_path){
        auto file = HighFive::File(file_path, HighFive::File::ReadOnly);
        auto mlp_group = file.getGroup("mlp");
        persist::backends::hdf5::Group<> group{mlp_group};
        load(device, network, group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
