#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "model.h"

#include <highfive/H5Group.hpp>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& model, HighFive::Group group) {
        group = group.createGroup("multi_agent_wrapper");
        save(device, model.content, group.createGroup("content"));
    }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& model, HighFive::Group group, typename DEVICE::index_t layer_i = 0) {
        group = group.getGroup("multi_agent_wrapper");
        load(device, model.content, group.getGroup("content"));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
