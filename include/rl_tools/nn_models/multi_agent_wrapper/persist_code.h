#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_CODE_H
#include "../../containers/persist_code.h"
#include "../../persist/code.h"
#include "../../nn/persist_code.h"
#include "model.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& model, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0) {
        return save_code_split(device, model.content, name, const_declaration, indent);
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& network, std::string name, bool const_declaration = false, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, network, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
