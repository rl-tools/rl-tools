#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_CODE_H



#include <string>
#include <sstream>
#include "../../persist/code.h"
#include "model.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& module, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        ss_header << "#include <rl_tools/nn_models/multi_agent_wrapper/model.h>\n";
        std::stringstream ss;
        ss << ind << "namespace " << name << " {\n";
        auto content = save_code_split(device, module.content, "content", const_declaration, indent+1);
        ss_header << content.header;
        ss << content.body;
        std::string T_string = containers::persist::get_type_string<T>();
        std::string TI_string = containers::persist::get_type_string<TI>();
        ss << "\n";
        ss << ind << "    using SPEC = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::multi_agent_wrapper::SpecificationFixedModule<";
        ss << T_string << ", ";
        ss << TI_string << ", ";
        ss << std::to_string(SPEC::N_AGENTS) << ", ";
        ss << "content::MODEL" << ", ";
        ss << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag";
        ss << ">; \n";
        ss << ind << "    template <typename CAPABILITY>" << "\n";
        ss << ind << "    using TEMPLATE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::multi_agent_wrapper::Module<CAPABILITY, SPEC>; \n";
        ss << ind << "    using CAPABILITY = " << to_string(typename SPEC::CAPABILITY{}) << "; \n";
        ss << ind << "    using MODEL = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::multi_agent_wrapper::Module<CAPABILITY, SPEC>; \n";
        std::stringstream ss_initializer_list;
        ss_initializer_list << "{content::module}";
        ss << ind << "    " << (const_declaration ? "const " : "") << "MODEL module = " << ss_initializer_list.str() << ";\n";
        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
//    template<typename DEVICE, typename SPEC>
//    persist::Code save_code_split(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<SPEC>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
//        return save_code_split(device, static_cast<nn_models::mlp::NeuralNetworkForward<SPEC>&>(network), name, const_declaration, indent);
//    }
//    template<typename DEVICE, typename SPEC>
//    persist::Code save_code_split(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<SPEC>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
//        return save_code_split(device, static_cast<nn_models::mlp::NeuralNetworkBackward<SPEC>&>(network), name, const_declaration, indent);
//    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC>& module, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, module, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
