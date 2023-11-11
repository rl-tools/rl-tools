#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_H
#include "../../containers/persist_code.h"
#include "../../persist/code.h"
#include "model.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::sequential::Module<SPEC>& model, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0, typename DEVICE::index_t layer_i = 0) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        std::stringstream ss;
        if(layer_i == 0){
            ss_header << "#include <rl_tools/nn_models/sequential/model.h>\n";
            ss << ind << "namespace " << name << " {\n";
        }
        auto layer_output = save_split(device, model.content, "layer_" + std::to_string(layer_i), const_declaration, indent+1);
        ss_header << layer_output.header;
        ss << layer_output.body;
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            auto downstream_output = save_code_split(device, model.next_module, name, const_declaration, indent, layer_i+1);
            ss_header << downstream_output.header;
            ss << downstream_output.body;
        }
        if(layer_i == 0){
            ss << ind << "    " << "namespace model_definition {\n";
            ss << ind << "    " << "    " << "using namespace RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::interface;\n";
            ss << ind << "    " << "    " << "using MODEL = Module<";
            for(TI layer_i = 0; layer_i < num_layers(model); layer_i++){
                ss << "layer_" << layer_i << "::TYPE";
                if(layer_i < num_layers(model)-1){
                    ss << ", Module<";
                }
            }
            for(TI layer_i = 0; layer_i < num_layers(model); layer_i++){
                ss << ">";
            }
            ss << ind << ";\n";
            ss << ind << "    " << "}\n";
            ss << ind << "    " << "using MODEL = model_definition::MODEL;\n";
            ss << ind << "    " << (const_declaration ? "const " : "") << "MODEL model = {";
            for(TI layer_i = 0; layer_i < num_layers(model); layer_i++){
                ss << "layer_" << layer_i << "::layer";
                if(layer_i < num_layers(model)-1){
                    ss << ", {";
                }
            }
            for(TI layer_i = 0; layer_i < num_layers(model); layer_i++){
                ss << "}";
            }
            ss << ";\n";

//            ss << ind << "    " << (const_declaration ? "const " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Module<" << layer_i << "> module = {layer_0::container, " << get_type_string<typename SPEC::NEXT_MODULE>() << "::module, };\n";
            ss << ind << "}";
        }
        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::sequential::Module<SPEC>& network, std::string name, bool const_declaration = false, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, network, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
