#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_H
#include "../../containers/matrix/persist_code.h"
#include "../../persist/code.h"
#include "../../nn/persist_code.h"
#include "model.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<auto LAYER_I = 0, typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& model, std::string name, bool const_declaration=true, typename DEVICE::index_t indent = 0) {
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        persist::Code layer_output = save_code_split(device, get_layer<LAYER_I>(model), "layer_" + std::to_string(LAYER_I), const_declaration, indent+1);
        ss_header << layer_output.header;
        ss_header << "#include <rl_tools/nn_models/sequential/model.h>\n";
        if constexpr(LAYER_I == 0){
            ss << ind << "namespace " << name << " {\n";
        }
        ss << layer_output.body;
        if constexpr(LAYER_I + 1 < SPEC::NUM_LAYERS){
            auto downstream_output = save_code_split<LAYER_I + 1>(device, model, name, const_declaration, indent);
            ss_header << downstream_output.header;
            ss << downstream_output.body;
        }
        if constexpr(LAYER_I == 0){
            ss << ind << "    " << "namespace model_definition {\n";
//            ss << ind << "    " << "    " << "using namespace RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::interface;\n";
//            std::string capability = "Forward";
            ss << ind << "    " << "    " << "using CAPABILITY = " << to_string(typename SPEC::CAPABILITY::template CHANGE_PARAMETERS<true, true>{}) << "; \n";
            ss << ind << "    " << "    " << "template <typename... T_CONTENTS>\n";
            ss << ind << "    " << "    " << "using Module = typename RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Module<T_CONTENTS...>;\n";
            ss << ind << "    " << "    " << "using MODULE_CHAIN = Module<";
            for(TI layer_i = 0; layer_i < num_layers(model); layer_i++){
                ss << "layer_" << layer_i << "::TEMPLATE";
                if(layer_i < num_layers(model)-1){
                    ss << ", ";
                }
            }
            ss << ">;\n";
            ss << ind << "    " << "    " << "using MODEL = typename RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, layer_0::INPUT_SHAPE>;\n";
            ss << ind << "    " << "}\n";
            ss << ind << "    " << "using TYPE = model_definition::MODEL;\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "TYPE module = [](){\n";
            ss << ind << "    " << "    TYPE m{};\n";
            for(TI inner_layer_i = 0; inner_layer_i < num_layers(model); inner_layer_i++){
                ss << ind << "    " << "    RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << inner_layer_i << ">(m.content) = layer_" << inner_layer_i << "::factory<typename RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::tuple_element<" << inner_layer_i << ", typename TYPE::SPEC::LAYER_SPECS>::type::CONTENT>;\n";
            }
            ss << ind << "    " << "    return m;\n";
            ss << ind << "    " << "}();\n";

            ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory = [](){\n";
            ss << ind << "    " << "    T_TYPE m{};\n";
            for(TI inner_layer_i = 0; inner_layer_i < num_layers(model); inner_layer_i++){
                ss << ind << "    " << "    RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << inner_layer_i << ">(m.content) = layer_" << inner_layer_i << "::factory<typename RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::tuple_element<" << inner_layer_i << ", typename T_TYPE::SPEC::LAYER_SPECS>::type::CONTENT>;\n";
            }
            ss << ind << "    " << "    return m;\n";
            ss << ind << "    " << "}();" << "\n";
            ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory_function(){\n";
            ss << ind << "    " << "    T_TYPE m{};\n";
            for(TI inner_layer_i = 0; inner_layer_i < num_layers(model); inner_layer_i++){
                ss << ind << "    " << "    RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << inner_layer_i << ">(m.content) = layer_" << inner_layer_i << "::factory_function<typename RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::tuple_element<" << inner_layer_i << ", typename T_TYPE::SPEC::LAYER_SPECS>::type::CONTENT>();\n";
            }
            ss << ind << "    " << "    return m;\n";
            ss << ind << "    " << "}\n";
            ss << ind << "}";


//            ss << ind << "    " << (const_declaration ? "const " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::Module<" << layer_i << "> module = {layer_0::container, " << get_type_string<typename SPEC::NEXT_MODULE>() << "::module, };\n";
        }
        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& network, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, network, name, const_declaration, indent);
        return code.header + code.body;
    }
    namespace nn_models::sequential{
        template <auto LAYER_I = 0, typename DEVICE, typename SPEC>
        void nn_analytics_layers(std::string& data, DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& model) {
            if constexpr(LAYER_I < SPEC::NUM_LAYERS) {
                if constexpr(LAYER_I > 0){ data += ", "; }
                data += nn_analytics(device, get_layer<LAYER_I>(model));
                nn_analytics_layers<LAYER_I + 1>(data, device, model);
            }
        }
    }
    template <typename DEVICE, typename SPEC>
    std::string nn_analytics(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& model) {
        std::string data = "{\"layers\":[";
        nn_models::sequential::nn_analytics_layers(data, device, model);
        data += "]}";
        return data;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
