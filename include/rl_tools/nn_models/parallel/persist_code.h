#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_CODE_H
#include "../../persist/code.h"
#include "../sequential/persist_code.h"
#include "model.h"

#include <string>
#include <sstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::parallel::persist_code{
        template <typename SHAPE, auto INDEX = 0>
        void shape_to_string(std::stringstream& ss, const std::string& ti_string){
            if constexpr(INDEX == 0){
                ss << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<" << ti_string;
            }
            ss << ", " << get<INDEX>(SHAPE{});
            if constexpr(INDEX + 1 < length(SHAPE{})){
                shape_to_string<SHAPE, INDEX + 1>(ss, ti_string);
            }
            else{
                ss << ">";
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, std::string name, bool const_declaration=true, typename DEVICE::index_t indent = 0) {
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;

        auto pipeline_a_code = save_code_split(device, model.pipeline_a, "pipeline_a", const_declaration, indent+1);
        ss_header << pipeline_a_code.header;

        auto pipeline_b_code = save_code_split(device, model.pipeline_b, "pipeline_b", const_declaration, indent+1);
        ss_header << pipeline_b_code.header;

        persist::Code head_code;
        if constexpr(SPEC::HAS_HEAD){
            head_code = save_code_split(device, model.head, "head", const_declaration, indent+1);
            ss_header << head_code.header;
        }

        ss_header << "#include <rl_tools/nn_models/parallel/model.h>\n";

        ss << ind << "namespace " << name << " {\n";
        ss << pipeline_a_code.body;
        ss << pipeline_b_code.body;
        if constexpr(SPEC::HAS_HEAD){
            ss << head_code.body;
        }

        std::string TI_string = containers::persist::get_type_string<TI>();

        ss << ind << "    " << "namespace model_definition {\n";
        ss << ind << "    " << "    " << "using CAPABILITY = " << to_string(typename SPEC::CAPABILITY::template CHANGE_PARAMETERS<true, true>{}) << "; \n";
        {
            std::stringstream shape_ss;
            nn_models::parallel::persist_code::shape_to_string<typename SPEC::INPUT_SHAPE_A>(shape_ss, TI_string);
            ss << ind << "    " << "    " << "using INPUT_SHAPE_A = " << shape_ss.str() << ";\n";
        }
        {
            std::stringstream shape_ss;
            nn_models::parallel::persist_code::shape_to_string<typename SPEC::INPUT_SHAPE_B>(shape_ss, TI_string);
            ss << ind << "    " << "    " << "using INPUT_SHAPE_B = " << shape_ss.str() << ";\n";
        }
        ss << ind << "    " << "    " << "using MODULE_A = pipeline_a::TEMPLATE;\n";
        ss << ind << "    " << "    " << "using MODULE_B = pipeline_b::TEMPLATE;\n";
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    " << "using HEAD = head::TEMPLATE;\n";
            ss << ind << "    " << "    " << "using MODEL = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD>;\n";
        }
        else{
            ss << ind << "    " << "    " << "using MODEL = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;\n";
        }
        ss << ind << "    " << "}\n";

        ss << ind << "    " << "using TYPE = model_definition::MODEL;\n";

        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "TYPE module = [](){\n";
        ss << ind << "    " << "    TYPE m{};\n";
        ss << ind << "    " << "    m.pipeline_a = pipeline_a::factory<decltype(m.pipeline_a)>;\n";
        ss << ind << "    " << "    m.pipeline_b = pipeline_b::factory<decltype(m.pipeline_b)>;\n";
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory<decltype(m.head)>;\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}();\n";

        ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory = [](){\n";
        ss << ind << "    " << "    T_TYPE m{};\n";
        ss << ind << "    " << "    m.pipeline_a = pipeline_a::factory<decltype(m.pipeline_a)>;\n";
        ss << ind << "    " << "    m.pipeline_b = pipeline_b::factory<decltype(m.pipeline_b)>;\n";
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory<decltype(m.head)>;\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}();\n";

        ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory_function(){\n";
        ss << ind << "    " << "    T_TYPE m{};\n";
        ss << ind << "    " << "    m.pipeline_a = pipeline_a::factory_function<decltype(m.pipeline_a)>();\n";
        ss << ind << "    " << "    m.pipeline_b = pipeline_b::factory_function<decltype(m.pipeline_b)>();\n";
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory_function<decltype(m.head)>();\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}\n";

        ss << ind << "}";

        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, model, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
