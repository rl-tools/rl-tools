#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_CODE_H
#include "../../persist/code.h"
#include "../sequential/persist_code.h"
#include "model.h"

#include <string>
#include <sstream>
#include <vector>

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

        template <auto I = 0, typename DEVICE, typename SPEC>
        void _save_code_branches(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model,
                                  std::vector<persist::Code>& branch_codes,
                                  bool const_declaration, typename DEVICE::index_t indent){
            if constexpr(I < SPEC::NUM_BRANCHES){
                std::string name = "branch_" + std::to_string(I);
                auto code = save_code_split(device, get<I>(model.pipelines), name, const_declaration, indent + 1);
                branch_codes.push_back(code);
                _save_code_branches<I + 1>(device, model, branch_codes, const_declaration, indent);
            }
        }

        template <auto I = 0, typename SPEC>
        void _emit_branch_input_shapes(std::stringstream& ss, const std::string& ind, const std::string& ti_string){
            if constexpr(I < SPEC::NUM_BRANCHES){
                using BRANCH_TUPLE = typename SPEC::BRANCH_TUPLE;
                using BRANCH = typename utils::tuple_element<I, BRANCH_TUPLE>::type;
                std::stringstream shape_ss;
                shape_to_string<typename BRANCH::INPUT_SHAPE>(shape_ss, ti_string);
                ss << ind << "    " << "    " << "using INPUT_SHAPE_" << I << " = " << shape_ss.str() << ";\n";
                _emit_branch_input_shapes<I + 1, SPEC>(ss, ind, ti_string);
            }
        }

        template <auto I = 0, typename SPEC>
        void _emit_branch_types(std::stringstream& ss, const std::string& ind){
            if constexpr(I < SPEC::NUM_BRANCHES){
                ss << ind << "    " << "    " << "using BRANCH_" << I << " = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::parallel::Branch<branch_" << I << "::TEMPLATE, INPUT_SHAPE_" << I << ">;\n";
                _emit_branch_types<I + 1, SPEC>(ss, ind);
            }
        }

        template <auto I = 0, typename SPEC>
        void _emit_branch_list(std::stringstream& ss){
            if constexpr(I < SPEC::NUM_BRANCHES){
                ss << ", BRANCH_" << I;
                _emit_branch_list<I + 1, SPEC>(ss);
            }
        }

        template <auto I = 0, typename SPEC>
        void _emit_factory_assignments(std::stringstream& ss, const std::string& ind, const std::string& accessor){
            if constexpr(I < SPEC::NUM_BRANCHES){
                if constexpr(I == 0){
                    ss << ind << "    " << "    " << accessor << ".pipelines.content = branch_0::factory<decltype(" << accessor << ".pipelines.content)>;\n";
                }
                else{
                    // Navigate the MapTuple hierarchy: for I-th element, we need I levels of static_cast
                    // Use rl_tools::get<I>() instead
                    ss << ind << "    " << "    " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << I << ">(" << accessor << ".pipelines) = branch_" << I << "::factory<decltype(RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << I << ">(" << accessor << ".pipelines))>;\n";
                }
                _emit_factory_assignments<I + 1, SPEC>(ss, ind, accessor);
            }
        }

        template <auto I = 0, typename SPEC>
        void _emit_factory_function_assignments(std::stringstream& ss, const std::string& ind, const std::string& accessor){
            if constexpr(I < SPEC::NUM_BRANCHES){
                if constexpr(I == 0){
                    ss << ind << "    " << "    " << accessor << ".pipelines.content = branch_0::factory_function<decltype(" << accessor << ".pipelines.content)>();\n";
                }
                else{
                    ss << ind << "    " << "    " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << I << ">(" << accessor << ".pipelines) = branch_" << I << "::factory_function<decltype(RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::get<" << I << ">(" << accessor << ".pipelines))>();\n";
                }
                _emit_factory_function_assignments<I + 1, SPEC>(ss, ind, accessor);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, std::string name, bool const_declaration=true, typename DEVICE::index_t indent = 0){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;

        std::vector<persist::Code> branch_codes;
        nn_models::parallel::persist_code::_save_code_branches(device, model, branch_codes, const_declaration, indent);
        for(auto& code : branch_codes){
            ss_header << code.header;
        }

        persist::Code head_code;
        if constexpr(SPEC::HAS_HEAD){
            head_code = save_code_split(device, model.head, "head", const_declaration, indent+1);
            ss_header << head_code.header;
        }

        ss_header << "#include <rl_tools/nn_models/parallel/model.h>\n";

        std::string TI_string = containers::persist::get_type_string<TI>();

        ss << ind << "namespace " << name << " {\n";
        for(auto& code : branch_codes){
            ss << code.body;
        }
        if constexpr(SPEC::HAS_HEAD){
            ss << head_code.body;
        }

        ss << ind << "    " << "namespace model_definition {\n";
        ss << ind << "    " << "    " << "using CAPABILITY = " << to_string(typename SPEC::CAPABILITY::template CHANGE_PARAMETERS<true, true>{}) << "; \n";

        nn_models::parallel::persist_code::_emit_branch_input_shapes<0, SPEC>(ss, ind, TI_string);
        nn_models::parallel::persist_code::_emit_branch_types<0, SPEC>(ss, ind);

        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    " << "using HEAD = head::TEMPLATE;\n";
            ss << ind << "    " << "    " << "using MODEL = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::parallel::Build<CAPABILITY, HEAD";
        }
        else{
            ss << ind << "    " << "    " << "using MODEL = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::parallel::Build<CAPABILITY, void";
        }
        nn_models::parallel::persist_code::_emit_branch_list<0, SPEC>(ss);
        ss << ">;\n";

        ss << ind << "    " << "}\n";

        ss << ind << "    " << "using TYPE = model_definition::MODEL;\n";

        // constexpr module
        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "TYPE module = [](){\n";
        ss << ind << "    " << "    TYPE m{};\n";
        nn_models::parallel::persist_code::_emit_factory_assignments<0, SPEC>(ss, ind, "m");
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory<decltype(m.head)>;\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}();\n";

        // factory variable template
        ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory = [](){\n";
        ss << ind << "    " << "    T_TYPE m{};\n";
        nn_models::parallel::persist_code::_emit_factory_assignments<0, SPEC>(ss, ind, "m");
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory<decltype(m.head)>;\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}();\n";

        // factory function
        ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
        ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory_function(){\n";
        ss << ind << "    " << "    T_TYPE m{};\n";
        nn_models::parallel::persist_code::_emit_factory_function_assignments<0, SPEC>(ss, ind, "m");
        if constexpr(SPEC::HAS_HEAD){
            ss << ind << "    " << "    m.head = head::factory_function<decltype(m.head)>();\n";
        }
        ss << ind << "    " << "    return m;\n";
        ss << ind << "    " << "}\n";

        ss << ind << "}";

        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0){
        auto code = save_code_split(device, model, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
