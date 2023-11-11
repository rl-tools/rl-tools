#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_PARAMETERS_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_PARAMETERS_PERSIST_CODE_H
#include "../../containers/persist_code.h"
#include "parameters.h"
#include <sstream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    std::string get_type_string(nn::parameters::Plain p){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain";
    }
    std::string get_type_string(nn::parameters::Gradient p){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Gradient";
    }

    template <typename DEVICE>
    std::string get_type_string_tag(const DEVICE&, const nn::parameters::categories::Weights){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights";
    }

    template <typename DEVICE>
    std::string get_type_string_tag(const DEVICE&, const nn::parameters::categories::Biases){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases";
    }

    template <typename DEVICE>
    std::string get_type_string_tag(const DEVICE&, const nn::parameters::groups::Normal){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal";
    }

    template <typename DEVICE>
    std::string get_type_string_tag(const DEVICE&, const nn::parameters::groups::Input){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Input";
    }

    template <typename DEVICE>
    std::string get_type_string_tag(const DEVICE&, const nn::parameters::groups::Output){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Output";
    }

    template<typename DEVICE, typename CONTAINER>
    persist::Code save_split(DEVICE& device, nn::parameters::Plain::instance<CONTAINER>& parameter, std::string name, bool const_declaration=false, typename DEVICE::index_t indent=0, bool output_memory_only=false){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        ss << ind << "namespace " << name << " {\n";
        auto container = save_split(device, parameter.parameters, "parameters_memory", const_declaration, indent+1);
        ss_header << container.header;
        ss << container.body;
        if(!output_memory_only){
            ss << ind << "    " << "using PARAMETER_SPEC = " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, "
            << get_type_string_tag(device, typename CONTAINER::GROUP_TAG{})
            << ", "
            << get_type_string_tag(device, typename CONTAINER::CATEGORY_TAG{})
            << ">;\n";
            ss << ind << "    " << (const_declaration ? "const " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {parameters_memory::container};\n";
        }
        ss << ind << "}\n";
        return {"", ss.str()};
    }

    template<typename DEVICE, typename CONTAINER>
    persist::Code save_split(DEVICE& device, nn::parameters::Gradient::instance<CONTAINER>& parameter, std::string name, bool const_declaration=false, typename DEVICE::index_t indent=0, bool output_memory_only=false){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        ss_header << "#include <rl_tools/utils/generic/typing.h>\n";
        auto plain = save_split(device, (nn::parameters::Plain::instance<CONTAINER>&) parameter, name, const_declaration, indent, true);
        ss_header << plain.header;
        ss << plain.body;
        ss << ind << "namespace " << name << " {\n";
        auto gradient = save_split(device, parameter.gradient, "gradient_memory", const_declaration, indent+1);
        ss_header << gradient.header;
        ss << gradient.body;
        if(!output_memory_only){
            ss << ind << "    " << "static_assert(RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::utils::typing::is_same_v<parameters_memory::CONTAINER_TYPE, gradient_memory::CONTAINER_TYPE>);\n";
            ss << ind << "    " << "using PARAMETER_SPEC = " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Gradient::spec<parameters_memory::CONTAINER_TYPE, " << get_type_string_tag(device, typename CONTAINER::CATEGORY_TAG{}) << ">;\n";
            ss << ind << "    " << (const_declaration ? "const " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Gradient::instance<parameters_memory::CONTAINER_TYPE> parameters = {parameters_memory::container, gradient_memory::container};\n";
        }
        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif