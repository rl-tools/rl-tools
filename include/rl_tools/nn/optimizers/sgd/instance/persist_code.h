#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_PERSIST_CODE_H

#include "../sgd.h"
#include "../../../../nn/parameters/persist_code.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    std::string get_type_string(nn::parameters::SGD p){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::SGD";
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::parameters::SGD::Instance<SPEC>& parameter, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0, bool output_memory_only=false){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        auto plain = save_code_split(device, (nn::parameters::Gradient::Instance<SPEC>&) parameter, name, const_declaration, indent, true);
        ss_header << plain.header;
        ss_header << "#include <rl_tools/nn/optimizers/sgd/sgd.h>\n";
        ss << plain.body;
        ss << ind << "namespace " << name << " {\n";
        auto velocity = save_code_split(device, parameter.velocity, "velocity_memory", const_declaration, indent+1);
        ss_header << velocity.header;
        ss << velocity.body;
        if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
            auto master_parameters = save_code_split(device, parameter.master_parameters, "master_parameters_memory", const_declaration, indent+1);
            ss_header << master_parameters.header;
            ss << master_parameters.body;
        }
        if(!output_memory_only){
            ss << ind << "    " << "using TYPE_POLICY = " << to_string(typename SPEC::TYPE_POLICY{}) << ";\n";
            ss << ind << "    " << "using PARAMETER_SPEC = " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::SGD::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, "
               << get_type_string_tag(device, typename SPEC::GROUP_TAG{})
               << ", "
               << get_type_string_tag(device, typename SPEC::CATEGORY_TAG{})
               << ", true, true>;\n";
            if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
                ss << ind << "    " << (const_declaration ? "constexpr " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::SGD::Instance<PARAMETER_SPEC> parameters = {{{parameters_memory::container}, gradient_memory::container}, velocity_memory::container, master_parameters_memory::container};\n";
            }
            else{
                ss << ind << "    " << (const_declaration ? "constexpr " : "") << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::SGD::Instance<PARAMETER_SPEC> parameters = {{{parameters_memory::container}, gradient_memory::container}, velocity_memory::container};\n";
            }
        }
        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
    template <typename DEVICE, typename SPEC>
    std::string nn_analytics(DEVICE& device, nn::parameters::SGD::Instance<SPEC>& p) {
        std::string data;
        data += "{";
        data += nn_analytics(device, static_cast<nn::parameters::Gradient::Instance<SPEC>&>(p), true) + ", ";
        data += "\"velocity\": " + json(device, p.velocity);
        if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
            data += ", \"master_parameters\": " + json(device, p.master_parameters);
        }
        data += "}";
        return data;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
