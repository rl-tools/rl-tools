#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_FLATTEN_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_FLATTEN_PERSIST_CODE_H
#include "layer.h"
#include <sstream>
#include "../../../persist/code.h"
#include "../../../nn/capability/persist_code.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace nn::layers::flatten::persist_code{
        template<typename DEVICE, typename SPEC>
        rl_tools::persist::Code finish(DEVICE& device, nn::layers::flatten::LayerForward<SPEC>& layer, std::string name, rl_tools::persist::Code input, bool const_declaration=true, typename DEVICE::index_t indent=0){
            using TI = typename DEVICE::index_t;
            std::stringstream indent_ss;
            for(TI i=0; i < indent; i++){
                indent_ss << "    ";
            }
            std::string ind = indent_ss.str();
            std::string TI_string = containers::persist::get_type_string<typename SPEC::TI>();
            std::stringstream ss, ss_header;
            ss_header << input.header;
            ss_header << "#include <rl_tools/nn/layers/flatten/layer.h>\n";
            ss << input.body;
            ss << ind << "namespace " << name << " {\n";
            ss << ind << "    using TYPE_POLICY = " << to_string(typename SPEC::TYPE_POLICY{}) << ";\n";
            ss << ind << "    using CONFIG = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::flatten::Configuration<TYPE_POLICY, " << TI_string << ">;\n";
            ss << ind << "    using TEMPLATE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::flatten::BindConfiguration<CONFIG>;\n";
            ss << ind << "    using INPUT_SHAPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<" << TI_string;
            constexpr auto RANK = length(typename SPEC::INPUT_SHAPE{});
            // emit each dim
            [&]<auto... Is>(std::index_sequence<Is...>){
                ((ss << ", " << get<Is>(typename SPEC::INPUT_SHAPE{})), ...);
            }(std::make_index_sequence<RANK>{});
            ss << ">;\n";
            using CONST_CAPABILITY = typename SPEC::CAPABILITY::template CHANGE_PARAMETERS<true, true>;
            ss << ind << "    using CAPABILITY = " << to_string(CONST_CAPABILITY{}) << ";\n";
            ss << ind << "    using TYPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::flatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;\n";
            std::string initializer_list;
            if constexpr(SPEC::CAPABILITY::TAG == nn::LayerCapability::Forward){
                initializer_list = "{}";
            }
            else if constexpr(SPEC::CAPABILITY::TAG == nn::LayerCapability::Backward){
                initializer_list = "{{}}";
            }
            else{
                initializer_list = "{{{}, output::container}}";
            }
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "TYPE module = " << initializer_list << ";\n";
            ss << ind << "    template <typename T_TYPE = TYPE>\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory = " << initializer_list << ";\n";
            ss << ind << "    template <typename T_TYPE = TYPE>\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory_function(){return T_TYPE" << initializer_list << ";}\n";
            ss << ind << "}\n";
            return {ss_header.str(), ss.str()};
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::flatten::LayerForward<SPEC>& layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0, bool finish=true){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::stringstream ss, ss_header;
        if(finish){
            return nn::layers::flatten::persist_code::finish(device, layer, name, {ss_header.str(), ss.str()}, const_declaration, indent);
        }
        else{
            return {ss_header.str(), ss.str()};
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::flatten::LayerBackward<SPEC>& layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0, bool finish=true){
        return save_code_split(device, static_cast<nn::layers::flatten::LayerForward<SPEC>&>(layer), name, const_declaration, indent, finish);
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::flatten::LayerGradient<SPEC>& layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        auto previous = save_code_split(device, static_cast<nn::layers::flatten::LayerBackward<SPEC>&>(layer), name, const_declaration, indent, false);
        ss_header << previous.header;
        ss << previous.body;
        ss << ind << "namespace " << name << " {\n";
        auto output = save_code_split(device, layer.output, "output", const_declaration, indent+1);
        ss_header << output.header;
        ss << output.body;
        ss << ind << "}\n";
        return nn::layers::flatten::persist_code::finish(device, layer, name, {ss_header.str(), ss.str()}, const_declaration, indent);
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn::layers::flatten::LayerForward<SPEC>& layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0){
        auto code = save_code_split(device, layer, name, const_declaration, indent);
        return code.header + code.body;
    }
    template <typename DEVICE, typename SPEC>
    std::string nn_analytics(DEVICE& device, nn::layers::flatten::LayerGradient<SPEC>& layer) {
        return "{\"output\": " + json(device, layer.output) + "}";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
