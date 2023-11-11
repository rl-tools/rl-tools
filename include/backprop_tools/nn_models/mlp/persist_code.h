
#include "network.h"
#include "../../nn/optimizers/adam/persist_code.h"
#include "../../nn/layers/dense/persist_code.h"



#include <string>
#include <sstream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    persist::Code save_split(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using STRUCTURE_SPEC = typename SPEC::STRUCTURE_SPEC;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        ss_header << "#include <rl_tools/nn_models/mlp/network.h>\n";
        std::stringstream ss;
        ss << ind << "namespace " << name << " {\n";
        auto input_layer = save_split(device, network.input_layer, "input_layer", const_declaration, indent+1);
        ss_header << input_layer.header;
        ss << input_layer.body;
        for(TI hidden_layer_i = 0; hidden_layer_i < SPEC::NUM_HIDDEN_LAYERS; hidden_layer_i++){
            auto hidden_layer = save_split(device, network.hidden_layers[hidden_layer_i], "hidden_layer_" + std::to_string(hidden_layer_i), const_declaration, indent+1);
            ss_header << hidden_layer.header;
            ss << hidden_layer.body;
        }
        auto output_layer = save_split(device, network.output_layer, "output_layer", const_declaration, indent+1);
        ss_header << output_layer.header;
        ss << output_layer.body;
        ss << ind << "    using STRUCTURE_SPEC = rl_tools::nn_models::mlp::StructureSpecification<";
        ss << containers::persist::get_type_string<T>() << ", ";
        ss << containers::persist::get_type_string<TI>() << ", ";
        ss << STRUCTURE_SPEC::INPUT_DIM << ", " << STRUCTURE_SPEC::OUTPUT_DIM << ", " << STRUCTURE_SPEC::NUM_LAYERS << ", " << STRUCTURE_SPEC::HIDDEN_DIM << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>() << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<STRUCTURE_SPEC::OUTPUT_ACTIVATION_FUNCTION>() << ", ";
        ss << ind << "1, rl_tools::MatrixDynamicTag, true, rl_tools::matrix::layouts::RowMajorAlignment<" << containers::persist::get_type_string<TI>() << ", 1>>; \n";
        ss << ind << "    using SPEC = rl_tools::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>; \n";
        ss << ind << "    " << (const_declaration ? "const " : "") << "rl_tools::nn_models::mlp::NeuralNetwork<SPEC> mlp = {";
        ss << "input_layer::layer, ";
        ss << "{";
        for(TI hidden_layer_i = 0; hidden_layer_i < SPEC::NUM_HIDDEN_LAYERS; hidden_layer_i++){
            if(hidden_layer_i > 0){
                ss << ", ";
            }
            ss << "hidden_layer_" << hidden_layer_i << "::layer";
        }
        ss << "}, ";
        ss << "output_layer::layer";
        ss << "};\n";

        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0) {
        auto code = save_split(device, network, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
