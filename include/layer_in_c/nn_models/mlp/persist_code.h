
#include "network.h"
#include <layer_in_c/nn/layers/dense/persist_code.h>



#include <string>
#include <sstream>
namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    std::string save(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, std::string name) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using STRUCTURE_SPEC = typename SPEC::STRUCTURE_SPEC;
        std::stringstream ss;
        ss << "#include <layer_in_c/nn_models/mlp/network.h>\n";
        ss << "namespace " << name << " {\n";
        ss << save(device, network.input_layer, "input_layer");
        for(TI hidden_layer_i = 0; hidden_layer_i < SPEC::NUM_HIDDEN_LAYERS; hidden_layer_i++){
            ss << save(device, network.hidden_layers[hidden_layer_i], "hidden_layer_" + std::to_string(hidden_layer_i));
        }
        ss << save(device, network.output_layer, "output_layer");
        ss << "using STRUCTURE_SPEC = layer_in_c::nn_models::mlp::StructureSpecification<";
        ss << containers::persist::get_type_string<T>() << ", ";
        ss << containers::persist::get_type_string<TI>() << ", ";
        ss << STRUCTURE_SPEC::INPUT_DIM << ", " << STRUCTURE_SPEC::OUTPUT_DIM << ", " << STRUCTURE_SPEC::NUM_LAYERS << ", " << STRUCTURE_SPEC::HIDDEN_DIM << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>() << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<STRUCTURE_SPEC::OUTPUT_ACTIVATION_FUNCTION>() << ", ";
        ss << "1, true, layer_in_c::matrix::layouts::RowMajorAlignment<" << containers::persist::get_type_string<TI>() << ", 1>>; \n";
        ss << "using SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>; \n";
        ss << "layer_in_c::nn_models::mlp::NeuralNetwork<SPEC> mlp = {";
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

        ss << "}\n";
        return ss.str();
    }
}