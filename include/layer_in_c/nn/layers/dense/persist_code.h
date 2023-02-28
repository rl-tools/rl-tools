#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_CODE_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_CODE_H
#include "layer.h"
#include <sstream>

namespace layer_in_c {
    namespace nn::layers::dense::persist{
        template<nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
        auto get_activation_function_string(){
            static_assert(ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::IDENTITY ||
                          ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU ||
                          ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::GELU ||
                          ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::TANH ||
                          ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::SIGMOID);

            if constexpr (ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::IDENTITY){
                return "layer_in_c::nn::activation_functions::ActivationFunction::IDENTITY";
            } else if constexpr (ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
                return "layer_in_c::nn::activation_functions::ActivationFunction::RELU";
            } else if constexpr (ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::TANH){
                return "layer_in_c::nn::activation_functions::ActivationFunction::TANH";
            } else if constexpr (ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::SIGMOID){
                return "layer_in_c::nn::activation_functions::ActivationFunction::SIGMOID";
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    std::string save(DEVICE &device, nn::layers::dense::Layer <SPEC> &layer, std::string name) {
        using TI = typename DEVICE::index_t;
        std::stringstream ss;
        ss << "#include <layer_in_c/nn/layers/dense/layer.h>\n";
        ss << "namespace " << name << " {\n";
        ss << save(device, layer.weights, "weights");
        ss << save(device, layer.biases, "biases");
        ss << "using SPEC = " << "layer_in_c::nn::layers::dense::Specification<"
            << containers::persist::get_type_string<typename SPEC::T>() << ", "
            << containers::persist::get_type_string<typename SPEC::TI>() << ", "
            << SPEC::INPUT_DIM << ", "
            << SPEC::OUTPUT_DIM << ", "
            << nn::layers::dense::persist::get_activation_function_string<SPEC::ACTIVATION_FUNCTION>() << ", "
            << 1 << ", "
            << "true , "
            << "layer_in_c::matrix::layouts::RowMajorAlignment<" << containers::persist::get_type_string<TI>() << ", 1>"
            << ">; \n";
        ss << "layer_in_c::nn::layers::dense::Layer<SPEC> layer = {weights::matrix, biases::matrix};\n";
        ss << "}\n";

        return ss.str();
    }
}

#endif
