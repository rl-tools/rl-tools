#ifndef NEURAL_NETWORK_MODELS_H
#define NEURAL_NETWORK_MODELS_H
#include <layer_in_c/nn/nn.h>
namespace layer_in_c::nn_models::three_layer_fc {
    template <
            typename T_T,
            size_t INPUT_DIM,
            size_t LAYER_1_DIM,
            nn::activation_functions::ActivationFunction LAYER_1_FN,
            size_t LAYER_2_DIM,
            nn::activation_functions::ActivationFunction LAYER_2_FN,
            size_t OUTPUT_DIM,
            nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN
    >
    struct StructureSpecification{
        typedef T_T T;
        typedef nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> LAYER_1;
        typedef nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> LAYER_2;
        typedef nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> OUTPUT_LAYER;
        // Summary
        static constexpr size_t NUM_WEIGHTS = LAYER_1::NUM_WEIGHTS + LAYER_2::NUM_WEIGHTS + OUTPUT_LAYER::NUM_WEIGHTS;
    };
    template <typename T_DEVICE, typename T_STRUCTURE_SPEC>
    struct InferenceSpecification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using T = typename STRUCTURE_SPEC::T;
        using DEVICE = T_DEVICE;

        using LAYER_1 = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::LAYER_1>;
        using LAYER_2 = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::LAYER_2>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER>;
    };

    template <typename T_DEVICE,typename T_STRUCTURE_SPEC>
    struct InferenceBackwardSpecification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using T = typename STRUCTURE_SPEC::T;
        using DEVICE = T_DEVICE;
        using LAYER_1 = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::LAYER_1>;
        using LAYER_2 = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::LAYER_2>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER>;
    };

    template<typename T_DEVICE, typename T_STRUCTURE_SPEC, typename T_SGD_PARAMETERS>
    struct SGDSpecification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using T = typename STRUCTURE_SPEC::T;
        using DEVICE = T_DEVICE;
        using SGD_PARAMETERS = T_SGD_PARAMETERS;
        using LAYER_1 = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::LAYER_1, SGD_PARAMETERS>;
        using LAYER_2 = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::LAYER_2, SGD_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER, SGD_PARAMETERS>;
    };

    template<typename T_DEVICE, typename T_STRUCTURE_SPEC, typename T_ADAM_PARAMETERS>
    struct AdamSpecification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using T = typename STRUCTURE_SPEC::T;
        using DEVICE = T_DEVICE ;
        using ADAM_PARAMETERS = T_ADAM_PARAMETERS;
        using LAYER_1 = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::LAYER_1, ADAM_PARAMETERS>;
        using LAYER_2 = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::LAYER_2, ADAM_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER, ADAM_PARAMETERS>;
    };

    template<typename T_DEVICE, typename T_SPEC>
    struct NeuralNetwork{
        using DEVICE = T_DEVICE;
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        static_assert(std::is_same_v<DEVICE, typename T_SPEC::DEVICE>);
        static constexpr size_t  INPUT_DIM = SPEC::LAYER_1     ::SPEC::INPUT_DIM;
        static constexpr size_t OUTPUT_DIM = SPEC::OUTPUT_LAYER::SPEC::OUTPUT_DIM;
        static constexpr size_t NUM_WEIGHTS = SPEC::STRUCTURE_SPEC::NUM_WEIGHTS;
        typename SPEC::LAYER_1 input_layer;
        typename SPEC::LAYER_2 hidden_layer_0;
        typename SPEC::OUTPUT_LAYER output_layer;
        template<typename NN>
        NeuralNetwork& operator= (const NN& other) {
            input_layer = other.input_layer;
            hidden_layer_0 = other.hidden_layer_0;
            output_layer = other.output_layer;
            return *this;
        }
    };

    template<typename DEVICE, typename SPEC>
    struct NeuralNetworkBackward: public NeuralNetwork<DEVICE, SPEC>{
        template<typename NN>
        NeuralNetworkBackward& operator= (const NN& other) {
            NeuralNetwork<DEVICE, SPEC>::operator=(other);
            return *this;
        }
    };
    template<typename DEVICE, typename SPEC>
    struct NeuralNetworkBackwardGradient: public NeuralNetworkBackward<DEVICE, SPEC>{
    };

    template<typename DEVICE, typename SPEC>
    struct NeuralNetworkSGD: public NeuralNetworkBackwardGradient<DEVICE, SPEC>{
    };

    template<typename DEVICE, typename SPEC>
    struct NeuralNetworkAdam: public NeuralNetworkBackwardGradient<DEVICE, SPEC>{
        size_t age = 1;
    };


}

#endif