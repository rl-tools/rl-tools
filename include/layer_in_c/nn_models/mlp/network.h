#ifndef LAYER_IN_C_NN_MODELS_MLP_NETWORK_H
#define LAYER_IN_C_NN_MODELS_MLP_NETWORK_H

#include <layer_in_c/nn/nn.h>

namespace layer_in_c::nn_models::mlp {
    template <typename T_T>
    struct ExampleStructureSpecification{
        typedef T_T T;
        static constexpr size_t INPUT_DIM = 10;
        static constexpr size_t OUTPUT_DIM = 5;
        static constexpr int NUM_HIDDEN_LAYERS = 2;
        static constexpr int HIDDEN_DIM = 30;
        static constexpr nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = nn::activation_functions::GELU;
        static constexpr nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = nn::activation_functions::IDENTITY;
    };
    template <typename T_DEVICE, typename T_STRUCTURE_SPEC>
    struct InferenceSpecification{
        using DEVICE = T_DEVICE;
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using T = typename S::T;

        using  INPUT_LAYER = nn::layers::dense::Layer<DEVICE, nn::layers::dense::LayerSpec<T,  S::INPUT_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>>;
        using HIDDEN_LAYER = nn::layers::dense::Layer<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION>>;
    };

    template <typename T_DEVICE, typename T_STRUCTURE_SPEC>
    struct InferenceBackwardSpecification{
        using DEVICE = T_DEVICE;
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using T = typename S::T;

        using  INPUT_LAYER = nn::layers::dense::LayerBackward<DEVICE, nn::layers::dense::LayerSpec<T,  S::INPUT_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackward<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackward<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION>>;
    };

    template<typename T_DEVICE, typename T_STRUCTURE_SPEC, typename T_SGD_PARAMETERS>
    struct SGDSpecification{
        using DEVICE = T_DEVICE;
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using T = typename S::T;
        using SGD_PARAMETERS = T_SGD_PARAMETERS;

        using  INPUT_LAYER = nn::layers::dense::LayerBackwardSGD<DEVICE, nn::layers::dense::LayerSpec<T,  S::INPUT_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>, SGD_PARAMETERS>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardSGD<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>, SGD_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardSGD<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION>, SGD_PARAMETERS>;
    };

    template<typename T_DEVICE, typename T_STRUCTURE_SPEC, typename T_ADAM_PARAMETERS>
    struct AdamSpecification{
        using DEVICE = T_DEVICE;
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using T = typename S::T ;
        using ADAM_PARAMETERS = T_ADAM_PARAMETERS;

        using  INPUT_LAYER = nn::layers::dense::LayerBackwardAdam<DEVICE, nn::layers::dense::LayerSpec<T,  S::INPUT_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>, ADAM_PARAMETERS>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardAdam<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION>, ADAM_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardAdam<DEVICE, nn::layers::dense::LayerSpec<T, S::HIDDEN_DIM, S::OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION>, ADAM_PARAMETERS>;
    };

    template<typename T_DEVICE, typename T_SPEC>
    struct NeuralNetwork{
        typedef T_DEVICE DEVICE;
        typedef T_SPEC SPEC;
        typedef typename SPEC::T T;
        static_assert(std::is_same_v<DEVICE, typename T_SPEC::DEVICE>);
        static constexpr size_t  INPUT_DIM = SPEC::INPUT_LAYER ::SPEC::INPUT_DIM;
        static constexpr size_t OUTPUT_DIM = SPEC::OUTPUT_LAYER::SPEC::OUTPUT_DIM;
        static constexpr size_t NUM_HIDDEN_LAYERS = SPEC::STRUCTURE_SPEC::NUM_HIDDEN_LAYERS;
        typename SPEC:: INPUT_LAYER input_layer;
        typename SPEC::HIDDEN_LAYER hidden_layers[NUM_HIDDEN_LAYERS];
        typename SPEC::OUTPUT_LAYER output_layer;
        template<typename NN>
        NeuralNetwork& operator= (const NN& other) {
            static_assert(std::is_same_v<typename NeuralNetwork::SPEC::STRUCTURE_SPEC, typename NN::SPEC::STRUCTURE_SPEC>);
            input_layer = other.input_layer;
            for(size_t i = 0; i < NUM_HIDDEN_LAYERS; i++){
                hidden_layers[i] = other.hidden_layers[i];
            }
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
