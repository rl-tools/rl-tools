#ifndef NEURAL_NETWORK_MODELS_H
#define NEURAL_NETWORK_MODELS_H
#include <layer_in_c/nn/nn.h>
#include <stdint.h>
namespace layer_in_c::nn_models::three_layer_fc {
    template <
            typename T_T,
            int INPUT_DIM,
            int LAYER_1_DIM,
            nn::activation_functions::ActivationFunction LAYER_1_FN,
            int LAYER_2_DIM,
            nn::activation_functions::ActivationFunction LAYER_2_FN,
            int OUTPUT_DIM,
            nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN
    >
    struct StructureSpecification{
        typedef T_T T;
        typedef nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> LAYER_1;
        typedef nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> LAYER_2;
        typedef nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> OUTPUT_LAYER;
        // Summary
        static constexpr int NUM_WEIGHTS = LAYER_1::NUM_WEIGHTS + LAYER_2::NUM_WEIGHTS + OUTPUT_LAYER::NUM_WEIGHTS;
    };
    template <typename T_DEVICE, typename STRUCTURE_SPEC>
    struct InferenceSpecification{
        typedef typename STRUCTURE_SPEC::T T;
        using DEVICE = T_DEVICE;

        using LAYER_1 = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::LAYER_1>;
        using LAYER_2 = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::LAYER_2>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER>;
    };

    template <typename T_DEVICE,typename STRUCTURE_SPEC>
    struct InferenceBackwardSpecification{
        typedef typename STRUCTURE_SPEC::T T;
        using DEVICE = T_DEVICE;
        using LAYER_1 = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::LAYER_1>;
        using LAYER_2 = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::LAYER_2>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackward<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER>;
    };

    template<typename T_DEVICE, typename STRUCTURE_SPEC, typename T_SGD_PARAMETERS>
    struct SGDSpecification{
        typedef typename STRUCTURE_SPEC::T T;
        using DEVICE = T_DEVICE;
        typedef T_SGD_PARAMETERS SGD_PARAMETERS;
        using LAYER_1 = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::LAYER_1, SGD_PARAMETERS>;
        using LAYER_2 = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::LAYER_2, SGD_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardSGD<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER, SGD_PARAMETERS>;
    };

    template<typename T_DEVICE, typename STRUCTURE_SPEC, typename T_ADAM_PARAMETERS>
    struct AdamSpecification{
        typedef typename STRUCTURE_SPEC::T T;
        using DEVICE = T_DEVICE ;
        typedef T_ADAM_PARAMETERS ADAM_PARAMETERS;
        using LAYER_1 = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::LAYER_1, ADAM_PARAMETERS>;
        using LAYER_2 = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::LAYER_2, ADAM_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardAdam<DEVICE, typename STRUCTURE_SPEC::OUTPUT_LAYER, ADAM_PARAMETERS>;
    };

    template<typename T_DEVICE, typename T_SPEC>
    struct NeuralNetwork{
        typedef T_DEVICE DEVICE;
        typedef T_SPEC SPEC;
        typedef typename SPEC::T T;
        static_assert(std::is_same_v<DEVICE, typename T_SPEC::DEVICE>);
        static constexpr int  INPUT_DIM = SPEC::LAYER_1     ::SPEC::INPUT_DIM;
        static constexpr int OUTPUT_DIM = SPEC::OUTPUT_LAYER::SPEC::OUTPUT_DIM;
        typename SPEC::LAYER_1 layer_1;
        typename SPEC::LAYER_2 layer_2;
        typename SPEC::OUTPUT_LAYER output_layer;
        template<typename NN>
        NeuralNetwork& operator= (const NN& other) {
            layer_1 = other.layer_1;
            layer_2 = other.layer_2;
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
        uint32_t age = 1;
    };


}

#endif