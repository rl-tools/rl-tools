#ifndef LAYER_IN_C_NN_MODELS_MLP_NETWORK_H
#define LAYER_IN_C_NN_MODELS_MLP_NETWORK_H

#include <layer_in_c/nn/nn.h>
#include <layer_in_c/utils/generic/typing.h>
#include <layer_in_c/containers.h>

namespace layer_in_c::nn_models::mlp {
    template <typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM, T_TI T_NUM_LAYERS, T_TI T_HIDDEN_DIM, nn::activation_functions::ActivationFunction T_HIDDEN_ACTIVATION_FUNCTION, nn::activation_functions::ActivationFunction T_OUTPUT_ACTIVATION_FUNCTION, T_TI T_BATCH_SIZE=1, bool T_ENFORCE_FLOATING_POINT_TYPE=true>
    struct StructureSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr T_TI INPUT_DIM = T_INPUT_DIM;
        static constexpr T_TI OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr T_TI NUM_LAYERS = T_NUM_LAYERS; // The input and output layers count towards the total number of layers
        static constexpr T_TI HIDDEN_DIM = T_HIDDEN_DIM;
        static constexpr auto HIDDEN_ACTIVATION_FUNCTION = T_HIDDEN_ACTIVATION_FUNCTION;
        static constexpr auto OUTPUT_ACTIVATION_FUNCTION = T_OUTPUT_ACTIVATION_FUNCTION;
        static constexpr T_TI BATCH_SIZE = T_BATCH_SIZE;

        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
    };
    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
            utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>
            && SPEC_1::INPUT_DIM == SPEC_2::INPUT_DIM
            && SPEC_1::OUTPUT_DIM == SPEC_2::OUTPUT_DIM
            && SPEC_1::NUM_LAYERS == SPEC_2::NUM_LAYERS
            && SPEC_1::HIDDEN_DIM == SPEC_2::HIDDEN_DIM;
    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec =
            check_spec_memory<SPEC_1, SPEC_2>
            && SPEC_1::HIDDEN_ACTIVATION_FUNCTION == SPEC_2::HIDDEN_ACTIVATION_FUNCTION
            && SPEC_1::OUTPUT_ACTIVATION_FUNCTION == SPEC_2::OUTPUT_ACTIVATION_FUNCTION;


    template <typename T_STRUCTURE_SPEC>
    struct Specification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using T = typename S::T;
        using TI = typename S::TI;
        static constexpr TI NUM_HIDDEN_LAYERS = STRUCTURE_SPEC::NUM_LAYERS - 2;
        static constexpr TI INPUT_DIM = S::INPUT_DIM;
        static constexpr TI HIDDEN_DIM = S::HIDDEN_DIM;
        static constexpr TI OUTPUT_DIM = S::OUTPUT_DIM;
        static constexpr TI BATCH_SIZE = S::BATCH_SIZE;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = S::ENFORCE_FLOATING_POINT_TYPE;

        using INPUT_LAYER_SPEC  = nn::layers::dense::Specification<T, TI, INPUT_DIM , HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION, BATCH_SIZE, ENFORCE_FLOATING_POINT_TYPE>;
        using HIDDEN_LAYER_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION, BATCH_SIZE, ENFORCE_FLOATING_POINT_TYPE>;
        using OUTPUT_LAYER_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION, BATCH_SIZE, ENFORCE_FLOATING_POINT_TYPE>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct InferenceSpecification: Specification<T_STRUCTURE_SPEC>{
        using  INPUT_LAYER = nn::layers::dense::Layer<typename Specification<T_STRUCTURE_SPEC>::INPUT_LAYER_SPEC >;
        using HIDDEN_LAYER = nn::layers::dense::Layer<typename Specification<T_STRUCTURE_SPEC>::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<typename Specification<T_STRUCTURE_SPEC>::OUTPUT_LAYER_SPEC>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct InferenceBackwardSpecification: Specification<T_STRUCTURE_SPEC>{
        using  INPUT_LAYER = nn::layers::dense::LayerBackward<typename Specification<T_STRUCTURE_SPEC>::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackward<typename Specification<T_STRUCTURE_SPEC>::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackward<typename Specification<T_STRUCTURE_SPEC>::OUTPUT_LAYER_SPEC>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct BackwardGradientSpecification: Specification<T_STRUCTURE_SPEC>{
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename Specification<T_STRUCTURE_SPEC>::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardGradient<typename Specification<T_STRUCTURE_SPEC>::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename Specification<T_STRUCTURE_SPEC>::OUTPUT_LAYER_SPEC>;
    };

    template<typename T_STRUCTURE_SPEC, typename T_SGD_PARAMETERS>
    struct SGDSpecification: Specification<T_STRUCTURE_SPEC>{
        using SGD_PARAMETERS = T_SGD_PARAMETERS;
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardSGD<typename Specification<T_STRUCTURE_SPEC>::INPUT_LAYER_SPEC, T_SGD_PARAMETERS>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardSGD<typename Specification<T_STRUCTURE_SPEC>::HIDDEN_LAYER_SPEC, T_SGD_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardSGD<typename Specification<T_STRUCTURE_SPEC>::OUTPUT_LAYER_SPEC, T_SGD_PARAMETERS>;
    };

    template<typename T_STRUCTURE_SPEC, typename T_ADAM_PARAMETERS>
    struct AdamSpecification: Specification<T_STRUCTURE_SPEC>{
        using ADAM_PARAMETERS = T_ADAM_PARAMETERS;
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardAdam<typename Specification<T_STRUCTURE_SPEC>::INPUT_LAYER_SPEC, T_ADAM_PARAMETERS>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardAdam<typename Specification<T_STRUCTURE_SPEC>::HIDDEN_LAYER_SPEC, T_ADAM_PARAMETERS>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardAdam<typename Specification<T_STRUCTURE_SPEC>::OUTPUT_LAYER_SPEC, T_ADAM_PARAMETERS>;
    };

    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE = T_SPEC::BATCH_SIZE>
    struct NeuralNetworkBuffers{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        Matrix<MatrixSpecification<T, TI, BATCH_SIZE, SPEC::HIDDEN_DIM>> tick;
        Matrix<MatrixSpecification<T, TI, BATCH_SIZE, SPEC::HIDDEN_DIM>> tock;
    };
    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE = T_SPEC::BATCH_SIZE>
    struct NeuralNetworkBuffersForwardBackward: NeuralNetworkBuffers<T_SPEC, T_BATCH_SIZE>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        Matrix<MatrixSpecification<T, TI, BATCH_SIZE, SPEC::INPUT_DIM>> d_input;
        Matrix<MatrixSpecification<T, TI, BATCH_SIZE, SPEC::OUTPUT_DIM>> d_output;
    };

    template<typename T_SPEC>
    struct NeuralNetwork{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        template<TI BUFFER_BATCH_SIZE = SPEC::BATCH_SIZE>
        using Buffers = NeuralNetworkBuffers<SPEC, BUFFER_BATCH_SIZE>;
        template<TI BUFFER_BATCH_SIZE = SPEC::BATCH_SIZE>
        using BuffersForwardBackward = NeuralNetworkBuffersForwardBackward<SPEC, BUFFER_BATCH_SIZE>;

        // Convenience
        static_assert(SPEC::STRUCTURE_SPEC::NUM_LAYERS >= 2); // At least input and output layer are required
        static constexpr TI NUM_HIDDEN_LAYERS = SPEC::STRUCTURE_SPEC::NUM_LAYERS - 2;
        static_assert(SPEC::NUM_HIDDEN_LAYERS == NUM_HIDDEN_LAYERS);

        // Interface
        static constexpr TI  INPUT_DIM = SPEC::INPUT_LAYER ::SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_LAYER::SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::INPUT_LAYER::NUM_WEIGHTS + SPEC::HIDDEN_LAYER::NUM_WEIGHTS * NUM_HIDDEN_LAYERS + SPEC::OUTPUT_LAYER::NUM_WEIGHTS;


        // Storage
        typename SPEC:: INPUT_LAYER input_layer;
        typename SPEC::HIDDEN_LAYER hidden_layers[NUM_HIDDEN_LAYERS];
        typename SPEC::OUTPUT_LAYER output_layer;
    };

    template<typename SPEC>
    struct NeuralNetworkBackward: public NeuralNetwork<SPEC>{
    };
    template<typename SPEC>
    struct NeuralNetworkBackwardGradient: public NeuralNetworkBackward<SPEC>{
    };

    template<typename SPEC>
    struct NeuralNetworkSGD: public NeuralNetworkBackwardGradient<SPEC>{
    };

    template<typename SPEC>
    struct NeuralNetworkAdam: public NeuralNetworkBackwardGradient<SPEC>{
        typename SPEC::TI age = 1;
    };


}
#endif
