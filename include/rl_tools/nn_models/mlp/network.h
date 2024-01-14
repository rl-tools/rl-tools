#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_NETWORK_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_NETWORK_H

#include "../../nn/nn.h"
#include "../../nn/parameters/parameters.h"
#include "../../nn/optimizers/sgd/sgd.h"
#include "../../nn/optimizers/adam/adam.h"
#include "../../utils/generic/typing.h"
#include "../../containers.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::mlp {
    template <typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM, T_TI T_NUM_LAYERS, T_TI T_HIDDEN_DIM, nn::activation_functions::ActivationFunction T_HIDDEN_ACTIVATION_FUNCTION, nn::activation_functions::ActivationFunction T_OUTPUT_ACTIVATION_FUNCTION, T_TI T_BATCH_SIZE=1, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
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

        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
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


    template <typename T_STRUCTURE_SPEC, typename T_PARAMETER_TYPE>
    struct Specification{
        using STRUCTURE_SPEC = T_STRUCTURE_SPEC;
        using S = STRUCTURE_SPEC;
        using PARAMETER_TYPE = T_PARAMETER_TYPE;
        using T = typename S::T;
        using TI = typename S::TI;
        static constexpr TI NUM_HIDDEN_LAYERS = STRUCTURE_SPEC::NUM_LAYERS - 2;
        static constexpr TI INPUT_DIM = S::INPUT_DIM;
        static constexpr TI HIDDEN_DIM = S::HIDDEN_DIM;
        static constexpr TI OUTPUT_DIM = S::OUTPUT_DIM;
        static constexpr TI BATCH_SIZE = S::BATCH_SIZE;
        using CONTAINER_TYPE_TAG = typename S::CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = S::ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = typename S::MEMORY_LAYOUT;

        using INPUT_LAYER_SPEC  = nn::layers::dense::Specification<T, TI, INPUT_DIM , HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, nn::parameters::groups::Input , CONTAINER_TYPE_TAG, ENFORCE_FLOATING_POINT_TYPE, MEMORY_LAYOUT>;
        using HIDDEN_LAYER_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, S::HIDDEN_ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, nn::parameters::groups::Normal, CONTAINER_TYPE_TAG, ENFORCE_FLOATING_POINT_TYPE, MEMORY_LAYOUT>;
        using OUTPUT_LAYER_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, S::OUTPUT_ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, nn::parameters::groups::Output, CONTAINER_TYPE_TAG, ENFORCE_FLOATING_POINT_TYPE, MEMORY_LAYOUT>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct InferenceSpecification: Specification<T_STRUCTURE_SPEC, nn::parameters::Plain>{
        using SUPER = Specification<T_STRUCTURE_SPEC, nn::parameters::Plain>;
        using  INPUT_LAYER = nn::layers::dense::Layer<typename SUPER::INPUT_LAYER_SPEC >;
        using HIDDEN_LAYER = nn::layers::dense::Layer<typename SUPER::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<typename SUPER::OUTPUT_LAYER_SPEC>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct InferenceBackwardSpecification: Specification<T_STRUCTURE_SPEC, nn::parameters::Plain>{
        using SUPER = Specification<T_STRUCTURE_SPEC, nn::parameters::Plain>;
        using  INPUT_LAYER = nn::layers::dense::LayerBackward<typename SUPER::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackward<typename SUPER::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackward<typename SUPER::OUTPUT_LAYER_SPEC>;
    };

    template <typename T_STRUCTURE_SPEC>
    struct BackwardGradientSpecification: Specification<T_STRUCTURE_SPEC, nn::parameters::Gradient>{
        using SUPER = Specification<T_STRUCTURE_SPEC, nn::parameters::Gradient>;
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::OUTPUT_LAYER_SPEC>;
    };

    template<typename T_STRUCTURE_SPEC>
    struct SGDSpecification: Specification<T_STRUCTURE_SPEC, nn::parameters::SGD>{
        using SUPER = Specification<T_STRUCTURE_SPEC, nn::parameters::SGD>;
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::OUTPUT_LAYER_SPEC>;
    };

    template<typename T_STRUCTURE_SPEC>
    struct AdamSpecification: Specification<T_STRUCTURE_SPEC, nn::parameters::Adam>{
        using SUPER = Specification<T_STRUCTURE_SPEC, nn::parameters::Adam>;
        using  INPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::INPUT_LAYER_SPEC>;
        using HIDDEN_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::HIDDEN_LAYER_SPEC>;
        using OUTPUT_LAYER = nn::layers::dense::LayerBackwardGradient<typename SUPER::OUTPUT_LAYER_SPEC>;
    };

    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct NeuralNetworkBuffersSpecification{
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr TI DIM = SPEC::HIDDEN_DIM;
    };

    template<typename T_BUFFER_SPEC>
    struct NeuralNetworkBuffers{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        using TICK_TOCK_CONTAINER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, BUFFER_SPEC::DIM, typename SPEC::MEMORY_LAYOUT>;
        using TICK_TOCK_CONTAINER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
    };

    template<typename T_SPEC>
    struct NeuralNetwork{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        template<TI BUFFER_BATCH_SIZE = SPEC::BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        using Buffer = NeuralNetworkBuffers<NeuralNetworkBuffersSpecification<SPEC, BUFFER_BATCH_SIZE, T_CONTAINER_TYPE_TAG>>;

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
    struct NeuralNetworkBackward: public NeuralNetwork<SPEC>{};
    template<typename SPEC>
    struct NeuralNetworkBackwardGradient: public NeuralNetworkBackward<SPEC>{};
    template<typename SPEC>
    struct NeuralNetworkSGD: public NeuralNetworkBackwardGradient<SPEC>{};
    template<typename SPEC>
    struct NeuralNetworkAdam: public NeuralNetworkBackwardGradient<SPEC>{};


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
