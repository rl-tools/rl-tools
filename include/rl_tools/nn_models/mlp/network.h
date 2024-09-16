#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_NETWORK_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_NETWORK_H

#include "../../nn/nn.h"
#include "../../nn/parameters/parameters.h"
#include "../../nn/optimizers/sgd/sgd.h"
#include "../../nn/optimizers/adam/adam.h"
#include "../../utils/generic/typing.h"
#include "../../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::mlp {
    template <typename T_T, typename T_TI, T_TI T_OUTPUT_DIM, T_TI T_NUM_LAYERS, T_TI T_HIDDEN_DIM, nn::activation_functions::ActivationFunction T_HIDDEN_ACTIVATION_FUNCTION, nn::activation_functions::ActivationFunction T_OUTPUT_ACTIVATION_FUNCTION, typename T_LAYER_INITIALIZER=nn::layers::dense::DefaultInitializer<T_T, T_TI>, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
    struct Configuration{
        using T = T_T;
        using TI = T_TI;
        static constexpr T_TI OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr T_TI NUM_LAYERS = T_NUM_LAYERS; // The input and output layers count towards the total number of layers
        static_assert(NUM_LAYERS >= 2); // At least input and output layer are required
        static constexpr TI NUM_HIDDEN_LAYERS = NUM_LAYERS - 2;
        static constexpr T_TI HIDDEN_DIM = T_HIDDEN_DIM;
        static constexpr auto HIDDEN_ACTIVATION_FUNCTION = T_HIDDEN_ACTIVATION_FUNCTION;
        static constexpr auto OUTPUT_ACTIVATION_FUNCTION = T_OUTPUT_ACTIVATION_FUNCTION;

        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;

        using LAYER_INITIALIZER = T_LAYER_INITIALIZER;

    };
    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification{
        using CONFIG = T_CONFIG;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        static constexpr TI INPUT_DIM = get_last(INPUT_SHAPE{});
        using OUTPUT_SHAPE = tensor::Replace<T_INPUT_SHAPE, CONFIG::OUTPUT_DIM, length(INPUT_SHAPE{})-1>;
        static constexpr TI INTERNAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<tensor::PopBack<INPUT_SHAPE>>{}); // Since the Dense layer is based on Matrices (2D Tensors) the dense layer operation is broadcasted over the leading dimensions. Hence, the actual batch size is the product of all leading dimensions, excluding the last one (containing the features). Since rl_tools::matrix_view is used for zero-cost conversion the INTERNAL_BATCH_SIZE accounts for all leading dimensions.
        static constexpr TI NUM_WEIGHTS = CONFIG::OUTPUT_DIM * INPUT_DIM + CONFIG::OUTPUT_DIM;

        using INPUT_LAYER_CONFIG  = nn::layers::dense::Configuration<T, TI, CONFIG::HIDDEN_DIM, CONFIG::HIDDEN_ACTIVATION_FUNCTION, typename CONFIG::LAYER_INITIALIZER, nn::parameters::groups::Input >;
        using HIDDEN_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, CONFIG::HIDDEN_DIM, CONFIG::HIDDEN_ACTIVATION_FUNCTION, typename CONFIG::LAYER_INITIALIZER, nn::parameters::groups::Normal>;
        using OUTPUT_LAYER_CONFIG = nn::layers::dense::Configuration<T, TI, CONFIG::OUTPUT_DIM, CONFIG::OUTPUT_ACTIVATION_FUNCTION, typename CONFIG::LAYER_INITIALIZER, nn::parameters::groups::Output>;
        using INPUT_LAYER = nn::layers::dense::Layer<INPUT_LAYER_CONFIG, CAPABILITY, INPUT_SHAPE>;
        using HIDDEN_LAYER = nn::layers::dense::Layer<HIDDEN_LAYER_CONFIG, CAPABILITY, typename INPUT_LAYER::SPEC::OUTPUT_SHAPE>;
        using OUTPUT_LAYER = nn::layers::dense::Layer<OUTPUT_LAYER_CONFIG, CAPABILITY, typename HIDDEN_LAYER::SPEC::OUTPUT_SHAPE>;
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

//    template <typename T_CAPABILITY, typename T_SPEC>
//    struct CapabilitySpecification: T_SPEC, T_CAPABILITY{
//        using CAPABILITY = T_CAPABILITY;
//        using PARAMETER_TYPE = typename CAPABILITY::PARAMETER_TYPE;
//    };

    // T_LAYER_PROTOTYPE is any of INPUT_LAYER, HIDDEN_LAYER, or OUTPUT_LAYER. It is assumed that the buffer for all of them are the same (likely empty)
    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_LAYER_PROTOTYPE>
    struct NeuralNetworkBuffersSpecification{
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE_FACTORY::template SHAPE<TI, BATCH_SIZE, SPEC::INPUT_DIM>;
        using OUTPUT_SHAPE = tensor::Replace<INPUT_SHAPE, SPEC::OUTPUT_DIM, length(INPUT_SHAPE{})-1>;
        static constexpr TI ACTUAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<tensor::PopBack<OUTPUT_SHAPE>>{}); // Since the Dense layer is based on Matrices (2D Tensors) the dense layer operation is broadcasted over the leading dimensions. Hence, the actual batch size is the product of all leading dimensions, excluding the last one (containing the features). Since rl_tools::matrix_view is used for zero-cost conversion the ACTUAL_BATCH_SIZE accounts for all leading dimensions.
        static constexpr TI DIM = SPEC::HIDDEN_DIM;
        using LAYER_PROTOTYPE = T_LAYER_PROTOTYPE;
    };

    template<typename T_BUFFER_SPEC>
    struct NeuralNetworkBuffers{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        static constexpr TI ACTUAL_BATCH_SIZE = T_BUFFER_SPEC::ACTUAL_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = utils::typing::conditional_t<SPEC::DYNAMIC_ALLOCATION, MatrixDynamicTag, MatrixStaticTag>;
        using TICK_TOCK_CONTAINER_SPEC = matrix::Specification<T, TI, ACTUAL_BATCH_SIZE, BUFFER_SPEC::DIM, typename SPEC::MEMORY_LAYOUT>;
        using TICK_TOCK_CONTAINER_TYPE = typename CONTAINER_TYPE_TAG::template type<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
        using LayerBuffer = typename BUFFER_SPEC::LAYER_PROTOTYPE::template Buffer<ACTUAL_BATCH_SIZE, SPEC::DYNAMIC_ALLOCATION>;
        LayerBuffer layer_buffer;
    };

    template<typename T_SPEC>
    struct NeuralNetworkForward{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = utils::typing::conditional_t<SPEC::DYNAMIC_ALLOCATION, MatrixDynamicTag, MatrixStaticTag>;
        // Could be dependent on the capability but in this case the buffer-requirements of forward and backward are the same

        // Convenience
        static_assert(SPEC::NUM_LAYERS >= 2); // At least input and output layer are required
        static constexpr TI NUM_HIDDEN_LAYERS = SPEC::NUM_LAYERS - 2;
        static_assert(SPEC::NUM_HIDDEN_LAYERS == NUM_HIDDEN_LAYERS);

        // Interface
        static constexpr TI  INPUT_DIM = SPEC::INPUT_LAYER_SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_LAYER_SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::INPUT_LAYER_SPEC::NUM_WEIGHTS + SPEC::HIDDEN_LAYER_SPEC::NUM_WEIGHTS * NUM_HIDDEN_LAYERS + SPEC::OUTPUT_LAYER_SPEC::NUM_WEIGHTS;

        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE_FACTORY::template SHAPE<TI, SPEC::BATCH_SIZE, INPUT_DIM>;
        using OUTPUT_SHAPE = tensor::Replace<INPUT_SHAPE, OUTPUT_DIM, 2>;

        static constexpr TI ACTUAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<tensor::PopBack<OUTPUT_SHAPE>>{}); // Since the Dense layer is based on Matrices (2D Tensors) the dense layer operation is broadcasted over the leading dimensions. Hence, the actual batch size is the product of all leading dimensions, excluding the last one (containing the features). Since rl_tools::matrix_view is used for zero-cost conversion the ACTUAL_BATCH_SIZE accounts for all leading dimensions.
        struct EXPANDED_BATCH_SIZE_CAPABILITY: SPEC::CAPABILITY{
            static constexpr TI BATCH_SIZE = ACTUAL_BATCH_SIZE;
        };

        // Storage
        typename SPEC::INPUT_LAYER input_layer;
        typename SPEC::HIDDEN_LAYER  hidden_layers[NUM_HIDDEN_LAYERS];
        typename SPEC::OUTPUT_LAYER  output_layer;

        using LAYER_PROTOTYPE = decltype(input_layer);

        template<TI BUFFER_BATCH_SIZE, bool DYNAMIC_ALLOCATION>
        using Buffer = NeuralNetworkBuffers<NeuralNetworkBuffersSpecification<SPEC, BUFFER_BATCH_SIZE, LAYER_PROTOTYPE>>;
    };

    template<typename SPEC>
    struct NeuralNetworkBackward: public NeuralNetworkForward<SPEC>{
//        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
    };
    template<typename SPEC>
    struct NeuralNetworkGradient: public NeuralNetworkBackward<SPEC>{};

    template<typename CONFIG, typename CAPABILITY, typename INPUT_SHAPE>
    using _NeuralNetwork =
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward, NeuralNetworkForward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward, NeuralNetworkBackward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient, NeuralNetworkGradient<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>, void>>>;

    template <typename T_CONFIG>
    struct BindConfiguration{
        template <typename CAPABILITY, typename INPUT_SHAPE>
        using Layer = nn_models::mlp::_NeuralNetwork<T_CONFIG, CAPABILITY, INPUT_SHAPE>;
    };


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
