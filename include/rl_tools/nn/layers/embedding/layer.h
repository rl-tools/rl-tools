#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_EMBEDDING_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_EMBEDDING_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"

//#include "../../../nn/nn.h"
#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::embedding {
//    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output_f(){
//        static_assert(length(typename INPUT_SPEC::SHAPE{}) == 1);
//        static_assert(length(typename OUTPUT_SPEC::SHAPE{}) == 2);
//        static_assert(get<0>(typename INPUT_SPEC::SHAPE{}) == get<0>(typename OUTPUT_SPEC::SHAPE{}));
//        static_assert(get<1>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::OUTPUT_DIM);
//        return true;
//    }
//    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
//    constexpr bool check_input_output = check_input_output_f<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>();
    template <typename T_T, typename T_TI>
    struct StandardNormalSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr T SCALE = 1;
    };
    template<typename SPEC>
    struct StandardNormal {
    };
    template<typename T_T, typename T_TI>
    using DefaultInitializer = StandardNormal<StandardNormalSpecification<T_T, T_TI>>;

    template <typename T_TI>
    struct DefaultInputShapeFactory{
        template <T_TI BATCH_SIZE>
        using SHAPE = tensor::Shape<T_TI, BATCH_SIZE>;
    };
    template<typename T_T, typename T_TI, T_TI T_NUM_CLASSES, T_TI T_OUTPUT_DIM, template <T_TI> typename T_INPUT_SHAPE = DefaultInputShapeFactory<T_TI>::template SHAPE, typename T_INITIALIZER = DefaultInitializer<T_T, T_TI>, typename T_PARAMETER_GROUP=parameters::groups::Input, typename T_CONTAINER_TYPE_TAG = TensorDynamicTag>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI NUM_CLASSES = T_NUM_CLASSES;
        static constexpr TI OUTPUT_DIM = T_OUTPUT_DIM;
        using INITIALIZER = T_INITIALIZER;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        template <TI BATCH_SIZE>
        using INPUT_SHAPE = T_INPUT_SHAPE<BATCH_SIZE>;
        template <TI BATCH_SIZE>
        using OUTPUT_SHAPE = tensor::Append<INPUT_SHAPE<BATCH_SIZE>, OUTPUT_DIM>;
        // Summary
        static constexpr TI NUM_WEIGHTS = OUTPUT_DIM * NUM_CLASSES;
    };
    template <typename T_CAPABILITY, typename T_SPEC>
    struct CapabilitySpecification: T_CAPABILITY, T_SPEC{
        using CAPABILITY = T_CAPABILITY;
    };

    struct Buffer{};

    template<typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
        static constexpr TI NUM_CLASSES = SPEC::NUM_CLASSES;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        using INPUT_SHAPE = typename SPEC::template INPUT_SHAPE<SPEC::BATCH_SIZE>;
        using OUTPUT_SHAPE = typename SPEC::template OUTPUT_SHAPE<SPEC::BATCH_SIZE>;
        using WEIGHTS_SHAPE = tensor::Shape<TI, NUM_CLASSES, OUTPUT_DIM>;
        using WEIGHTS_CONTAINER_SPEC = tensor::Specification<T, TI, WEIGHTS_SHAPE>;
        using WEIGHTS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<WEIGHTS_CONTAINER_SPEC>;
        using WEIGHTS_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<WEIGHTS_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights>;
        typename SPEC::PARAMETER_TYPE::template instance<WEIGHTS_PARAMETER_SPEC> weights;

        template<TI BUFFER_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        using Buffer = embedding::Buffer;
    };
    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC>{
        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
    };
    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC>{
        // This layer supports backpropagation wrt its input but including its weights (for this it stores the intermediate outputs in addition to the pre_activations because they determine the gradient wrt the weights of the following layer)
        using OUTPUT_CONTAINER_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::template OUTPUT_SHAPE<SPEC::BATCH_SIZE>>;
        using OUTPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };
    template<typename CAPABILITY, typename SPEC>
    using Layer =
            typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
                    LayerForward<CapabilitySpecification<CAPABILITY, SPEC>>,
                    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
                            LayerBackward<CapabilitySpecification<CAPABILITY, SPEC>>,
                            typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
                                    LayerGradient<CapabilitySpecification<CAPABILITY, SPEC>>, void>>>;

    template <typename T_SPEC>
    struct BindSpecification{
        template <typename CAPABILITY>
        using Layer = nn::layers::embedding::Layer<CAPABILITY, T_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
