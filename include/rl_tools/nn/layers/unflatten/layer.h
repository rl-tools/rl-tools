#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_UNFLATTEN_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_UNFLATTEN_LAYER_H
#include "../../../utils/generic/typing.h"
#include "../../../containers/tensor/tensor.h"
#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::unflatten {

    template<typename T_TYPE_POLICY, typename T_TI, T_TI T_HEIGHT, T_TI T_WIDTH, T_TI T_CHANNELS>
    struct Configuration{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI HEIGHT = T_HEIGHT;
        static constexpr TI WIDTH = T_WIDTH;
        static constexpr TI CHANNELS = T_CHANNELS;
    };

    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG{
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static constexpr TI HEIGHT = CONFIG::HEIGHT;
        static constexpr TI WIDTH = CONFIG::WIDTH;
        static constexpr TI CHANNELS = CONFIG::CHANNELS;
        static constexpr TI INPUT_DIM = HEIGHT * WIDTH * CHANNELS;
        static_assert(length(INPUT_SHAPE{}) >= 2, "Unflatten input shape must have at least 2 dimensions (...BATCH x H*W*C)");
        static_assert(get_last(INPUT_SHAPE{}) == INPUT_DIM, "Unflatten: last input dimension must equal H*W*C");

        using BATCH_SHAPE = tensor::PopBack<INPUT_SHAPE>;
        static constexpr TI INTERNAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<BATCH_SHAPE>{});

        // Output shape: (...BATCH, H, W, C)
        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY{
            static_assert(length(NEW_INPUT_SHAPE{}) >= 2);
            static_assert(get_last(NEW_INPUT_SHAPE{}) == INPUT_DIM);
            using NEW_BATCH_SHAPE = tensor::PopBack<NEW_INPUT_SHAPE>;
            using SHAPE = tensor::Append<tensor::Append<tensor::Append<NEW_BATCH_SHAPE, HEIGHT>, WIDTH>, CHANNELS>;
        };
        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;
        static constexpr TI NUM_WEIGHTS = 0;
    };

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
        SPEC_1::HEIGHT == SPEC_2::HEIGHT
        && SPEC_1::WIDTH == SPEC_2::WIDTH
        && SPEC_1::CHANNELS == SPEC_2::CHANNELS;

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec = check_spec_memory<SPEC_1, SPEC_2>;

    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output =
        length(typename INPUT_SPEC::SHAPE{}) >= 2 &&
        length(typename OUTPUT_SPEC::SHAPE{}) >= 4 &&
        get_last(typename INPUT_SPEC::SHAPE{}) == LAYER_SPEC::INPUT_DIM &&
        get<length(typename OUTPUT_SPEC::SHAPE{})-3>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::HEIGHT &&
        get<length(typename OUTPUT_SPEC::SHAPE{})-2>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::WIDTH &&
        get_last(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::CHANNELS;

    struct State{};
    struct Buffer{};

    template<typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr TI HEIGHT = SPEC::HEIGHT;
        static constexpr TI WIDTH = SPEC::WIDTH;
        static constexpr TI CHANNELS = SPEC::CHANNELS;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        static constexpr TI INTERNAL_BATCH_SIZE = SPEC::INTERNAL_BATCH_SIZE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;
        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = unflatten::Buffer;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = unflatten::State;
    };

    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC>{};

    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC>{
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        using OUTPUT_CONTAINER_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::HEIGHT, SPEC::WIDTH, SPEC::CHANNELS>;
        using OUTPUT_CONTAINER_SPEC = tensor::Specification<T, TI, OUTPUT_CONTAINER_SHAPE, SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<OUTPUT_CONTAINER_SHAPE>, SPEC::CONST>;
        using OUTPUT_CONTAINER_TYPE = Tensor<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };

    template<typename CONFIG, typename CAPABILITY, typename INPUT_SHAPE>
    using Layer =
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
            LayerForward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
            LayerBackward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
            LayerGradient<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>, void>>>;

    template <typename CONFIG>
    struct BindConfiguration{
        template <typename CAPABILITY, typename INPUT_SHAPE>
        using Layer = nn::layers::unflatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
