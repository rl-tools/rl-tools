#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_UPSAMPLE2D_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_UPSAMPLE2D_LAYER_H
#include "../../../utils/generic/typing.h"
#include "../../../containers/tensor/tensor.h"
#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::upsample2d {

    template<typename T_TYPE_POLICY, typename T_TI,
             T_TI T_SCALE_H = 2, T_TI T_SCALE_W = T_SCALE_H>
    struct Configuration{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI SCALE_H = T_SCALE_H;
        static constexpr TI SCALE_W = T_SCALE_W;
    };

    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG{
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static_assert(length(INPUT_SHAPE{}) >= 4, "Upsample2d input shape must have at least 4 dimensions (...BATCH x H x W x C)");
        static constexpr TI INPUT_HEIGHT = get<length(INPUT_SHAPE{})-3>(INPUT_SHAPE{});
        static constexpr TI INPUT_WIDTH = get<length(INPUT_SHAPE{})-2>(INPUT_SHAPE{});
        static constexpr TI INPUT_CHANNELS = get_last(INPUT_SHAPE{});
        static constexpr TI OUTPUT_HEIGHT = INPUT_HEIGHT * CONFIG::SCALE_H;
        static constexpr TI OUTPUT_WIDTH = INPUT_WIDTH * CONFIG::SCALE_W;
        static constexpr TI OUTPUT_CHANNELS = INPUT_CHANNELS;
        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY{
            static_assert(length(NEW_INPUT_SHAPE{}) >= 4);
            static constexpr TI NEW_H = get<length(NEW_INPUT_SHAPE{})-3>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_W = get<length(NEW_INPUT_SHAPE{})-2>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_C = get_last(NEW_INPUT_SHAPE{});
            static_assert(NEW_H == INPUT_HEIGHT);
            static_assert(NEW_W == INPUT_WIDTH);
            static_assert(NEW_C == INPUT_CHANNELS);
            static constexpr TI NEW_OH = NEW_H * CONFIG::SCALE_H;
            static constexpr TI NEW_OW = NEW_W * CONFIG::SCALE_W;
            using SHAPE_HEIGHT = tensor::Replace<NEW_INPUT_SHAPE, NEW_OH, length(NEW_INPUT_SHAPE{}) - 3>;
            using SHAPE = tensor::Replace<SHAPE_HEIGHT, NEW_OW, length(NEW_INPUT_SHAPE{}) - 2>;
        };
        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;
        static constexpr TI INTERNAL_BATCH_SIZE = tensor::shape_math::leading_product(tensor::shape_math::element_to_array<INPUT_SHAPE>(), 3);
        static constexpr TI NUM_WEIGHTS = 0;
    };

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
        SPEC_1::INPUT_HEIGHT == SPEC_2::INPUT_HEIGHT
        && SPEC_1::INPUT_WIDTH == SPEC_2::INPUT_WIDTH
        && SPEC_1::INPUT_CHANNELS == SPEC_2::INPUT_CHANNELS
        && SPEC_1::SCALE_H == SPEC_2::SCALE_H
        && SPEC_1::SCALE_W == SPEC_2::SCALE_W;

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec = check_spec_memory<SPEC_1, SPEC_2>;

    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output =
        length(typename INPUT_SPEC::SHAPE{}) >= 4 &&
        length(typename OUTPUT_SPEC::SHAPE{}) >= 4 &&
        get<length(typename INPUT_SPEC::SHAPE{})-1>(typename INPUT_SPEC::SHAPE{}) == LAYER_SPEC::INPUT_CHANNELS &&
        get<length(typename INPUT_SPEC::SHAPE{})-2>(typename INPUT_SPEC::SHAPE{}) == LAYER_SPEC::INPUT_WIDTH &&
        get<length(typename INPUT_SPEC::SHAPE{})-3>(typename INPUT_SPEC::SHAPE{}) == LAYER_SPEC::INPUT_HEIGHT &&
        get<length(typename OUTPUT_SPEC::SHAPE{})-1>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::OUTPUT_CHANNELS &&
        get<length(typename OUTPUT_SPEC::SHAPE{})-2>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::OUTPUT_WIDTH &&
        get<length(typename OUTPUT_SPEC::SHAPE{})-3>(typename OUTPUT_SPEC::SHAPE{}) == LAYER_SPEC::OUTPUT_HEIGHT;

    struct State{};
    struct Buffer{};

    template<typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_HEIGHT = SPEC::INPUT_HEIGHT;
        static constexpr TI INPUT_WIDTH = SPEC::INPUT_WIDTH;
        static constexpr TI INPUT_CHANNELS = SPEC::INPUT_CHANNELS;
        static constexpr TI OUTPUT_HEIGHT = SPEC::OUTPUT_HEIGHT;
        static constexpr TI OUTPUT_WIDTH = SPEC::OUTPUT_WIDTH;
        static constexpr TI OUTPUT_CHANNELS = SPEC::OUTPUT_CHANNELS;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        static constexpr TI INTERNAL_BATCH_SIZE = SPEC::INTERNAL_BATCH_SIZE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;
        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = upsample2d::Buffer;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = upsample2d::State;
    };

    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC>{};

    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC>{
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        using OUTPUT_CONTAINER_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::OUTPUT_HEIGHT, SPEC::OUTPUT_WIDTH, SPEC::OUTPUT_CHANNELS>;
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
        using Layer = nn::layers::upsample2d::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
