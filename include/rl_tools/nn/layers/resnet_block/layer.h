#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_RESNET_BLOCK_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_RESNET_BLOCK_LAYER_H
#include "../../../utils/generic/typing.h"
#include "../../../containers/tensor/tensor.h"
#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"
#include "../conv2d/layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::resnet_block {

    template<typename T_TYPE_POLICY, typename T_TI,
             T_TI T_OUTPUT_CHANNELS,
             T_TI T_STRIDE = 1,
             typename T_PARAMETER_GROUP = parameters::groups::Normal>
    struct Configuration{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI OUTPUT_CHANNELS = T_OUTPUT_CHANNELS;
        static constexpr TI STRIDE = T_STRIDE;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
    };

    // Conditional downsample storage (parameterized on the conv layer type, not just the spec)
    template<bool HAS_DOWNSAMPLE, typename CONV_LAYER>
    struct DownsampleStorage {};

    template<typename CONV_LAYER>
    struct DownsampleStorage<true, CONV_LAYER> {
        CONV_LAYER conv;
    };

    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG{
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static_assert(length(INPUT_SHAPE{}) >= 4, "ResnetBlock input must have >= 4 dimensions (...BATCH x H x W x C)");

        static constexpr TI INPUT_HEIGHT = get<length(INPUT_SHAPE{})-3>(INPUT_SHAPE{});
        static constexpr TI INPUT_WIDTH = get<length(INPUT_SHAPE{})-2>(INPUT_SHAPE{});
        static constexpr TI INPUT_CHANNELS = get_last(INPUT_SHAPE{});
        static constexpr TI OUTPUT_CHANNELS = CONFIG::OUTPUT_CHANNELS;
        static constexpr TI STRIDE = CONFIG::STRIDE;

        // Block output spatial dims (determined by first conv: 3x3, stride=STRIDE, pad=1)
        static constexpr TI OUTPUT_HEIGHT = (INPUT_HEIGHT + 2 - 3) / STRIDE + 1;
        static constexpr TI OUTPUT_WIDTH = (INPUT_WIDTH + 2 - 3) / STRIDE + 1;
        static constexpr bool HAS_DOWNSAMPLE = (INPUT_CHANNELS != OUTPUT_CHANNELS) || (STRIDE != 1);

        static constexpr TI INTERNAL_BATCH_SIZE = tensor::shape_math::leading_product(tensor::shape_math::element_to_array<INPUT_SHAPE>(), 3);

        // Conv1: 3x3, stride=STRIDE, pad=1, BN + ReLU
        using CONV1_CONFIG = conv2d::Configuration<TYPE_POLICY, TI, OUTPUT_CHANNELS, 3, 3, STRIDE, STRIDE, 1, 1,
            nn::activation_functions::ActivationFunction::RELU,
            conv2d::Normalization::BATCH_NORM,
            conv2d::DefaultInitializer<TYPE_POLICY, TI>,
            typename CONFIG::PARAMETER_GROUP>;
        using CONV1_SPEC = conv2d::Specification<CONV1_CONFIG, T_CAPABILITY, INPUT_SHAPE>;

        // Conv2: 3x3, stride=1, pad=1, BN + IDENTITY (no activation before skip)
        using CONV1_OUTPUT_SHAPE = typename CONV1_SPEC::OUTPUT_SHAPE;
        using CONV2_CONFIG = conv2d::Configuration<TYPE_POLICY, TI, OUTPUT_CHANNELS, 3, 3, 1, 1, 1, 1,
            nn::activation_functions::ActivationFunction::IDENTITY,
            conv2d::Normalization::BATCH_NORM,
            conv2d::DefaultInitializer<TYPE_POLICY, TI>,
            typename CONFIG::PARAMETER_GROUP>;
        using CONV2_SPEC = conv2d::Specification<CONV2_CONFIG, T_CAPABILITY, CONV1_OUTPUT_SHAPE>;

        // Downsample: 1x1, stride=STRIDE, pad=0, BN + IDENTITY
        using DOWNSAMPLE_CONFIG = conv2d::Configuration<TYPE_POLICY, TI, OUTPUT_CHANNELS, 1, 1, STRIDE, STRIDE, 0, 0,
            nn::activation_functions::ActivationFunction::IDENTITY,
            conv2d::Normalization::BATCH_NORM,
            conv2d::DefaultInitializer<TYPE_POLICY, TI>,
            typename CONFIG::PARAMETER_GROUP>;
        using DOWNSAMPLE_SPEC = conv2d::Specification<DOWNSAMPLE_CONFIG, T_CAPABILITY, INPUT_SHAPE>;

        // Helper: select conv2d layer type matching the outer capability
        template <typename CONV_SPEC>
        using ConvLayerType = typename utils::typing::conditional_t<T_CAPABILITY::TAG == nn::LayerCapability::Forward,
            conv2d::LayerForward<CONV_SPEC>,
            typename utils::typing::conditional_t<T_CAPABILITY::TAG == nn::LayerCapability::Backward,
                conv2d::LayerBackward<CONV_SPEC>,
                conv2d::LayerGradient<CONV_SPEC>>>;

        using CONV1_LAYER = ConvLayerType<CONV1_SPEC>;
        using CONV2_LAYER = ConvLayerType<CONV2_SPEC>;
        using DOWNSAMPLE_LAYER = ConvLayerType<DOWNSAMPLE_SPEC>;

        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY{
            static_assert(length(NEW_INPUT_SHAPE{}) >= 4);
            static constexpr TI NEW_H = get<length(NEW_INPUT_SHAPE{})-3>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_W = get<length(NEW_INPUT_SHAPE{})-2>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_C = get_last(NEW_INPUT_SHAPE{});
            static_assert(NEW_H == INPUT_HEIGHT);
            static_assert(NEW_W == INPUT_WIDTH);
            static_assert(NEW_C == INPUT_CHANNELS);
            static constexpr TI NEW_OH = (NEW_H + 2 - 3) / STRIDE + 1;
            static constexpr TI NEW_OW = (NEW_W + 2 - 3) / STRIDE + 1;
            using SHAPE_HEIGHT = tensor::Replace<NEW_INPUT_SHAPE, NEW_OH, length(NEW_INPUT_SHAPE{}) - 3>;
            using SHAPE_WIDTH = tensor::Replace<SHAPE_HEIGHT, NEW_OW, length(NEW_INPUT_SHAPE{}) - 2>;
            using SHAPE = tensor::Replace<SHAPE_WIDTH, OUTPUT_CHANNELS, length(NEW_INPUT_SHAPE{}) - 1>;
        };
        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;

        static constexpr TI NUM_WEIGHTS = CONV1_SPEC::NUM_WEIGHTS + CONV2_SPEC::NUM_WEIGHTS + (HAS_DOWNSAMPLE ? DOWNSAMPLE_SPEC::NUM_WEIGHTS : 0);
    };

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
        SPEC_1::INPUT_HEIGHT == SPEC_2::INPUT_HEIGHT
        && SPEC_1::INPUT_WIDTH == SPEC_2::INPUT_WIDTH
        && SPEC_1::INPUT_CHANNELS == SPEC_2::INPUT_CHANNELS
        && SPEC_1::OUTPUT_CHANNELS == SPEC_2::OUTPUT_CHANNELS;

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec =
        check_spec_memory<SPEC_1, SPEC_2>
        && SPEC_1::STRIDE == SPEC_2::STRIDE;

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

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct BufferSpecification {
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };
    template<typename T_BUFFER_SPEC>
    struct Buffer{
        using SPEC = typename T_BUFFER_SPEC::SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_BUFFER_SPEC::DYNAMIC_ALLOCATION;
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        using INTERMEDIATE_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::OUTPUT_HEIGHT, SPEC::OUTPUT_WIDTH, SPEC::OUTPUT_CHANNELS>;
        using INTERMEDIATE_SPEC = tensor::Specification<T, TI, INTERMEDIATE_SHAPE, DYNAMIC_ALLOCATION>;
        Tensor<INTERMEDIATE_SPEC> intermediate;
        using SHORTCUT_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::OUTPUT_HEIGHT, SPEC::OUTPUT_WIDTH, SPEC::OUTPUT_CHANNELS>;
        using SHORTCUT_SPEC = tensor::Specification<T, TI, SHORTCUT_SHAPE, DYNAMIC_ALLOCATION>;
        Tensor<SHORTCUT_SPEC> shortcut;
        using D_INPUT_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::INPUT_HEIGHT, SPEC::INPUT_WIDTH, SPEC::INPUT_CHANNELS>;
        using D_INPUT_SPEC = tensor::Specification<T, TI, D_INPUT_SHAPE, DYNAMIC_ALLOCATION>;
        Tensor<D_INPUT_SPEC> d_input_buffer;
        conv2d::Buffer<conv2d::BufferSpecification<typename SPEC::CONV1_SPEC, DYNAMIC_ALLOCATION>> conv1_buffer;
        conv2d::Buffer<conv2d::BufferSpecification<typename SPEC::CONV2_SPEC, DYNAMIC_ALLOCATION>> conv2_buffer;
        conv2d::Buffer<conv2d::BufferSpecification<typename SPEC::DOWNSAMPLE_SPEC, DYNAMIC_ALLOCATION>> downsample_buffer;
    };

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
        static constexpr bool HAS_DOWNSAMPLE = SPEC::HAS_DOWNSAMPLE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        // Internal conv layers (matching the outer capability level)
        typename SPEC::CONV1_LAYER conv1;
        typename SPEC::CONV2_LAYER conv2;
        DownsampleStorage<SPEC::HAS_DOWNSAMPLE, typename SPEC::DOWNSAMPLE_LAYER> downsample;

        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = resnet_block::Buffer<resnet_block::BufferSpecification<SPEC, DYNAMIC_ALLOCATION>>;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = resnet_block::State;
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
        using Layer = nn::layers::resnet_block::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
