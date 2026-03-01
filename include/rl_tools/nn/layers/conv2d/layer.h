#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONV2D_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONV2D_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers/tensor/tensor.h"

#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::conv2d {

    enum class Normalization { NONE, BATCH_NORM, LAYER_NORM };

    template <typename T_TYPE_POLICY, typename T_TI>
    struct KaimingUniformSpecification{
        using T = typename T_TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr bool INIT_LEGACY = true;
        static constexpr T SCALE = 1;
    };
    template<typename SPEC>
    struct KaimingUniform {
    };
    template<typename T_TYPE_POLICY, typename T_TI>
    using DefaultInitializer = KaimingUniform<KaimingUniformSpecification<T_TYPE_POLICY, T_TI>>;


    template<typename T_TYPE_POLICY, typename T_TI,
             T_TI T_OUTPUT_CHANNELS,
             T_TI T_KERNEL_HEIGHT, T_TI T_KERNEL_WIDTH = T_KERNEL_HEIGHT,
             T_TI T_STRIDE_H = 1, T_TI T_STRIDE_W = T_STRIDE_H,
             T_TI T_PADDING_H = 0, T_TI T_PADDING_W = T_PADDING_H,
             nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION = nn::activation_functions::ActivationFunction::IDENTITY,
             Normalization T_NORMALIZATION = Normalization::NONE,
             typename T_INITIALIZER = DefaultInitializer<T_TYPE_POLICY, T_TI>,
             typename T_PARAMETER_GROUP = parameters::groups::Normal>
    struct Configuration{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI OUTPUT_CHANNELS = T_OUTPUT_CHANNELS;
        static constexpr TI KERNEL_HEIGHT = T_KERNEL_HEIGHT;
        static constexpr TI KERNEL_WIDTH = T_KERNEL_WIDTH;
        static constexpr TI STRIDE_H = T_STRIDE_H;
        static constexpr TI STRIDE_W = T_STRIDE_W;
        static constexpr TI PADDING_H = T_PADDING_H;
        static constexpr TI PADDING_W = T_PADDING_W;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
        static constexpr Normalization NORMALIZATION = T_NORMALIZATION;
        using INITIALIZER = T_INITIALIZER;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
        static constexpr float NORM_EPSILON = 1e-5f;
        static constexpr float BN_MOMENTUM = 0.1f;
    };

    // ======================== Normalization forward storage (learnable parameters + running stats) ========================
    template<Normalization NORM, typename SPEC>
    struct NormForward {};

    template<typename SPEC>
    struct NormForward<Normalization::BATCH_NORM, SPEC> {
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using PARAM_SHAPE = tensor::Shape<TI, SPEC::OUTPUT_CHANNELS>;

        using GAMMA_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, PARAM_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST, numeric_types::categories::NormParameter>;
        typename SPEC::PARAMETER_TYPE::template Instance<GAMMA_PARAMETER_SPEC> gamma;

        using BETA_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, PARAM_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST, numeric_types::categories::NormParameter>;
        typename SPEC::PARAMETER_TYPE::template Instance<BETA_PARAMETER_SPEC> beta;

        // Running statistics (non-learnable, plain tensors)
        using T_NORM_STAT = typename TYPE_POLICY::template GET<numeric_types::categories::NormStatistics>;
        using STAT_TENSOR_SPEC = tensor::Specification<T_NORM_STAT, TI, PARAM_SHAPE, SPEC::DYNAMIC_ALLOCATION>;
        Tensor<STAT_TENSOR_SPEC> running_mean;
        Tensor<STAT_TENSOR_SPEC> running_var;
    };

    template<typename SPEC>
    struct NormForward<Normalization::LAYER_NORM, SPEC> {
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using PARAM_SHAPE = tensor::Shape<TI, SPEC::OUTPUT_CHANNELS>;

        using GAMMA_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, PARAM_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST, numeric_types::categories::NormParameter>;
        typename SPEC::PARAMETER_TYPE::template Instance<GAMMA_PARAMETER_SPEC> gamma;

        using BETA_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, PARAM_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST, numeric_types::categories::NormParameter>;
        typename SPEC::PARAMETER_TYPE::template Instance<BETA_PARAMETER_SPEC> beta;
    };

    // ======================== Normalization backward storage (cached statistics) ========================
    template<Normalization NORM, typename SPEC>
    struct NormBackward {};

    template<typename SPEC>
    struct NormBackward<Normalization::BATCH_NORM, SPEC> {
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        using STAT_SHAPE = tensor::Shape<TI, SPEC::OUTPUT_CHANNELS>;
        using STAT_SPEC = tensor::Specification<T, TI, STAT_SHAPE, SPEC::DYNAMIC_ALLOCATION>;
        Tensor<STAT_SPEC> mean;
        Tensor<STAT_SPEC> inv_std;
    };

    template<typename SPEC>
    struct NormBackward<Normalization::LAYER_NORM, SPEC> {
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        using STAT_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE>;
        using STAT_SPEC = tensor::Specification<T, TI, STAT_SHAPE, SPEC::DYNAMIC_ALLOCATION>;
        Tensor<STAT_SPEC> mean;
        Tensor<STAT_SPEC> inv_std;
    };

    // ======================== Specification ========================
    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG{
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static_assert(length(INPUT_SHAPE{}) >= 4, "Conv2d input shape must have at least 4 dimensions (...BATCH x H x W x C)");
        static constexpr TI INPUT_HEIGHT = get<length(INPUT_SHAPE{})-3>(INPUT_SHAPE{});
        static constexpr TI INPUT_WIDTH = get<length(INPUT_SHAPE{})-2>(INPUT_SHAPE{});
        static constexpr TI INPUT_CHANNELS = get_last(INPUT_SHAPE{});
        static constexpr TI OUTPUT_HEIGHT = (INPUT_HEIGHT + 2 * CONFIG::PADDING_H - CONFIG::KERNEL_HEIGHT) / CONFIG::STRIDE_H + 1;
        static constexpr TI OUTPUT_WIDTH = (INPUT_WIDTH + 2 * CONFIG::PADDING_W - CONFIG::KERNEL_WIDTH) / CONFIG::STRIDE_W + 1;

        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY{
            static_assert(length(NEW_INPUT_SHAPE{}) >= 4);
            static constexpr TI NEW_H = get<length(NEW_INPUT_SHAPE{})-3>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_W = get<length(NEW_INPUT_SHAPE{})-2>(NEW_INPUT_SHAPE{});
            static constexpr TI NEW_C = get_last(NEW_INPUT_SHAPE{});
            static_assert(NEW_H == INPUT_HEIGHT);
            static_assert(NEW_W == INPUT_WIDTH);
            static_assert(NEW_C == INPUT_CHANNELS);
            static constexpr TI NEW_OH = (NEW_H + 2 * CONFIG::PADDING_H - CONFIG::KERNEL_HEIGHT) / CONFIG::STRIDE_H + 1;
            static constexpr TI NEW_OW = (NEW_W + 2 * CONFIG::PADDING_W - CONFIG::KERNEL_WIDTH) / CONFIG::STRIDE_W + 1;
            using SHAPE_HEIGHT = tensor::Replace<NEW_INPUT_SHAPE, NEW_OH, length(NEW_INPUT_SHAPE{}) - 3>;
            using SHAPE_WIDTH = tensor::Replace<SHAPE_HEIGHT, NEW_OW, length(NEW_INPUT_SHAPE{}) - 2>;
            using SHAPE = tensor::Replace<SHAPE_WIDTH, CONFIG::OUTPUT_CHANNELS, length(NEW_INPUT_SHAPE{}) - 1>;
        };
        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;
        static constexpr TI INTERNAL_BATCH_SIZE = tensor::shape_math::leading_product(tensor::shape_math::element_to_array<INPUT_SHAPE>(), 3);
        static constexpr TI NUM_WEIGHTS = CONFIG::OUTPUT_CHANNELS * INPUT_CHANNELS * CONFIG::KERNEL_HEIGHT * CONFIG::KERNEL_WIDTH + CONFIG::OUTPUT_CHANNELS
            + (CONFIG::NORMALIZATION != Normalization::NONE ? 2 * CONFIG::OUTPUT_CHANNELS : 0);
    };

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
            SPEC_1::INPUT_CHANNELS == SPEC_2::INPUT_CHANNELS
            && SPEC_1::OUTPUT_CHANNELS == SPEC_2::OUTPUT_CHANNELS
            && SPEC_1::KERNEL_HEIGHT == SPEC_2::KERNEL_HEIGHT
            && SPEC_1::KERNEL_WIDTH == SPEC_2::KERNEL_WIDTH
            && SPEC_1::INPUT_HEIGHT == SPEC_2::INPUT_HEIGHT
            && SPEC_1::INPUT_WIDTH == SPEC_2::INPUT_WIDTH
            && SPEC_1::NORMALIZATION == SPEC_2::NORMALIZATION;

    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec =
        check_spec_memory<SPEC_1, SPEC_2>
        && SPEC_1::ACTIVATION_FUNCTION == SPEC_2::ACTIVATION_FUNCTION
        && SPEC_1::STRIDE_H == SPEC_2::STRIDE_H
        && SPEC_1::STRIDE_W == SPEC_2::STRIDE_W
        && SPEC_1::PADDING_H == SPEC_2::PADDING_H
        && SPEC_1::PADDING_W == SPEC_2::PADDING_W;

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
        static constexpr TI KERNEL_HEIGHT = SPEC::KERNEL_HEIGHT;
        static constexpr TI KERNEL_WIDTH = SPEC::KERNEL_WIDTH;
        static constexpr TI STRIDE_H = SPEC::STRIDE_H;
        static constexpr TI STRIDE_W = SPEC::STRIDE_W;
        static constexpr TI PADDING_H = SPEC::PADDING_H;
        static constexpr TI PADDING_W = SPEC::PADDING_W;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        static constexpr TI INTERNAL_BATCH_SIZE = SPEC::INTERNAL_BATCH_SIZE;
        static constexpr Normalization NORMALIZATION = SPEC::NORMALIZATION;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        // Weights: [OUTPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_CHANNELS] (NHWC filter layout)
        using WEIGHTS_SHAPE = tensor::Shape<TI, OUTPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_CHANNELS>;
        using WEIGHTS_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, WEIGHTS_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<WEIGHTS_PARAMETER_SPEC> weights;

        // Biases: [OUTPUT_CHANNELS]
        using BIASES_SHAPE = tensor::Shape<TI, OUTPUT_CHANNELS>;
        using BIASES_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, BIASES_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<BIASES_PARAMETER_SPEC> biases;

        // Normalization parameters (conditionally populated)
        NormForward<SPEC::NORMALIZATION, SPEC> norm;

        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = conv2d::Buffer;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = conv2d::State;
    };

    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC>{
        using PARENT = LayerForward<SPEC>;
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename SPEC::TI;
        // Pre-activations stores the conv output (before normalization and activation)
        using PRE_ACTIVATIONS_SHAPE = tensor::Shape<TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::OUTPUT_HEIGHT, SPEC::OUTPUT_WIDTH, SPEC::OUTPUT_CHANNELS>;
        using PRE_ACTIVATIONS_SPEC = tensor::Specification<T, TI, PRE_ACTIVATIONS_SHAPE, SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<PRE_ACTIVATIONS_SHAPE>, SPEC::CONST>;
        using PRE_ACTIVATIONS_TYPE = Tensor<PRE_ACTIVATIONS_SPEC>;
        PRE_ACTIVATIONS_TYPE pre_activations;

        // Cached normalization statistics (conditionally populated)
        NormBackward<SPEC::NORMALIZATION, SPEC> norm_cache;
    };

    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC>{
        using PARENT = LayerBackward<SPEC>;
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
        using Layer = nn::layers::conv2d::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
