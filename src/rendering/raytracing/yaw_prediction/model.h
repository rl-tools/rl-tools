#pragma once

#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/avg_pool2d/layer.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/layers/dynamic_conv2d/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

namespace rl_tools::rendering::raytracing::yaw_prediction {

    // --- Conv2d layer configs ---

    // Conv2d: 3x3, stride=2, pad=1, 16ch, BN+ReLU -> 64x64x16
    template<typename TYPE_POLICY, typename TI>
    using CONV_16_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 16, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Conv2d: 3x3, stride=2, pad=1, 32ch, BN+ReLU -> 32x32x32
    template<typename TYPE_POLICY, typename TI>
    using CONV_32_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 32, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Conv2d: 3x3, stride=2, pad=1, 64ch, BN+ReLU -> 16x16x64
    template<typename TYPE_POLICY, typename TI>
    using CONV_64_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 64, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Conv2d: 3x3, stride=2, pad=1, 256ch, BN+ReLU -> 4x4x256
    template<typename TYPE_POLICY, typename TI>
    using CONV_256_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 256, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // --- Sub-module configs ---

    // Early encoder: CONV_32 -> CONV_64 (64x64x3 -> 16x16x64)
    template<typename TYPE_POLICY, typename TI>
    using EARLY_ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<CONV_32_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<CONV_64_CONFIG<TYPE_POLICY, TI>>
    >;

    template<typename TYPE_POLICY, typename TI>
    using AVGPOOL_CONFIG = nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;

    // Cross-conv: dynamic_conv2d (8x8, stride 2, pad 3, ReLU)
    // Kernel weights are the standard_conv_a output viewed as [BS, C, 8, 8]
    template<typename TYPE_POLICY, typename TI>
    using CROSS_CONV_CONFIG = nn::layers::dynamic_conv2d::Configuration<
        TYPE_POLICY, TI, 8, 8, 2, 2, 3, 3,
        nn::activation_functions::ActivationFunction::RELU>;

    // Late encoder: 1x1 Conv(64->128, BN+ReLU) -> CONV_256(128->256, stride 2, BN+ReLU)
    template<typename TYPE_POLICY, typename TI>
    using LATE_CONV_1x1_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 128, 1, 1, 1, 1, 0, 0,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    template<typename TYPE_POLICY, typename TI>
    using LATE_ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<LATE_CONV_1x1_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<CONV_256_CONFIG<TYPE_POLICY, TI>>
    >;

    // Head: 1x1 Conv(512->256, BN+ReLU) -> AvgPool -> Dense(256->128, ReLU) -> Dense(128->3)
    template<typename TYPE_POLICY, typename TI>
    using HEAD_CONV_1x1_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 256, 1, 1, 1, 1, 0, 0,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    template<typename TYPE_POLICY, typename TI>
    using DENSE_128_CONFIG = nn::layers::dense::Configuration<
        TYPE_POLICY, TI, 128, nn::activation_functions::ActivationFunction::RELU>;

    template<typename TYPE_POLICY, typename TI>
    using DENSE_3_CONFIG = nn::layers::dense::Configuration<
        TYPE_POLICY, TI, 3, nn::activation_functions::ActivationFunction::IDENTITY>;

    template<typename TYPE_POLICY, typename TI>
    using HEAD_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<HEAD_CONV_1x1_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::dense::BindConfiguration<DENSE_128_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::dense::BindConfiguration<DENSE_3_CONFIG<TYPE_POLICY, TI>>
    >;

    // --- Model Specification ---

    template<typename T_CAPABILITY, typename T_TYPE_POLICY, typename T_TI, T_TI T_BATCH_SIZE, T_TI T_HEIGHT, T_TI T_WIDTH>
    struct Specification {
        using CAPABILITY = T_CAPABILITY;
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr TI HEIGHT = T_HEIGHT;
        static constexpr TI WIDTH = T_WIDTH;

        using INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, 3>;

        using EARLY_ENCODER_TYPE = typename EARLY_ENCODER_MODULE<TYPE_POLICY, TI>::template Layer<CAPABILITY, INPUT_SHAPE>;
        using EARLY_OUTPUT_SHAPE = typename EARLY_ENCODER_TYPE::OUTPUT_SHAPE;
        static constexpr TI EARLY_CHANNELS = get_last(EARLY_OUTPUT_SHAPE{});

        using CROSS_CONV_TYPE = nn::layers::dynamic_conv2d::Layer<CROSS_CONV_CONFIG<TYPE_POLICY, TI>, CAPABILITY, EARLY_OUTPUT_SHAPE>;
        using CROSS_CONV_OUTPUT_SHAPE = typename CROSS_CONV_TYPE::OUTPUT_SHAPE;

        using STANDARD_CONV_A_TYPE = typename nn::layers::conv2d::BindConfiguration<CONV_64_CONFIG<TYPE_POLICY, TI>>::template Layer<CAPABILITY, EARLY_OUTPUT_SHAPE>;
        static constexpr TI KERNEL_HEIGHT = CROSS_CONV_CONFIG<TYPE_POLICY, TI>::KERNEL_HEIGHT;
        static constexpr TI KERNEL_WIDTH = CROSS_CONV_CONFIG<TYPE_POLICY, TI>::KERNEL_WIDTH;
        // Rank-4 shape for dynamic_conv2d kernel weights (view_memory requires matching rank)
        using KERNEL_WEIGHTS_4D_SHAPE = tensor::Shape<TI, BATCH_SIZE, EARLY_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH>;

        using LATE_ENCODER_TYPE = typename LATE_ENCODER_MODULE<TYPE_POLICY, TI>::template Layer<CAPABILITY, CROSS_CONV_OUTPUT_SHAPE>;
        using LATE_OUTPUT_SHAPE = typename LATE_ENCODER_TYPE::OUTPUT_SHAPE;
        static constexpr TI LATE_LAST_DIM = get_last(LATE_OUTPUT_SHAPE{});
        static constexpr auto LATE_RANK = length(LATE_OUTPUT_SHAPE{});
        using CONCAT_SHAPE = tensor::Replace<LATE_OUTPUT_SHAPE, LATE_LAST_DIM * 2, LATE_RANK - 1>;

        using HEAD_TYPE = typename HEAD_MODULE<TYPE_POLICY, TI>::template Layer<CAPABILITY, CONCAT_SHAPE>;
        using OUTPUT_SHAPE = typename HEAD_TYPE::OUTPUT_SHAPE;
    };

    // --- Forward declarations ---

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct BufferSpecification;
    template <typename T_BUFFER_SPEC>
    struct Buffer;
    struct State {};

    // --- Module hierarchy ---

    template<typename T_SPEC>
    struct ModuleForward {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        typename SPEC::EARLY_ENCODER_TYPE early_encoder_a;
        typename SPEC::EARLY_ENCODER_TYPE early_encoder_b;
        typename SPEC::STANDARD_CONV_A_TYPE standard_conv_a;
        typename SPEC::CROSS_CONV_TYPE cross_conv_b;
        typename SPEC::LATE_ENCODER_TYPE late_encoder_a;
        typename SPEC::LATE_ENCODER_TYPE late_encoder_b;
        typename SPEC::HEAD_TYPE head;

        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = yaw_prediction::Buffer<yaw_prediction::BufferSpecification<SPEC, DYNAMIC_ALLOCATION>>;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = yaw_prediction::State;
    };

    template<typename T_SPEC>
    struct ModuleBackward: public ModuleForward<T_SPEC> {};

    template<typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC> {
        using TI = typename T_SPEC::TI;
        using T = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using OUTPUT_CONTAINER_SHAPE = typename T_SPEC::OUTPUT_SHAPE;
        using OUTPUT_CONTAINER_SPEC = tensor::Specification<T, TI, OUTPUT_CONTAINER_SHAPE, T_SPEC::CAPABILITY::DYNAMIC_ALLOCATION, tensor::RowMajorStride<OUTPUT_CONTAINER_SHAPE>, T_SPEC::CAPABILITY::CONST>;
        using OUTPUT_CONTAINER_TYPE = Tensor<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };

    // --- Buffer ---

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct BufferSpecification {
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <typename T_BUFFER_SPEC>
    struct Buffer {
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using TI = typename SPEC::TI;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T = typename TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        static constexpr bool DYNAMIC_ALLOCATION = BUFFER_SPEC::DYNAMIC_ALLOCATION;

        // Sub-module buffers
        typename SPEC::EARLY_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_early_a;
        typename SPEC::EARLY_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_early_b;
        typename SPEC::STANDARD_CONV_A_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_standard_conv_a;
        typename SPEC::CROSS_CONV_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_cross_b;
        typename SPEC::LATE_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_late_a;
        typename SPEC::LATE_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_late_b;
        typename SPEC::HEAD_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_head;

        // Forward saved tensors (contiguous copies for backward)
        using FEATURES_SPEC = tensor::Specification<T, TI, typename SPEC::EARLY_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::EARLY_OUTPUT_SHAPE>>;
        Tensor<FEATURES_SPEC> features_a;
        Tensor<FEATURES_SPEC> features_b;

        // Forward intermediates (for concat)
        using LATE_OUTPUT_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::LATE_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::LATE_OUTPUT_SHAPE>>;
        Tensor<LATE_OUTPUT_TENSOR_SPEC> intermediate_a;
        Tensor<LATE_OUTPUT_TENSOR_SPEC> intermediate_b;

        using CONCAT_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::CONCAT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::CONCAT_SHAPE>>;
        Tensor<CONCAT_TENSOR_SPEC> concatenated;

        // Backward temporaries
        Tensor<FEATURES_SPEC> d_features_a;
        Tensor<FEATURES_SPEC> d_features_b;

        using KERNEL_WEIGHTS_4D_SPEC = tensor::Specification<T, TI, typename SPEC::KERNEL_WEIGHTS_4D_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::KERNEL_WEIGHTS_4D_SHAPE>>;
        Tensor<KERNEL_WEIGHTS_4D_SPEC> d_kw_for_b;

        using CROSS_OUTPUT_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::CROSS_CONV_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::CROSS_CONV_OUTPUT_SHAPE>>;
        Tensor<CROSS_OUTPUT_TENSOR_SPEC> d_cross_a;
        Tensor<CROSS_OUTPUT_TENSOR_SPEC> d_cross_b;

        Tensor<CONCAT_TENSOR_SPEC> d_concatenated;
        Tensor<LATE_OUTPUT_TENSOR_SPEC> d_output_a;
        Tensor<LATE_OUTPUT_TENSOR_SPEC> d_output_b;
    };

    // --- Build ---

    template <typename CAPABILITY, typename SPEC>
    struct BuildModuleType {
        using FORWARD = ModuleForward<SPEC>;
        using BACKWARD = ModuleBackward<SPEC>;
        using GRADIENT = ModuleGradient<SPEC>;
        using type = utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward, FORWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward, BACKWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient, GRADIENT, void>>>;
    };

    template<typename CAPABILITY, typename TYPE_POLICY, typename TI, TI BATCH_SIZE, TI HEIGHT = 64, TI WIDTH = 64>
    struct MODEL : BuildModuleType<CAPABILITY, Specification<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE, HEIGHT, WIDTH>>::type {
        template <typename NEW_CAPABILITY>
        using CHANGE_CAPABILITY = MODEL<NEW_CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE, HEIGHT, WIDTH>;
    };
}
