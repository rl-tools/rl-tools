#pragma once

#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/avg_pool2d/layer.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/layers/dynamic_conv2d/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

#include "model_config.h"

namespace rl_tools::rendering::raytracing::yaw_prediction {

    // --- Conv2d layer config templates parameterized by channel count ---

    template<typename TYPE_POLICY, typename TI, TI OUTPUT_CHANNELS>
    using CONV_3x3_S2_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, OUTPUT_CHANNELS, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    template<typename TYPE_POLICY, typename TI, TI OUTPUT_CHANNELS>
    using CONV_1x1_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, OUTPUT_CHANNELS, 1, 1, 1, 1, 0, 0,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // --- Sub-module configs parameterized by ModelConfig ---

    template<typename TYPE_POLICY, typename TI, typename MC>
    using EARLY_ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<CONV_3x3_S2_CONFIG<TYPE_POLICY, TI, MC::EARLY_CH_1>>,
        nn::layers::conv2d::BindConfiguration<CONV_3x3_S2_CONFIG<TYPE_POLICY, TI, MC::EARLY_CH_2>>
    >;

    template<typename TYPE_POLICY, typename TI>
    using AVGPOOL_CONFIG = nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;

    template<typename TYPE_POLICY, typename TI, typename MC>
    using LATE_ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<CONV_1x1_CONFIG<TYPE_POLICY, TI, MC::LATE_1X1_CH>>,
        nn::layers::conv2d::BindConfiguration<CONV_3x3_S2_CONFIG<TYPE_POLICY, TI, MC::LATE_CH>>
    >;

    template<typename TYPE_POLICY, typename TI, typename MC>
    using HEAD_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<CONV_1x1_CONFIG<TYPE_POLICY, TI, MC::HEAD_1X1_CH>>,
        nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<
            TYPE_POLICY, TI, MC::HEAD_DENSE_CH, nn::activation_functions::ActivationFunction::RELU>>,
        nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<
            TYPE_POLICY, TI, 3, nn::activation_functions::ActivationFunction::IDENTITY>>
    >;

    // --- Model Specification ---

    template<typename T_CAPABILITY, typename T_TYPE_POLICY, typename T_TI, T_TI T_BATCH_SIZE, T_TI T_HEIGHT, T_TI T_WIDTH, typename T_MODEL_CONFIG = ModelConfig<T_TI>>
    struct Specification {
        using CAPABILITY = T_CAPABILITY;
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        using MODEL_CONFIG = T_MODEL_CONFIG;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr TI HEIGHT = T_HEIGHT;
        static constexpr TI WIDTH = T_WIDTH;

        using INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, 3>;

        using EARLY_ENCODER_TYPE = typename EARLY_ENCODER_MODULE<TYPE_POLICY, TI, MODEL_CONFIG>::template Layer<CAPABILITY, INPUT_SHAPE>;
        using EARLY_OUTPUT_SHAPE = typename EARLY_ENCODER_TYPE::OUTPUT_SHAPE;
        static constexpr TI EARLY_CHANNELS = get_last(EARLY_OUTPUT_SHAPE{});

        // mid_conv_a: standard conv on branch A (same channel count as early encoder output)
        using MID_CONV_A_CONFIG = CONV_3x3_S2_CONFIG<TYPE_POLICY, TI, EARLY_CHANNELS>;
        using MID_CONV_A_TYPE = typename nn::layers::conv2d::BindConfiguration<MID_CONV_A_CONFIG>::template Layer<CAPABILITY, EARLY_OUTPUT_SHAPE>;
        using MID_CONV_A_OUTPUT_SHAPE = typename MID_CONV_A_TYPE::OUTPUT_SHAPE;

        // Auto-derive cross-conv kernel size from mid_conv_a output spatial dims
        static constexpr TI RANK = length(MID_CONV_A_OUTPUT_SHAPE{});
        static constexpr TI MID_OUT_H = get<RANK - 3>(MID_CONV_A_OUTPUT_SHAPE{});
        static constexpr TI MID_OUT_W = get<RANK - 2>(MID_CONV_A_OUTPUT_SHAPE{});
        static constexpr TI CROSS_KERNEL_H = MID_OUT_H;
        static constexpr TI CROSS_KERNEL_W = MID_OUT_W;
        static constexpr TI CROSS_PAD_H = (CROSS_KERNEL_H - 2) / 2;
        static constexpr TI CROSS_PAD_W = (CROSS_KERNEL_W - 2) / 2;

        // mid_conv_b: either dynamic_conv2d (cross-conv) or standard conv2d
        using CROSS_CONV_CONFIG = nn::layers::dynamic_conv2d::Configuration<
            TYPE_POLICY, TI, CROSS_KERNEL_H, CROSS_KERNEL_W, 2, 2, CROSS_PAD_H, CROSS_PAD_W,
            nn::activation_functions::ActivationFunction::RELU>;
        using CROSS_CONV_TYPE = nn::layers::dynamic_conv2d::Layer<CROSS_CONV_CONFIG, CAPABILITY, EARLY_OUTPUT_SHAPE>;
        using STANDARD_CONV_B_TYPE = typename nn::layers::conv2d::BindConfiguration<MID_CONV_A_CONFIG>::template Layer<CAPABILITY, EARLY_OUTPUT_SHAPE>;

        using MID_CONV_B_TYPE = utils::typing::conditional_t<MODEL_CONFIG::USE_CROSS_CONV, CROSS_CONV_TYPE, STANDARD_CONV_B_TYPE>;
        using MID_CONV_B_OUTPUT_SHAPE = typename MID_CONV_B_TYPE::OUTPUT_SHAPE;

        // Rank-4 shape for dynamic_conv2d kernel weights (view_memory requires matching rank)
        using KERNEL_WEIGHTS_4D_SHAPE = tensor::Shape<TI, BATCH_SIZE, EARLY_CHANNELS, CROSS_KERNEL_H, CROSS_KERNEL_W>;

        using LATE_ENCODER_TYPE = typename LATE_ENCODER_MODULE<TYPE_POLICY, TI, MODEL_CONFIG>::template Layer<CAPABILITY, MID_CONV_B_OUTPUT_SHAPE>;
        using LATE_OUTPUT_SHAPE = typename LATE_ENCODER_TYPE::OUTPUT_SHAPE;
        static constexpr TI LATE_LAST_DIM = get_last(LATE_OUTPUT_SHAPE{});
        static constexpr auto LATE_RANK = length(LATE_OUTPUT_SHAPE{});
        using CONCAT_SHAPE = tensor::Replace<LATE_OUTPUT_SHAPE, LATE_LAST_DIM * 2, LATE_RANK - 1>;

        using HEAD_TYPE = typename HEAD_MODULE<TYPE_POLICY, TI, MODEL_CONFIG>::template Layer<CAPABILITY, CONCAT_SHAPE>;
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
        typename SPEC::MID_CONV_A_TYPE mid_conv_a;
        typename SPEC::MID_CONV_B_TYPE mid_conv_b;
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
        typename SPEC::MID_CONV_A_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_mid_a;
        typename SPEC::MID_CONV_B_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_mid_b;
        typename SPEC::LATE_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_late_a;
        typename SPEC::LATE_ENCODER_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_late_b;
        typename SPEC::HEAD_TYPE::template Buffer<DYNAMIC_ALLOCATION> buffer_head;

        // Forward saved tensors (contiguous copies for backward)
        using FEATURES_SPEC = tensor::Specification<T, TI, typename SPEC::EARLY_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::EARLY_OUTPUT_SHAPE>>;
        Tensor<FEATURES_SPEC> features_a;
        Tensor<FEATURES_SPEC> features_b;

        // Intermediate tensors for evaluate path (mid-conv outputs)
        using MID_A_OUTPUT_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::MID_CONV_A_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::MID_CONV_A_OUTPUT_SHAPE>>;
        Tensor<MID_A_OUTPUT_TENSOR_SPEC> intermediate_mid_a;
        using MID_B_OUTPUT_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::MID_CONV_B_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::MID_CONV_B_OUTPUT_SHAPE>>;
        Tensor<MID_B_OUTPUT_TENSOR_SPEC> intermediate_mid_b;

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

        using MID_B_BACKWARD_TENSOR_SPEC = tensor::Specification<T, TI, typename SPEC::MID_CONV_B_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::MID_CONV_B_OUTPUT_SHAPE>>;
        Tensor<MID_B_BACKWARD_TENSOR_SPEC> d_mid_a;
        Tensor<MID_B_BACKWARD_TENSOR_SPEC> d_mid_b;

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

    template<typename CAPABILITY, typename TYPE_POLICY, typename TI, TI BATCH_SIZE, TI HEIGHT = 64, TI WIDTH = 64, typename MC = ModelConfig<TI>>
    struct MODEL : BuildModuleType<CAPABILITY, Specification<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE, HEIGHT, WIDTH, MC>>::type {
        template <typename NEW_CAPABILITY>
        using CHANGE_CAPABILITY = MODEL<NEW_CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE, HEIGHT, WIDTH, MC>;
    };
}
