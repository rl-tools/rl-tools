#pragma once

#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn_models/parallel/model.h>

#include "model_config.h"

namespace rl_tools::rendering::raytracing::yaw_prediction {

    // Teacher encoder dim (must match teacher model's encoder output channels: LATE_CH*2)
    static constexpr unsigned long TEACHER_ENCODER_DIM = ModelConfig<unsigned long>::LATE_CH * 2;

    // Student encoder conv configs (much smaller channels than teacher)
    // 64x64x3 → 32x32x16
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_CONV_16_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 16, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // 32x32x16 → 16x16x32
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_CONV_32_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 32, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // 16x16x32 → 8x8x64
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_CONV_64_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 64, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // 8x8x64 → 4x4x128
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_CONV_128_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 128, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Projection: 1x1 conv from 128 → teacher encoder dim (512), linear (no activation, no BN)
    // This is a learned linear projection to align student features to the teacher's latent space
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_PROJECTION_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, TEACHER_ENCODER_DIM, 1, 1, 1, 1, 0, 0,
        nn::activation_functions::ActivationFunction::IDENTITY,
        nn::layers::conv2d::Normalization::NONE>;

    // Student encoder: 4 small conv layers + 1x1 projection to match teacher encoder dim
    // Output: 4x4xTEACHER_ENCODER_DIM (same spatial and channel dims as teacher encoder)
    template<typename TYPE_POLICY, typename TI>
    using STUDENT_ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<STUDENT_CONV_16_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<STUDENT_CONV_32_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<STUDENT_CONV_64_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<STUDENT_CONV_128_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<STUDENT_PROJECTION_CONFIG<TYPE_POLICY, TI>>
    >;

    // Headless student model: two encoder branches, output is concatenated projected features
    // Output shape: [BATCH_SIZE, 4, 4, 2*TEACHER_ENCODER_DIM] = [BATCH_SIZE, 4, 4, 1024]
    // Matches teacher's concatenated encoder output shape, so teacher's head can be applied directly
    template<typename CAPABILITY, typename TYPE_POLICY, typename TI, TI BATCH_SIZE>
    using STUDENT_MODEL = nn_models::parallel::Build<CAPABILITY,
        STUDENT_ENCODER_MODULE<TYPE_POLICY, TI>,
        STUDENT_ENCODER_MODULE<TYPE_POLICY, TI>,
        tensor::Shape<TI, BATCH_SIZE, 64, 64, 3>,
        tensor::Shape<TI, BATCH_SIZE, 64, 64, 3>
    >;
}
