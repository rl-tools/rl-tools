#pragma once

#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/avg_pool2d/layer.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn_models/parallel/model.h>

namespace rl_tools::rendering::raytracing::yaw_prediction {
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

    // Conv2d: 3x3, stride=2, pad=1, 128ch, BN+ReLU -> 8x8x128
    template<typename TYPE_POLICY, typename TI>
    using CONV_128_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 128, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Conv2d: 3x3, stride=2, pad=1, 256ch, BN+ReLU -> 4x4x256
    template<typename TYPE_POLICY, typename TI>
    using CONV_256_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 256, 3, 3, 2, 2, 1, 1,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // Global average pool -> 256
    template<typename TYPE_POLICY, typename TI>
    using AVGPOOL_CONFIG = nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;

    // CNN encoder branch: 4 conv layers + global avgpool
    template<typename TYPE_POLICY, typename TI>
    using ENCODER_MODULE = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<CONV_32_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<CONV_64_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<CONV_128_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::conv2d::BindConfiguration<CONV_256_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG<TYPE_POLICY, TI>>
    >;

    // MLP head: Dense 512->128 ReLU, Dense 128->2 Identity
    template<typename TYPE_POLICY, typename TI>
    using DENSE_128_CONFIG = nn::layers::dense::Configuration<
        TYPE_POLICY, TI, 128, nn::activation_functions::ActivationFunction::RELU>;

    template<typename TYPE_POLICY, typename TI>
    using DENSE_2_CONFIG = nn::layers::dense::Configuration<
        TYPE_POLICY, TI, 2, nn::activation_functions::ActivationFunction::IDENTITY>;

    template<typename TYPE_POLICY, typename TI>
    using HEAD_MODULE = nn_models::sequential::Module<
        nn::layers::dense::BindConfiguration<DENSE_128_CONFIG<TYPE_POLICY, TI>>,
        nn::layers::dense::BindConfiguration<DENSE_2_CONFIG<TYPE_POLICY, TI>>
    >;

    // Full parallel model: two CNN encoders + MLP head
    template<typename CAPABILITY, typename TYPE_POLICY, typename TI, TI BATCH_SIZE>
    using MODEL = nn_models::parallel::Build<CAPABILITY,
        ENCODER_MODULE<TYPE_POLICY, TI>,
        ENCODER_MODULE<TYPE_POLICY, TI>,
        tensor::Shape<TI, BATCH_SIZE, 64, 64, 3>,
        tensor::Shape<TI, BATCH_SIZE, 64, 64, 3>,
        HEAD_MODULE<TYPE_POLICY, TI>
    >;
}
