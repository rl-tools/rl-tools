#pragma once
// ======================== ResNet-18 Model Definition ========================
// Shared between the inference executable and the test suite.
// Uses timm/resnet18.a1_in1k architecture: stem → maxpool → 4 layer groups → avgpool → fc

#include <rl_tools/nn/layers/conv2d/layer.h>
#include <rl_tools/nn/layers/max_pool2d/layer.h>
#include <rl_tools/nn/layers/avg_pool2d/layer.h>
#include <rl_tools/nn/layers/resnet_block/layer.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

namespace rl_tools::nn_models::sequential {
    namespace detail {
        template <int N, typename REMAINING, typename ACC>
        struct TakeFirstImpl;
        template <typename HEAD, typename... TAIL, typename... ACC>
        struct TakeFirstImpl<0, Module<HEAD, TAIL...>, Module<ACC...>> {
            using type = Module<ACC...>;
        };
        template <typename... ACC>
        struct TakeFirstImpl<0, Module<>, Module<ACC...>> {
            using type = Module<ACC...>;
        };
        template <int N, typename HEAD, typename... TAIL, typename... ACC>
        struct TakeFirstImpl<N, Module<HEAD, TAIL...>, Module<ACC...>> {
            static_assert(N > 0);
            using type = typename TakeFirstImpl<N-1, Module<TAIL...>, Module<ACC..., HEAD>>::type;
        };
    }
    template <int N, typename MODULE>
    using TakeFirst = typename detail::TakeFirstImpl<N, MODULE, Module<>>::type;
}

namespace rl_tools::nn_models::resnet18 {
    // Stem: 7x7 conv, stride=2, pad=3, 64 channels, BN+ReLU
    template<typename TYPE_POLICY, typename TI>
    using STEM_CONV_CONFIG = nn::layers::conv2d::Configuration<
        TYPE_POLICY, TI, 64, 7, 7, 2, 2, 3, 3,
        nn::activation_functions::ActivationFunction::RELU,
        nn::layers::conv2d::Normalization::BATCH_NORM>;

    // MaxPool: 3x3, stride=2, pad=1
    template<typename TYPE_POLICY, typename TI>
    using MAXPOOL_CONFIG = nn::layers::max_pool2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1>;

    // ResNet blocks
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_64_S1_CONFIG  = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI,  64, 1>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_128_S2_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 128, 2>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_128_S1_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 128, 1>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_256_S2_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 256, 2>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_256_S1_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 256, 1>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_512_S2_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 512, 2>;
    template<typename TYPE_POLICY, typename TI>
    using BLOCK_512_S1_CONFIG = nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 512, 1>;

    // Global average pool
    template<typename TYPE_POLICY, typename TI>
    using AVGPOOL_CONFIG = nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;

    // FC: 512 -> 1000, identity activation
    template<typename TYPE_POLICY, typename TI>
    using FC_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, 1000,
        nn::activation_functions::ActivationFunction::IDENTITY>;

    template<typename TYPE_POLICY, typename TI>
    using MODULE_CHAIN = nn_models::sequential::Module<
        nn::layers::conv2d::BindConfiguration<STEM_CONV_CONFIG<TYPE_POLICY, TI>>,           // 0: stem
        nn::layers::max_pool2d::BindConfiguration<MAXPOOL_CONFIG<TYPE_POLICY, TI>>,         // 1: maxpool
        nn::layers::resnet_block::BindConfiguration<BLOCK_64_S1_CONFIG<TYPE_POLICY, TI>>,   // 2: layer1.0
        nn::layers::resnet_block::BindConfiguration<BLOCK_64_S1_CONFIG<TYPE_POLICY, TI>>,   // 3: layer1.1
        nn::layers::resnet_block::BindConfiguration<BLOCK_128_S2_CONFIG<TYPE_POLICY, TI>>,  // 4: layer2.0
        nn::layers::resnet_block::BindConfiguration<BLOCK_128_S1_CONFIG<TYPE_POLICY, TI>>,  // 5: layer2.1
        nn::layers::resnet_block::BindConfiguration<BLOCK_256_S2_CONFIG<TYPE_POLICY, TI>>,  // 6: layer3.0
        nn::layers::resnet_block::BindConfiguration<BLOCK_256_S1_CONFIG<TYPE_POLICY, TI>>,  // 7: layer3.1
        nn::layers::resnet_block::BindConfiguration<BLOCK_512_S2_CONFIG<TYPE_POLICY, TI>>,  // 8: layer4.0
        nn::layers::resnet_block::BindConfiguration<BLOCK_512_S1_CONFIG<TYPE_POLICY, TI>>,  // 9: layer4.1
        nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG<TYPE_POLICY, TI>>,         // 10: avgpool
        nn::layers::dense::BindConfiguration<FC_CONFIG<TYPE_POLICY, TI>>                    // 11: fc
    >;

    template<typename TYPE_POLICY, typename TI>
    using INPUT_SHAPE = tensor::Shape<TI, 1, 224, 224, 3>;

    template<typename TYPE_POLICY, typename TI, typename CAPABILITY>
    using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN<TYPE_POLICY, TI>, INPUT_SHAPE<TYPE_POLICY, TI>>;

    // ImageNet normalization constants
    static constexpr double IMAGENET_MEAN[3] = {0.485, 0.456, 0.406};
    static constexpr double IMAGENET_STD[3]  = {0.229, 0.224, 0.225};
}
