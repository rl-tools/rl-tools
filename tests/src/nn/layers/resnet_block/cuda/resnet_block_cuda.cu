// Per-layer CUDA test for resnet_block
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/resnet/resnet.h>
#include <gtest/gtest.h>
#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using RNG_CPU = DEVICE_CPU::SPEC::RANDOM::ENGINE<>;
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;
static constexpr T FWD_EPSILON = 2e-3;
static constexpr T BWD_EPSILON = 1.0;
using FWD_CAPABILITY = rlt::nn::capability::Forward<>;
using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_evaluate(const std::string& layer_idx, const std::string& input_name, const std::string& output_name) {
    using BLOCK_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_TYPE_CPU = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG, FWD_CAPABILITY, BLOCK_INPUT_SHAPE>;
    using BLOCK_TYPE_CUDA = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG_CUDA, FWD_CAPABILITY, BLOCK_INPUT_SHAPE_CUDA>;

    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");

    BLOCK_TYPE_CPU block_cpu; typename BLOCK_TYPE_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, block_cpu); rlt::malloc(device_cpu, bc);
    auto g = rlt::get_group(device_cpu, lg, layer_idx);
    ASSERT_TRUE(rlt::load(device_cpu, block_cpu, g));

    BLOCK_TYPE_CUDA block_cuda; typename BLOCK_TYPE_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, block_cuda); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, block_cpu, block_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    rlt::load(device_cpu, ic, tg, input_name);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

    using OC = typename BLOCK_TYPE_CPU::OUTPUT_SHAPE;
    using OCU = typename BLOCK_TYPE_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> oc; rlt::malloc(device_cpu, oc);
    rlt::evaluate(device_cpu, block_cpu, ic, oc, bc, rng_cpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> ocu; rlt::malloc(device_cuda, ocu);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::evaluate(device_cuda, block_cuda, icu, ocu, bcu, rng_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
    rlt::copy(device_cuda, device_cpu, ocu, och);
    T diff = rlt::abs_diff(device_cpu, oc, och) / decltype(oc)::SPEC::SIZE;
    std::cout << "ResNetBlock EVALUATE CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> exp; rlt::malloc(device_cpu, exp);
    rlt::load(device_cpu, exp, tg, output_name);
    T td = rlt::abs_diff(device_cpu, och, exp) / decltype(exp)::SPEC::SIZE;
    std::cout << "ResNetBlock EVALUATE CUDA vs timm: " << td << std::endl;
    EXPECT_LT(td, FWD_EPSILON);

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, oc); rlt::free(device_cuda, ocu);
    rlt::free(device_cpu, och); rlt::free(device_cpu, exp);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_forward(const std::string& layer_idx, const std::string& input_name) {
    using BLOCK_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_TYPE_CPU = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG, GRAD_CAPABILITY, BLOCK_INPUT_SHAPE>;
    using BLOCK_TYPE_CUDA = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG_CUDA, GRAD_CAPABILITY, BLOCK_INPUT_SHAPE_CUDA>;

    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");

    BLOCK_TYPE_CPU block_cpu; typename BLOCK_TYPE_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, block_cpu); rlt::malloc(device_cpu, bc);
    auto g = rlt::get_group(device_cpu, lg, layer_idx);
    ASSERT_TRUE(rlt::load(device_cpu, block_cpu, g));

    BLOCK_TYPE_CUDA block_cuda; typename BLOCK_TYPE_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, block_cuda); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, block_cpu, block_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    rlt::load(device_cpu, ic, tg, input_name);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cpu, block_cpu, ic, bc, rng_cpu, eval_mode);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, block_cuda, icu, bcu, rng_cuda, eval_mode);

    using OC = typename BLOCK_TYPE_CPU::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
    rlt::copy(device_cuda, device_cpu, block_cuda.output, och);
    T diff = rlt::abs_diff(device_cpu, block_cpu.output, och) / decltype(och)::SPEC::SIZE;
    std::cout << "ResNetBlock FORWARD CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, och);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_backward(const std::string& layer_idx, const std::string& input_name) {
    using BLOCK_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, HEIGHT, WIDTH, IN_CH>;
    using BLOCK_TYPE_CPU = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG, GRAD_CAPABILITY, BLOCK_INPUT_SHAPE>;
    using BLOCK_TYPE_CUDA = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG_CUDA, GRAD_CAPABILITY, BLOCK_INPUT_SHAPE_CUDA>;

    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");

    BLOCK_TYPE_CPU block_cpu; typename BLOCK_TYPE_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, block_cpu); rlt::malloc(device_cpu, bc);
    auto g = rlt::get_group(device_cpu, lg, layer_idx);
    ASSERT_TRUE(rlt::load(device_cpu, block_cpu, g));

    BLOCK_TYPE_CUDA block_cuda; typename BLOCK_TYPE_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, block_cuda); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, block_cpu, block_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    rlt::load(device_cpu, ic, tg, input_name);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cpu, block_cpu, ic, bc, rng_cpu, eval_mode);

    // Copy CPU forward state to CUDA to ensure backward starts from identical state.
    // This avoids ReLU mask differences from cuDNN/Winograd forward precision.
    rlt::copy(device_cpu, device_cuda, block_cpu, block_cuda);

    using OC = typename BLOCK_TYPE_CPU::OUTPUT_SHAPE;
    using OCU = typename BLOCK_TYPE_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> doc; rlt::malloc(device_cpu, doc); rlt::set_all(device_cpu, doc, (T)1);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> docu; rlt::malloc(device_cuda, docu); rlt::set_all(device_cuda, docu, (T)1);
    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> dic; rlt::malloc(device_cpu, dic);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> dicu; rlt::malloc(device_cuda, dicu);

    rlt::zero_gradient(device_cpu, block_cpu);
    rlt::backward_full(device_cpu, block_cpu, ic, doc, dic, bc, eval_mode);
    rlt::zero_gradient(device_cuda, block_cuda);
    rlt::backward_full(device_cuda, block_cuda, icu, docu, dicu, bcu, eval_mode);

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> dich; rlt::malloc(device_cpu, dich);
    rlt::copy(device_cuda, device_cpu, dicu, dich);
    BLOCK_TYPE_CPU block_cpu_ref; rlt::malloc(device_cpu, block_cpu_ref);
    rlt::copy(device_cuda, device_cpu, block_cuda, block_cpu_ref);

    T di = rlt::abs_diff(device_cpu, dic, dich) / decltype(dic)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD d_input: " << di << std::endl;
    EXPECT_LT(di, BWD_EPSILON);

    T dw1 = rlt::abs_diff(device_cpu, block_cpu.conv1.weights.gradient, block_cpu_ref.conv1.weights.gradient) / decltype(block_cpu.conv1.weights.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv1 d_weights: " << dw1 << std::endl;
    EXPECT_LT(dw1, BWD_EPSILON);
    T dg1 = rlt::abs_diff(device_cpu, block_cpu.conv1.norm.gamma.gradient, block_cpu_ref.conv1.norm.gamma.gradient) / decltype(block_cpu.conv1.norm.gamma.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv1 d_gamma: " << dg1 << std::endl;
    EXPECT_LT(dg1, BWD_EPSILON);
    T db1 = rlt::abs_diff(device_cpu, block_cpu.conv1.norm.beta.gradient, block_cpu_ref.conv1.norm.beta.gradient) / decltype(block_cpu.conv1.norm.beta.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv1 d_beta: " << db1 << std::endl;
    EXPECT_LT(db1, BWD_EPSILON);

    T dw2 = rlt::abs_diff(device_cpu, block_cpu.conv2.weights.gradient, block_cpu_ref.conv2.weights.gradient) / decltype(block_cpu.conv2.weights.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv2 d_weights: " << dw2 << std::endl;
    EXPECT_LT(dw2, BWD_EPSILON);
    T dg2 = rlt::abs_diff(device_cpu, block_cpu.conv2.norm.gamma.gradient, block_cpu_ref.conv2.norm.gamma.gradient) / decltype(block_cpu.conv2.norm.gamma.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv2 d_gamma: " << dg2 << std::endl;
    EXPECT_LT(dg2, BWD_EPSILON);
    T db2 = rlt::abs_diff(device_cpu, block_cpu.conv2.norm.beta.gradient, block_cpu_ref.conv2.norm.beta.gradient) / decltype(block_cpu.conv2.norm.beta.gradient)::SPEC::SIZE;
    std::cout << "ResNetBlock BACKWARD conv2 d_beta: " << db2 << std::endl;
    EXPECT_LT(db2, BWD_EPSILON);

    if constexpr(std::remove_reference_t<decltype(block_cpu)>::SPEC::HAS_DOWNSAMPLE) {
        T dwd = rlt::abs_diff(device_cpu, block_cpu.downsample.conv.weights.gradient, block_cpu_ref.downsample.conv.weights.gradient) / decltype(block_cpu.downsample.conv.weights.gradient)::SPEC::SIZE;
        std::cout << "ResNetBlock BACKWARD downsample d_weights: " << dwd << std::endl;
        EXPECT_LT(dwd, BWD_EPSILON);
        T dgd = rlt::abs_diff(device_cpu, block_cpu.downsample.conv.norm.gamma.gradient, block_cpu_ref.downsample.conv.norm.gamma.gradient) / decltype(block_cpu.downsample.conv.norm.gamma.gradient)::SPEC::SIZE;
        std::cout << "ResNetBlock BACKWARD downsample d_gamma: " << dgd << std::endl;
        EXPECT_LT(dgd, BWD_EPSILON);
        T dbd = rlt::abs_diff(device_cpu, block_cpu.downsample.conv.norm.beta.gradient, block_cpu_ref.downsample.conv.norm.beta.gradient) / decltype(block_cpu.downsample.conv.norm.beta.gradient)::SPEC::SIZE;
        std::cout << "ResNetBlock BACKWARD downsample d_beta: " << dbd << std::endl;
        EXPECT_LT(dbd, BWD_EPSILON);
    }

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, doc); rlt::free(device_cuda, docu);
    rlt::free(device_cpu, dic); rlt::free(device_cuda, dicu);
    rlt::free(device_cpu, dich); rlt::free(device_cpu, block_cpu_ref);
    rlt::free(device_cpu, rng_cpu);
}

using BLOCK_64_S1_CONFIG = rlt::nn_models::resnet18::BLOCK_64_S1_CONFIG<TYPE_POLICY, TI>;
using BLOCK_64_S1_CONFIG_CUDA = rlt::nn_models::resnet18::BLOCK_64_S1_CONFIG<TYPE_POLICY, TI_CUDA>;
using BLOCK_128_S2_CONFIG = rlt::nn_models::resnet18::BLOCK_128_S2_CONFIG<TYPE_POLICY, TI>;
using BLOCK_128_S2_CONFIG_CUDA = rlt::nn_models::resnet18::BLOCK_128_S2_CONFIG<TYPE_POLICY, TI_CUDA>;

TEST(NN_LAYERS_RESNET_BLOCK_CUDA, EVALUATE_64_S1) {
    test_resnet_block_evaluate<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool", "after_layer1_block0");
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA, EVALUATE_128_S2) {
    test_resnet_block_evaluate<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1", "after_layer2_block0");
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA, FORWARD_64_S1) {
    test_resnet_block_forward<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool");
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA, FORWARD_128_S2) {
    test_resnet_block_forward<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1");
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA, BACKWARD_64_S1) {
    test_resnet_block_backward<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool");
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA, BACKWARD_128_S2) {
    test_resnet_block_backward<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1");
}
