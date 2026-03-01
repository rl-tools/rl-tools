// BF16 CUDA test for resnet_block
// Compares FP32 generic (CPU) vs BF16 optimized (CUDA)
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
#include <cuda_bf16.h>
#include <gtest/gtest.h>
#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using RNG_CPU = DEVICE_CPU::SPEC::RANDOM::ENGINE<>;
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using T = float;
using T_BF16 = __nv_bfloat16;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TYPE_POLICY_BF16 = rlt::numeric_types::Policy<T,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, T_BF16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, T_BF16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, T_BF16>>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;
using FWD_CAPABILITY = rlt::nn::capability::Forward<>;
using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;

struct ConvTolerances {
    float dw, dg, db;
};
struct ResNetBwdTolerances {
    float d_input;
    ConvTolerances conv1, conv2, downsample;
};

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_evaluate_bf16(const std::string& layer_idx, const std::string& input_name, float fwd_eps) {
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
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

    using OC = typename BLOCK_TYPE_CPU::OUTPUT_SHAPE;
    using OCU = typename BLOCK_TYPE_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> oc; rlt::malloc(device_cpu, oc);
    rlt::evaluate(device_cpu, block_cpu, ic, oc, bc, rng_cpu);
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, OCU>> ocu; rlt::malloc(device_cuda, ocu);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::evaluate(device_cuda, block_cuda, icu, ocu, bcu, rng_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
    rlt::copy(device_cuda, device_cpu, ocu, och);
    T diff = rlt::abs_diff(device_cpu, oc, och) / decltype(oc)::SPEC::SIZE;
    std::cout << "ResNetBlock BF16 EVALUATE FP32(CPU) vs BF16(CUDA): " << diff << std::endl;
    EXPECT_LT(diff, fwd_eps);

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, oc); rlt::free(device_cuda, ocu); rlt::free(device_cpu, och);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_forward_bf16(const std::string& layer_idx, const std::string& input_name, float fwd_eps) {
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
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
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
    std::cout << "ResNetBlock BF16 FORWARD FP32(CPU) vs BF16(CUDA): " << diff << std::endl;
    EXPECT_LT(diff, fwd_eps);

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, och);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

template<typename BLOCK_CONFIG, typename BLOCK_CONFIG_CUDA, TI HEIGHT, TI WIDTH, TI IN_CH>
void test_resnet_block_backward_bf16(const std::string& layer_idx, const std::string& input_name, const ResNetBwdTolerances& tol) {
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
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cpu, block_cpu, ic, bc, rng_cpu, eval_mode);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, block_cuda, icu, bcu, rng_cuda, eval_mode);

    using OC = typename BLOCK_TYPE_CPU::OUTPUT_SHAPE;
    using OCU = typename BLOCK_TYPE_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> doc; rlt::malloc(device_cpu, doc); rlt::set_all(device_cpu, doc, (T)1);
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, OCU>> docu; rlt::malloc(device_cuda, docu); rlt::set_all(device_cuda, docu, (T_BF16)1);
    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> dic; rlt::malloc(device_cpu, dic);
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, BLOCK_INPUT_SHAPE_CUDA>> dicu; rlt::malloc(device_cuda, dicu);

    rlt::zero_gradient(device_cpu, block_cpu);
    rlt::backward_full(device_cpu, block_cpu, ic, doc, dic, bc, eval_mode);
    rlt::zero_gradient(device_cuda, block_cuda);
    rlt::backward_full(device_cuda, block_cuda, icu, docu, dicu, bcu, eval_mode);

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> dich; rlt::malloc(device_cpu, dich);
    rlt::copy(device_cuda, device_cpu, dicu, dich);
    T di = rlt::abs_diff(device_cpu, dic, dich) / decltype(dic)::SPEC::SIZE;
    std::cout << "ResNetBlock BF16 BACKWARD d_input: " << di << std::endl;
    EXPECT_LT(di, tol.d_input);

    BLOCK_TYPE_CPU block_ref; rlt::malloc(device_cpu, block_ref);
    rlt::copy(device_cuda, device_cpu, block_cuda, block_ref);

    auto compare_grad = [&](auto& cpu_grad, auto& cuda_grad, const char* name, float eps) {
        T d = rlt::abs_diff(device_cpu, cpu_grad, cuda_grad) / std::remove_reference_t<decltype(cpu_grad)>::SPEC::SIZE;
        std::cout << "ResNetBlock BF16 BACKWARD " << name << ": " << d << std::endl;
        EXPECT_LT(d, eps);
    };

    compare_grad(block_cpu.conv1.weights.gradient, block_ref.conv1.weights.gradient, "conv1 d_weights", tol.conv1.dw);
    compare_grad(block_cpu.conv1.norm.gamma.gradient, block_ref.conv1.norm.gamma.gradient, "conv1 d_gamma", tol.conv1.dg);
    compare_grad(block_cpu.conv1.norm.beta.gradient, block_ref.conv1.norm.beta.gradient, "conv1 d_beta", tol.conv1.db);
    compare_grad(block_cpu.conv2.weights.gradient, block_ref.conv2.weights.gradient, "conv2 d_weights", tol.conv2.dw);
    compare_grad(block_cpu.conv2.norm.gamma.gradient, block_ref.conv2.norm.gamma.gradient, "conv2 d_gamma", tol.conv2.dg);
    compare_grad(block_cpu.conv2.norm.beta.gradient, block_ref.conv2.norm.beta.gradient, "conv2 d_beta", tol.conv2.db);

    if constexpr(std::remove_reference_t<decltype(block_cpu)>::SPEC::HAS_DOWNSAMPLE) {
        compare_grad(block_cpu.downsample.conv.weights.gradient, block_ref.downsample.conv.weights.gradient, "downsample d_weights", tol.downsample.dw);
        compare_grad(block_cpu.downsample.conv.norm.gamma.gradient, block_ref.downsample.conv.norm.gamma.gradient, "downsample d_gamma", tol.downsample.dg);
        compare_grad(block_cpu.downsample.conv.norm.beta.gradient, block_ref.downsample.conv.norm.beta.gradient, "downsample d_beta", tol.downsample.db);
    }

    rlt::free(device_cpu, block_cpu); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, block_cuda); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, doc); rlt::free(device_cuda, docu);
    rlt::free(device_cpu, dic); rlt::free(device_cuda, dicu);
    rlt::free(device_cpu, dich); rlt::free(device_cpu, block_ref);
    rlt::free(device_cpu, rng_cpu);
}

using BLOCK_64_S1_CONFIG = rlt::nn_models::resnet18::BLOCK_64_S1_CONFIG<TYPE_POLICY, TI>;
using BLOCK_64_S1_CONFIG_CUDA = rlt::nn_models::resnet18::BLOCK_64_S1_CONFIG<TYPE_POLICY_BF16, TI_CUDA>;
using BLOCK_128_S2_CONFIG = rlt::nn_models::resnet18::BLOCK_128_S2_CONFIG<TYPE_POLICY, TI>;
using BLOCK_128_S2_CONFIG_CUDA = rlt::nn_models::resnet18::BLOCK_128_S2_CONFIG<TYPE_POLICY_BF16, TI_CUDA>;

// Tight per-config tolerances (observed values in comments)
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, EVALUATE_64_S1) {
    test_resnet_block_evaluate_bf16<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool", 0.015); // observed: 0.0104
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, EVALUATE_128_S2) {
    test_resnet_block_evaluate_bf16<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1", 0.008); // observed: 0.0056
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, FORWARD_64_S1) {
    test_resnet_block_forward_bf16<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool", 0.015); // observed: 0.0104
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, FORWARD_128_S2) {
    test_resnet_block_forward_bf16<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1", 0.008); // observed: 0.0056
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, BACKWARD_64_S1) {
    // observed: d_input=0.042, conv1 dw=2.96 dg=12.36 db=13.04, conv2 dw=1.71 dg=11.09 db=3.80
    test_resnet_block_backward_bf16<BLOCK_64_S1_CONFIG, BLOCK_64_S1_CONFIG_CUDA, 56, 56, 64>("2", "after_maxpool",
        {0.06, {3.5, 15.0, 16.0}, {2.1, 14.0, 5.0}, {}});
}
TEST(NN_LAYERS_RESNET_BLOCK_CUDA_BF16, BACKWARD_128_S2) {
    // observed: d_input=0.016, conv1 dw=0.59 dg=2.66 db=2.68, conv2 dw=0.16 dg=1.64 db=1.12, ds dw=1.08 dg=1.80 db=1.12
    test_resnet_block_backward_bf16<BLOCK_128_S2_CONFIG, BLOCK_128_S2_CONFIG_CUDA, 56, 56, 64>("4", "after_layer1_block1",
        {0.025, {0.8, 3.5, 3.5}, {0.25, 2.2, 1.5}, {1.4, 2.5, 1.5}});
}
