// Per-layer CUDA test for conv2d (BN+ReLU stem from ResNet-18)
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
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
static constexpr T FWD_EPSILON = 1e-5;
static constexpr T BWD_EPSILON = 0.05;
using STEM_CONFIG = rlt::nn_models::resnet18::STEM_CONV_CONFIG<TYPE_POLICY, TI>;
using STEM_CONFIG_CUDA = rlt::nn_models::resnet18::STEM_CONV_CONFIG<TYPE_POLICY, TI_CUDA>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 224, 224, 3>;
using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 224, 224, 3>;
using FWD_CAPABILITY = rlt::nn::capability::Forward<>;
using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
using STEM_FWD_CPU = rlt::nn::layers::conv2d::Layer<STEM_CONFIG, FWD_CAPABILITY, INPUT_SHAPE>;
using STEM_FWD_CUDA = rlt::nn::layers::conv2d::Layer<STEM_CONFIG_CUDA, FWD_CAPABILITY, INPUT_SHAPE_CUDA>;
using STEM_GRAD_CPU = rlt::nn::layers::conv2d::Layer<STEM_CONFIG, GRAD_CAPABILITY, INPUT_SHAPE>;
using STEM_GRAD_CUDA = rlt::nn::layers::conv2d::Layer<STEM_CONFIG_CUDA, GRAD_CAPABILITY, INPUT_SHAPE_CUDA>;

TEST(NN_LAYERS_CONV2D_CUDA, EVALUATE) {
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    STEM_FWD_CPU lc; typename STEM_FWD_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");
    auto g = rlt::get_group(device_cpu, lg, "0");
    ASSERT_TRUE(rlt::load(device_cpu, lc, g));
    STEM_FWD_CUDA lcu; typename STEM_FWD_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, lc, lcu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, ic, tg, "input");
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);
    using OC = typename STEM_FWD_CPU::OUTPUT_SHAPE;
    using OCU = typename STEM_FWD_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> oc; rlt::malloc(device_cpu, oc);
    rlt::evaluate(device_cpu, lc, ic, oc, bc, rng_cpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> ocu; rlt::malloc(device_cuda, ocu);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::evaluate(device_cuda, lcu, icu, ocu, bcu, rng_cuda);
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
    rlt::copy(device_cuda, device_cpu, ocu, och);
    T diff = rlt::abs_diff(device_cpu, oc, och) / decltype(oc)::SPEC::SIZE;
    std::cout << "Conv2d EVALUATE CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> exp; rlt::malloc(device_cpu, exp);
    rlt::load(device_cpu, exp, tg, "after_stem");
    T td = rlt::abs_diff(device_cpu, och, exp) / decltype(exp)::SPEC::SIZE;
    std::cout << "Conv2d EVALUATE CUDA vs timm: " << td << std::endl;
    EXPECT_LT(td, FWD_EPSILON);
    rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, oc); rlt::free(device_cuda, ocu);
    rlt::free(device_cpu, och); rlt::free(device_cpu, exp);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

TEST(NN_LAYERS_CONV2D_CUDA, FORWARD) {
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    STEM_GRAD_CPU lc; typename STEM_GRAD_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");
    auto g = rlt::get_group(device_cpu, lg, "0");
    ASSERT_TRUE(rlt::load(device_cpu, lc, g));
    STEM_GRAD_CUDA lcu; typename STEM_GRAD_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, lc, lcu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, ic, tg, "input");
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cpu, lc, ic, bc, rng_cpu, eval_mode);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, lcu, icu, bcu, rng_cuda, eval_mode);
    using OC = typename STEM_GRAD_CPU::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
    rlt::copy(device_cuda, device_cpu, lcu.output, och);
    T diff = rlt::abs_diff(device_cpu, lc.output, och) / decltype(och)::SPEC::SIZE;
    std::cout << "Conv2d FORWARD CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);
    rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, och);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

TEST(NN_LAYERS_CONV2D_CUDA, BACKWARD) {
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    STEM_GRAD_CPU lc; typename STEM_GRAD_CPU::template Buffer<true> bc;
    rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
    auto mg = rlt::get_group(device_cpu, file, "model");
    auto lg = rlt::get_group(device_cpu, mg, "layers");
    auto g = rlt::get_group(device_cpu, lg, "0");
    ASSERT_TRUE(rlt::load(device_cpu, lc, g));
    STEM_GRAD_CUDA lcu; typename STEM_GRAD_CUDA::template Buffer<true> bcu;
    rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);
    rlt::copy(device_cpu, device_cuda, lc, lcu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
    auto tg = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, ic, tg, "input");
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
    rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cpu, lc, ic, bc, rng_cpu, eval_mode);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, lcu, icu, bcu, rng_cuda, eval_mode);
    using OC = typename STEM_GRAD_CPU::OUTPUT_SHAPE;
    using OCU = typename STEM_GRAD_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> doc; rlt::malloc(device_cpu, doc); rlt::set_all(device_cpu, doc, (T)1);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> docu; rlt::malloc(device_cuda, docu); rlt::set_all(device_cuda, docu, (T)1);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> dic; rlt::malloc(device_cpu, dic);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> dicu; rlt::malloc(device_cuda, dicu);
    rlt::zero_gradient(device_cpu, lc);
    rlt::backward_full(device_cpu, lc, ic, doc, dic, bc, eval_mode);
    rlt::zero_gradient(device_cuda, lcu);
    rlt::backward_full(device_cuda, lcu, icu, docu, dicu, bcu, eval_mode);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> dich; rlt::malloc(device_cpu, dich);
    rlt::copy(device_cuda, device_cpu, dicu, dich);
    STEM_GRAD_CPU lch; rlt::malloc(device_cpu, lch);
    rlt::copy(device_cuda, device_cpu, lcu, lch);
    T di = rlt::abs_diff(device_cpu, dic, dich) / decltype(dic)::SPEC::SIZE;
    std::cout << "Conv2d BACKWARD d_input: " << di << std::endl; EXPECT_LT(di, BWD_EPSILON);
    T dw = rlt::abs_diff(device_cpu, lc.weights.gradient, lch.weights.gradient) / decltype(lc.weights.gradient)::SPEC::SIZE;
    std::cout << "Conv2d BACKWARD d_weights: " << dw << std::endl; EXPECT_LT(dw, BWD_EPSILON);
    T dg = rlt::abs_diff(device_cpu, lc.norm.gamma.gradient, lch.norm.gamma.gradient) / decltype(lc.norm.gamma.gradient)::SPEC::SIZE;
    std::cout << "Conv2d BACKWARD d_gamma: " << dg << std::endl; EXPECT_LT(dg, BWD_EPSILON);
    T db = rlt::abs_diff(device_cpu, lc.norm.beta.gradient, lch.norm.beta.gradient) / decltype(lc.norm.beta.gradient)::SPEC::SIZE;
    std::cout << "Conv2d BACKWARD d_beta: " << db << std::endl; EXPECT_LT(db, BWD_EPSILON);
    rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
    rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
    rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
    rlt::free(device_cpu, doc); rlt::free(device_cuda, docu);
    rlt::free(device_cpu, dic); rlt::free(device_cuda, dicu);
    rlt::free(device_cpu, dich); rlt::free(device_cpu, lch);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}
