// Per-layer CUDA test for upsample2d: CPU generic vs CUDA kernel
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/upsample2d/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
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
static constexpr T BWD_EPSILON = 1e-5;

template <typename CONFIG_CPU, typename CONFIG_CUDA, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI CHANNELS>
void test_upsample2d_cuda(const std::string& test_case_name) {
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);

    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using FWD_CAPABILITY = rlt::nn::capability::Forward<>;
    using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;

    using LAYER_FWD_CPU = rlt::nn::layers::upsample2d::Layer<CONFIG_CPU, FWD_CAPABILITY, INPUT_SHAPE>;
    using LAYER_FWD_CUDA = rlt::nn::layers::upsample2d::Layer<CONFIG_CUDA, FWD_CAPABILITY, INPUT_SHAPE_CUDA>;
    using LAYER_GRAD_CPU = rlt::nn::layers::upsample2d::Layer<CONFIG_CPU, GRAD_CAPABILITY, INPUT_SHAPE>;
    using LAYER_GRAD_CUDA = rlt::nn::layers::upsample2d::Layer<CONFIG_CUDA, GRAD_CAPABILITY, INPUT_SHAPE_CUDA>;

    using OC = typename LAYER_FWD_CPU::OUTPUT_SHAPE;
    using OCU = typename LAYER_FWD_CUDA::OUTPUT_SHAPE;

    // Load test data
    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(dp) + "/upsample2d_test_data.h5", HighFive::File::ReadOnly);
    auto group = rlt::get_group(device_cpu, file, test_case_name);

    // EVALUATE test: CPU vs CUDA
    {
        LAYER_FWD_CPU lc; typename LAYER_FWD_CPU::template Buffer<true> bc;
        rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
        LAYER_FWD_CUDA lcu; typename LAYER_FWD_CUDA::template Buffer<true> bcu;
        rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);

        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
        rlt::load(device_cpu, ic, group, "input");
        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
        rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

        rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> oc; rlt::malloc(device_cpu, oc);
        rlt::evaluate(device_cpu, lc, ic, oc, bc, rng_cpu);

        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> ocu; rlt::malloc(device_cuda, ocu);
        rlt::evaluate(device_cuda, lcu, icu, ocu, bcu, rng_cuda);

        rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
        rlt::copy(device_cuda, device_cpu, ocu, och);
        T diff = rlt::abs_diff(device_cpu, oc, och) / decltype(oc)::SPEC::SIZE;
        std::cout << "  Upsample2d EVALUATE CPU vs CUDA (" << test_case_name << "): " << diff << std::endl;
        EXPECT_LT(diff, FWD_EPSILON);

        rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
        rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
        rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
        rlt::free(device_cpu, oc); rlt::free(device_cuda, ocu); rlt::free(device_cpu, och);
    }
    // FORWARD test: CPU vs CUDA (Gradient capability, stores output)
    {
        LAYER_GRAD_CPU lc; typename LAYER_GRAD_CPU::template Buffer<true> bc;
        rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
        LAYER_GRAD_CUDA lcu; typename LAYER_GRAD_CUDA::template Buffer<true> bcu;
        rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);

        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
        rlt::load(device_cpu, ic, group, "input");
        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
        rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

        rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
        rlt::forward(device_cpu, lc, ic, bc, rng_cpu, eval_mode);
        rlt::forward(device_cuda, lcu, icu, bcu, rng_cuda, eval_mode);

        rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> och; rlt::malloc(device_cpu, och);
        rlt::copy(device_cuda, device_cpu, lcu.output, och);
        T diff = rlt::abs_diff(device_cpu, lc.output, och) / decltype(och)::SPEC::SIZE;
        std::cout << "  Upsample2d FORWARD CPU vs CUDA (" << test_case_name << "): " << diff << std::endl;
        EXPECT_LT(diff, FWD_EPSILON);

        rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
        rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
        rlt::free(device_cpu, ic); rlt::free(device_cuda, icu); rlt::free(device_cpu, och);
    }
    // BACKWARD test: CPU vs CUDA
    {
        LAYER_GRAD_CPU lc; typename LAYER_GRAD_CPU::template Buffer<true> bc;
        rlt::malloc(device_cpu, lc); rlt::malloc(device_cpu, bc);
        LAYER_GRAD_CUDA lcu; typename LAYER_GRAD_CUDA::template Buffer<true> bcu;
        rlt::malloc(device_cuda, lcu); rlt::malloc(device_cuda, bcu);

        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> ic; rlt::malloc(device_cpu, ic);
        rlt::load(device_cpu, ic, group, "input");
        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> icu;
        rlt::malloc(device_cuda, icu); rlt::copy(device_cpu, device_cuda, ic, icu);

        rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
        rlt::forward(device_cpu, lc, ic, bc, rng_cpu, eval_mode);
        rlt::forward(device_cuda, lcu, icu, bcu, rng_cuda, eval_mode);

        rlt::Tensor<rlt::tensor::Specification<T, TI, OC>> doc; rlt::malloc(device_cpu, doc);
        rlt::load(device_cpu, doc, group, "d_output");
        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OCU>> docu; rlt::malloc(device_cuda, docu);
        rlt::copy(device_cpu, device_cuda, doc, docu);

        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> dic; rlt::malloc(device_cpu, dic);
        rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> dicu; rlt::malloc(device_cuda, dicu);

        rlt::zero_gradient(device_cpu, lc);
        rlt::backward_full(device_cpu, lc, ic, doc, dic, bc, eval_mode);
        rlt::zero_gradient(device_cuda, lcu);
        rlt::backward_full(device_cuda, lcu, icu, docu, dicu, bcu, eval_mode);

        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> dich; rlt::malloc(device_cpu, dich);
        rlt::copy(device_cuda, device_cpu, dicu, dich);
        T di = rlt::abs_diff(device_cpu, dic, dich) / decltype(dic)::SPEC::SIZE;
        std::cout << "  Upsample2d BACKWARD d_input CPU vs CUDA (" << test_case_name << "): " << di << std::endl;
        EXPECT_LT(di, BWD_EPSILON);

        rlt::free(device_cpu, lc); rlt::free(device_cpu, bc);
        rlt::free(device_cuda, lcu); rlt::free(device_cuda, bcu);
        rlt::free(device_cpu, ic); rlt::free(device_cuda, icu);
        rlt::free(device_cpu, doc); rlt::free(device_cuda, docu);
        rlt::free(device_cpu, dic); rlt::free(device_cuda, dicu); rlt::free(device_cpu, dich);
    }
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

using SCALE2_CONFIG_CPU = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 2, 2>;
using SCALE2_CONFIG_CUDA = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI_CUDA, 2, 2>;
using SCALE3_CONFIG_CPU = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 3, 3>;
using SCALE3_CONFIG_CUDA = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI_CUDA, 3, 3>;
using SCALE4_CONFIG_CPU = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 4, 4>;
using SCALE4_CONFIG_CUDA = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI_CUDA, 4, 4>;

TEST(NN_LAYERS_UPSAMPLE2D_CUDA, SCALE2_SMALL) {
    test_upsample2d_cuda<SCALE2_CONFIG_CPU, SCALE2_CONFIG_CUDA, 2, 4, 4, 3>("scale2_small");
}
TEST(NN_LAYERS_UPSAMPLE2D_CUDA, SCALE2_RECT) {
    test_upsample2d_cuda<SCALE2_CONFIG_CPU, SCALE2_CONFIG_CUDA, 2, 3, 5, 8>("scale2_rect");
}
TEST(NN_LAYERS_UPSAMPLE2D_CUDA, SCALE3) {
    test_upsample2d_cuda<SCALE3_CONFIG_CPU, SCALE3_CONFIG_CUDA, 2, 4, 4, 4>("scale3");
}
TEST(NN_LAYERS_UPSAMPLE2D_CUDA, SCALE4) {
    test_upsample2d_cuda<SCALE4_CONFIG_CPU, SCALE4_CONFIG_CUDA, 2, 3, 3, 3>("scale4");
}
TEST(NN_LAYERS_UPSAMPLE2D_CUDA, SCALE2_MULTICHANNEL) {
    test_upsample2d_cuda<SCALE2_CONFIG_CPU, SCALE2_CONFIG_CUDA, 2, 7, 7, 64>("scale2_multichannel");
}
