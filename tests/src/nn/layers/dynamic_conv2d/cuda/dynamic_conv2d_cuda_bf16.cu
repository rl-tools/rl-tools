// BF16 CUDA test for dynamic_conv2d — FP32 CPU reference vs BF16 CUDA
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dynamic_conv2d/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>
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
static constexpr float FWD_EPSILON = 0.01;
static constexpr float BWD_D_DATA_EPSILON = 0.1;
static constexpr float BWD_D_KW_EPSILON = 0.1;

template <typename CONV_CONFIG_CPU, typename CONV_CONFIG_CUDA, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI CHANNELS>
void test_dynamic_conv2d_cuda_bf16(const char* label){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);

    using INPUT_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_CPU = rlt::nn::layers::dynamic_conv2d::Layer<CONV_CONFIG_CPU, CAPABILITY, INPUT_SHAPE_CPU>;
    using LAYER_CUDA = rlt::nn::layers::dynamic_conv2d::Layer<CONV_CONFIG_CUDA, CAPABILITY, INPUT_SHAPE_CUDA>;

    constexpr TI OH = LAYER_CPU::OUTPUT_HEIGHT;
    constexpr TI OW = LAYER_CPU::OUTPUT_WIDTH;
    constexpr TI KH = CONV_CONFIG_CPU::KERNEL_HEIGHT;
    constexpr TI KW = CONV_CONFIG_CPU::KERNEL_WIDTH;

    using OUTPUT_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, OH, OW, CHANNELS>;
    using OUTPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, OH, OW, CHANNELS>;
    using KW_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, CHANNELS, KH, KW>;
    using KW_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CHANNELS, KH, KW>;

    using T_ACT_CUDA = typename LAYER_CUDA::SPEC::TYPE_POLICY::template GET<rlt::numeric_types::categories::Activation>;
    using T_GRAD_CUDA = typename LAYER_CUDA::SPEC::TYPE_POLICY::template GET<rlt::numeric_types::categories::Gradient>;

    // CPU layer + buffer
    LAYER_CPU layer_cpu;
    typename LAYER_CPU::template Buffer<true> buffer_cpu;
    rlt::malloc(device_cpu, layer_cpu);
    rlt::malloc(device_cpu, buffer_cpu);

    // CUDA layer + buffer
    LAYER_CUDA layer_cuda;
    typename LAYER_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, layer_cuda);
    rlt::malloc(device_cuda, buffer_cuda);

    // CPU tensors (FP32)
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> data_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, KW_SHAPE_CPU>> kw_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> d_output_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> d_data_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, KW_SHAPE_CPU>> d_kw_cpu;
    rlt::malloc(device_cpu, data_cpu);
    rlt::malloc(device_cpu, kw_cpu);
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cpu, d_data_cpu);
    rlt::malloc(device_cpu, d_kw_cpu);

    // Fill with random values
    rlt::randn(device_cpu, data_cpu, rng_cpu);
    rlt::randn(device_cpu, kw_cpu, rng_cpu);
    rlt::randn(device_cpu, d_output_cpu, rng_cpu);

    // CUDA tensors (BF16)
    rlt::Tensor<rlt::tensor::Specification<T_ACT_CUDA, TI_CUDA, INPUT_SHAPE_CUDA>> data_cuda;
    rlt::Tensor<rlt::tensor::Specification<T_ACT_CUDA, TI_CUDA, KW_SHAPE_CUDA>> kw_cuda;
    rlt::Tensor<rlt::tensor::Specification<T_GRAD_CUDA, TI_CUDA, OUTPUT_SHAPE_CUDA>> d_output_cuda;
    rlt::Tensor<rlt::tensor::Specification<T_GRAD_CUDA, TI_CUDA, INPUT_SHAPE_CUDA>> d_data_cuda;
    rlt::Tensor<rlt::tensor::Specification<T_GRAD_CUDA, TI_CUDA, KW_SHAPE_CUDA>> d_kw_cuda;
    rlt::malloc(device_cuda, data_cuda);
    rlt::malloc(device_cuda, kw_cuda);
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::malloc(device_cuda, d_data_cuda);
    rlt::malloc(device_cuda, d_kw_cuda);

    // Copy inputs to CUDA (FP32 CPU -> BF16 CUDA)
    rlt::copy(device_cpu, device_cuda, data_cpu, data_cuda);
    rlt::copy(device_cpu, device_cuda, kw_cpu, kw_cuda);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // ===================== Forward =====================
    rlt::forward(device_cpu, layer_cpu, data_cpu, kw_cpu, buffer_cpu, rng_cpu);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, layer_cuda, data_cuda, kw_cuda, buffer_cuda, rng_cuda);

    // Copy CUDA output back (BF16 -> FP32)
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, layer_cuda.output, output_cuda_host);

    T fwd_diff = rlt::abs_diff(device_cpu, layer_cpu.output, output_cuda_host) / decltype(output_cuda_host)::SPEC::SIZE;
    std::cout << label << " BF16 FORWARD FP32(CPU) vs BF16(CUDA): " << fwd_diff << std::endl;
    EXPECT_LT(fwd_diff, FWD_EPSILON);

    // ===================== Backward =====================
    rlt::backward_full(device_cpu, layer_cpu, data_cpu, kw_cpu, d_output_cpu, d_data_cpu, d_kw_cpu, buffer_cpu);
    rlt::backward_full(device_cuda, layer_cuda, data_cuda, kw_cuda, d_output_cuda, d_data_cuda, d_kw_cuda, buffer_cuda);

    // Copy CUDA backward results back (BF16 -> FP32)
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> d_data_cuda_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, KW_SHAPE_CPU>> d_kw_cuda_host;
    rlt::malloc(device_cpu, d_data_cuda_host);
    rlt::malloc(device_cpu, d_kw_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_data_cuda, d_data_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_kw_cuda, d_kw_cuda_host);

    T d_data_diff = rlt::abs_diff(device_cpu, d_data_cpu, d_data_cuda_host) / decltype(d_data_cpu)::SPEC::SIZE;
    std::cout << label << " BF16 BACKWARD d_data: " << d_data_diff << std::endl;
    EXPECT_LT(d_data_diff, BWD_D_DATA_EPSILON);

    T d_kw_diff = rlt::abs_diff(device_cpu, d_kw_cpu, d_kw_cuda_host) / decltype(d_kw_cpu)::SPEC::SIZE;
    std::cout << label << " BF16 BACKWARD d_kw: " << d_kw_diff << std::endl;
    EXPECT_LT(d_kw_diff, BWD_D_KW_EPSILON);

    // Cleanup
    rlt::free(device_cpu, layer_cpu); rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, layer_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, data_cpu); rlt::free(device_cpu, kw_cpu);
    rlt::free(device_cpu, d_output_cpu); rlt::free(device_cpu, d_data_cpu); rlt::free(device_cpu, d_kw_cpu);
    rlt::free(device_cuda, data_cuda); rlt::free(device_cuda, kw_cuda);
    rlt::free(device_cuda, d_output_cuda); rlt::free(device_cuda, d_data_cuda); rlt::free(device_cuda, d_kw_cuda);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cpu, d_data_cuda_host); rlt::free(device_cpu, d_kw_cuda_host);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA_BF16, BASIC_3X3){
    using CONFIG_CPU = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONFIG_CUDA = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY_BF16, TI_CUDA, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda_bf16<CONFIG_CPU, CONFIG_CUDA, 2, 8, 8, 8>("BASIC_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA_BF16, PADDED_3X3){
    using CONFIG_CPU = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONFIG_CUDA = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY_BF16, TI_CUDA, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda_bf16<CONFIG_CPU, CONFIG_CUDA, 2, 8, 8, 16>("PADDED_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA_BF16, STRIDED_3X3){
    using CONFIG_CPU = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONFIG_CUDA = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY_BF16, TI_CUDA, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda_bf16<CONFIG_CPU, CONFIG_CUDA, 2, 16, 16, 32>("STRIDED_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA_BF16, RELU_3X3){
    using CONFIG_CPU = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONFIG_CUDA = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY_BF16, TI_CUDA, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_cuda_bf16<CONFIG_CPU, CONFIG_CUDA, 2, 8, 8, 8>("RELU_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA_BF16, STRIDED_RELU){
    using CONFIG_CPU = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONFIG_CUDA = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY_BF16, TI_CUDA, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_cuda_bf16<CONFIG_CPU, CONFIG_CUDA, 2, 16, 16, 16>("STRIDED_RELU");
}
