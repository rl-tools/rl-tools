// FP32 CUDA test for dynamic_conv2d — CPU generic vs CUDA custom kernels
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dynamic_conv2d/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
#include <gtest/gtest.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using RNG_CPU = DEVICE_CPU::SPEC::RANDOM::ENGINE<>;
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;
static constexpr T FWD_EPSILON = 1e-5;
static constexpr T BWD_EPSILON = 1e-4;

template <typename CONV_CONFIG, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI CHANNELS>
void test_dynamic_conv2d_cuda(const char* label){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);

    using INPUT_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_CPU = rlt::nn::layers::dynamic_conv2d::Layer<CONV_CONFIG, CAPABILITY, INPUT_SHAPE_CPU>;
    using LAYER_CUDA_CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI_CUDA,
        CONV_CONFIG::KERNEL_HEIGHT, CONV_CONFIG::KERNEL_WIDTH,
        CONV_CONFIG::STRIDE_H, CONV_CONFIG::STRIDE_W,
        CONV_CONFIG::PADDING_H, CONV_CONFIG::PADDING_W,
        CONV_CONFIG::ACTIVATION_FUNCTION>;
    using LAYER_CUDA = rlt::nn::layers::dynamic_conv2d::Layer<LAYER_CUDA_CONFIG, CAPABILITY, INPUT_SHAPE_CUDA>;

    constexpr TI OH = LAYER_CPU::OUTPUT_HEIGHT;
    constexpr TI OW = LAYER_CPU::OUTPUT_WIDTH;
    constexpr TI KH = CONV_CONFIG::KERNEL_HEIGHT;
    constexpr TI KW = CONV_CONFIG::KERNEL_WIDTH;

    using OUTPUT_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, OH, OW, CHANNELS>;
    using OUTPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, OH, OW, CHANNELS>;
    using KW_SHAPE_CPU = rlt::tensor::Shape<TI, BATCH_SIZE, CHANNELS, KH, KW>;
    using KW_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CHANNELS, KH, KW>;

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

    // Tensors on CPU
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

    // Tensors on CUDA
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> data_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, KW_SHAPE_CUDA>> kw_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OUTPUT_SHAPE_CUDA>> d_output_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> d_data_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, KW_SHAPE_CUDA>> d_kw_cuda;
    rlt::malloc(device_cuda, data_cuda);
    rlt::malloc(device_cuda, kw_cuda);
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::malloc(device_cuda, d_data_cuda);
    rlt::malloc(device_cuda, d_kw_cuda);

    // Copy inputs to CUDA
    rlt::copy(device_cpu, device_cuda, data_cpu, data_cuda);
    rlt::copy(device_cpu, device_cuda, kw_cpu, kw_cuda);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // ===================== Forward =====================
    rlt::forward(device_cpu, layer_cpu, data_cpu, kw_cpu, buffer_cpu, rng_cpu);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);
    rlt::forward(device_cuda, layer_cuda, data_cuda, kw_cuda, buffer_cuda, rng_cuda);

    // Copy CUDA output back
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, layer_cuda.output, output_cuda_host);

    T fwd_diff = rlt::abs_diff(device_cpu, layer_cpu.output, output_cuda_host) / decltype(output_cuda_host)::SPEC::SIZE;
    std::cout << label << " FORWARD CPU vs CUDA: " << fwd_diff << std::endl;
    EXPECT_LT(fwd_diff, FWD_EPSILON);

    // ===================== Backward =====================
    rlt::backward_full(device_cpu, layer_cpu, data_cpu, kw_cpu, d_output_cpu, d_data_cpu, d_kw_cpu, buffer_cpu);
    rlt::backward_full(device_cuda, layer_cuda, data_cuda, kw_cuda, d_output_cuda, d_data_cuda, d_kw_cuda, buffer_cuda);

    // Copy CUDA backward results back
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> d_data_cuda_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, KW_SHAPE_CPU>> d_kw_cuda_host;
    rlt::malloc(device_cpu, d_data_cuda_host);
    rlt::malloc(device_cpu, d_kw_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_data_cuda, d_data_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_kw_cuda, d_kw_cuda_host);

    T d_data_diff = rlt::abs_diff(device_cpu, d_data_cpu, d_data_cuda_host) / decltype(d_data_cpu)::SPEC::SIZE;
    std::cout << label << " BACKWARD d_data: " << d_data_diff << std::endl;
    EXPECT_LT(d_data_diff, BWD_EPSILON);

    T d_kw_diff = rlt::abs_diff(device_cpu, d_kw_cpu, d_kw_cuda_host) / decltype(d_kw_cpu)::SPEC::SIZE;
    std::cout << label << " BACKWARD d_kw: " << d_kw_diff << std::endl;
    EXPECT_LT(d_kw_diff, BWD_EPSILON);

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

TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA, BASIC_3X3){
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda<CONFIG, 2, 8, 8, 8>("BASIC_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA, PADDED_3X3){
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda<CONFIG, 2, 8, 8, 16>("PADDED_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA, STRIDED_3X3){
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_cuda<CONFIG, 2, 16, 16, 32>("STRIDED_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA, RELU_3X3){
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_cuda<CONFIG, 2, 8, 8, 8>("RELU_3X3");
}
TEST(NN_LAYERS_DYNAMIC_CONV2D_CUDA, STRIDED_RELU){
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_cuda<CONFIG, 2, 16, 16, 16>("STRIDED_RELU");
}
