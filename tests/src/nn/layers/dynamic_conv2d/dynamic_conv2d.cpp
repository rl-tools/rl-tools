#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/dynamic_conv2d/operations_generic.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE::index_t;

static constexpr T EPSILON = 1e-10;

template <typename CONV_CONFIG, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI CHANNELS>
void test_dynamic_conv2d_case(const std::string& test_case_name) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_TYPE = rlt::nn::layers::dynamic_conv2d::Layer<CONV_CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER_TYPE layer;
    typename LAYER_TYPE::template Buffer<true> buffer;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);

    constexpr TI OUTPUT_HEIGHT = LAYER_TYPE::OUTPUT_HEIGHT;
    constexpr TI OUTPUT_WIDTH = LAYER_TYPE::OUTPUT_WIDTH;
    constexpr TI KERNEL_HEIGHT = CONV_CONFIG::KERNEL_HEIGHT;
    constexpr TI KERNEL_WIDTH = CONV_CONFIG::KERNEL_WIDTH;

    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS>;
    using KERNEL_WEIGHTS_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH>;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> data, d_data, d_data_expected;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> expected_output, d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, KERNEL_WEIGHTS_SHAPE>> kernel_weights, d_kernel_weights, d_kernel_weights_expected;

    rlt::malloc(device, data);
    rlt::malloc(device, d_data);
    rlt::malloc(device, d_data_expected);
    rlt::malloc(device, expected_output);
    rlt::malloc(device, d_output);
    rlt::malloc(device, kernel_weights);
    rlt::malloc(device, d_kernel_weights);
    rlt::malloc(device, d_kernel_weights_expected);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/dynamic_conv2d_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;

    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);
    auto group = rlt::get_group(device, file, test_case_name);

    rlt::load(device, data, group, "data");
    rlt::load(device, kernel_weights, group, "kernel_weights");
    rlt::load(device, expected_output, group, "output");
    rlt::load(device, d_output, group, "d_output");
    rlt::load(device, d_data_expected, group, "d_data");
    rlt::load(device, d_kernel_weights_expected, group, "d_kernel_weights");

    // ===================== Forward pass =====================
    rlt::forward(device, layer, data, kernel_weights, buffer, rng);

    T forward_diff = rlt::abs_diff(device, expected_output, layer.output) / decltype(expected_output)::SPEC::SIZE;
    std::cout << "  Forward abs_diff (per element): " << forward_diff << std::endl;
    ASSERT_LT(forward_diff, EPSILON) << "Forward pass mismatch for " << test_case_name;

    // ===================== Backward pass =====================
    rlt::backward_full(device, layer, data, kernel_weights, d_output, d_data, d_kernel_weights, buffer);

    T d_data_diff = rlt::abs_diff(device, d_data, d_data_expected) / decltype(d_data)::SPEC::SIZE;
    std::cout << "  d_data abs_diff (per element): " << d_data_diff << std::endl;
    ASSERT_LT(d_data_diff, EPSILON) << "d_data mismatch for " << test_case_name;

    T d_kw_diff = rlt::abs_diff(device, d_kernel_weights, d_kernel_weights_expected) / decltype(d_kernel_weights)::SPEC::SIZE;
    std::cout << "  d_kernel_weights abs_diff (per element): " << d_kw_diff << std::endl;
    ASSERT_LT(d_kw_diff, EPSILON) << "d_kernel_weights mismatch for " << test_case_name;

    rlt::free(device, layer);
    rlt::free(device, buffer);
    rlt::free(device, data);
    rlt::free(device, d_data);
    rlt::free(device, d_data_expected);
    rlt::free(device, expected_output);
    rlt::free(device, d_output);
    rlt::free(device, kernel_weights);
    rlt::free(device, d_kernel_weights);
    rlt::free(device, d_kernel_weights_expected);
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, BASIC_3X3) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 8, 8, 8>("basic_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, PADDED_3X3) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 8, 8, 16>("padded_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, STRIDED_3X3) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 16, 16, 32>("strided_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, POINTWISE_1X1) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 4, 4, 16>("pointwise_1x1");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, RELU_3X3) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_case<CONFIG, 2, 8, 8, 8>("relu_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, STRIDED_RELU) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_dynamic_conv2d_case<CONFIG, 2, 16, 16, 16>("strided_relu");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, LARGE_BATCH) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 8, 16, 16, 64>("large_batch");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, NONSQUARE_3X5) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 5, 1, 1, 1, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 6, 10, 8>("nonsquare_3x5");
}

TEST(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D, SINGLE_CHANNEL) {
    using CONFIG = rlt::nn::layers::dynamic_conv2d::Configuration<TYPE_POLICY, TI, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_dynamic_conv2d_case<CONFIG, 2, 8, 8, 1>("single_channel");
}
