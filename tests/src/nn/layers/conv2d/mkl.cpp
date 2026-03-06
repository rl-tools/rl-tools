#include <gtest/gtest.h>
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/conv2d/operations_cpu_mkl.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE::index_t;

static constexpr T EPSILON = 1e-5;

template <typename CONV_CONFIG, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI INPUT_CHANNELS>
void test_conv2d_mkl_case(const std::string& test_case_name) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, INPUT_CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_TYPE = rlt::nn::layers::conv2d::Layer<CONV_CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER_TYPE layer;
    typename LAYER_TYPE::template Buffer<true> buffer;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);

    constexpr TI OUTPUT_HEIGHT = LAYER_TYPE::OUTPUT_HEIGHT;
    constexpr TI OUTPUT_WIDTH = LAYER_TYPE::OUTPUT_WIDTH;
    constexpr TI OUTPUT_CHANNELS = CONV_CONFIG::OUTPUT_CHANNELS;
    constexpr TI KERNEL_HEIGHT = CONV_CONFIG::KERNEL_HEIGHT;
    constexpr TI KERNEL_WIDTH = CONV_CONFIG::KERNEL_WIDTH;

    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNELS>;
    using WEIGHTS_SHAPE = rlt::tensor::Shape<TI, OUTPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_CHANNELS>;
    using BIASES_SHAPE = rlt::tensor::Shape<TI, OUTPUT_CHANNELS>;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input, d_input, d_input_expected;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> expected_output, d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, WEIGHTS_SHAPE>> d_weights_expected;
    rlt::Tensor<rlt::tensor::Specification<T, TI, BIASES_SHAPE>> d_biases_expected;

    rlt::malloc(device, input);
    rlt::malloc(device, d_input);
    rlt::malloc(device, d_input_expected);
    rlt::malloc(device, expected_output);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_weights_expected);
    rlt::malloc(device, d_biases_expected);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/conv2d_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;

    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);
    auto group = rlt::get_group(device, file, test_case_name);

    rlt::load(device, input, group, "input");
    rlt::load(device, expected_output, group, "output");
    rlt::load(device, layer.weights.parameters, group, "weights");
    rlt::load(device, layer.biases.parameters, group, "biases");
    rlt::load(device, d_output, group, "d_output");
    rlt::load(device, d_input_expected, group, "d_input");
    rlt::load(device, d_weights_expected, group, "d_weights");
    rlt::load(device, d_biases_expected, group, "d_biases");

    ASSERT_FALSE(rlt::is_nan(device, layer.weights.parameters));
    ASSERT_FALSE(rlt::is_nan(device, layer.biases.parameters));

    // ===================== Forward pass =====================
    rlt::forward(device, layer, input, buffer, rng);

    T forward_diff = rlt::abs_diff(device, expected_output, layer.output) / decltype(expected_output)::SPEC::SIZE;
    std::cout << "  Forward abs_diff (per element): " << forward_diff << std::endl;
    ASSERT_LT(forward_diff, EPSILON) << "Forward pass mismatch for " << test_case_name;

    // ===================== Backward pass =====================
    rlt::zero_gradient(device, layer);
    rlt::backward_full(device, layer, input, d_output, d_input, buffer);

    T d_input_diff = rlt::abs_diff(device, d_input, d_input_expected) / decltype(d_input)::SPEC::SIZE;
    std::cout << "  d_input abs_diff (per element): " << d_input_diff << std::endl;
    ASSERT_LT(d_input_diff, EPSILON) << "d_input mismatch for " << test_case_name;

    T d_weights_diff = rlt::abs_diff(device, layer.weights.gradient, d_weights_expected) / decltype(d_weights_expected)::SPEC::SIZE;
    std::cout << "  d_weights abs_diff (per element): " << d_weights_diff << std::endl;
    ASSERT_LT(d_weights_diff, EPSILON) << "d_weights mismatch for " << test_case_name;

    T d_biases_diff = rlt::abs_diff(device, layer.biases.gradient, d_biases_expected) / decltype(d_biases_expected)::SPEC::SIZE;
    std::cout << "  d_biases abs_diff (per element): " << d_biases_diff << std::endl;
    ASSERT_LT(d_biases_diff, EPSILON) << "d_biases mismatch for " << test_case_name;

    rlt::free(device, layer);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, d_input);
    rlt::free(device, d_input_expected);
    rlt::free(device, expected_output);
    rlt::free(device, d_output);
    rlt::free(device, d_weights_expected);
    rlt::free(device, d_biases_expected);
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, BASIC_3X3) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_conv2d_mkl_case<CONFIG, 2, 8, 8, 3>("basic_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, PADDED_3X3) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_conv2d_mkl_case<CONFIG, 2, 8, 8, 3>("padded_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, STRIDED_5X5) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 5, 5, 2, 2, 2, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_conv2d_mkl_case<CONFIG, 2, 16, 16, 8>("strided_5x5");
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, POINTWISE_1X1) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_conv2d_mkl_case<CONFIG, 2, 4, 4, 16>("pointwise_1x1");
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, RELU_3X3) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    test_conv2d_mkl_case<CONFIG, 2, 8, 8, 3>("relu_3x3");
}

TEST(RL_TOOLS_NN_LAYERS_CONV2D_MKL, NONSQUARE_3X5) {
    using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 5, 1, 1, 1, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    test_conv2d_mkl_case<CONFIG, 2, 6, 10, 4>("nonsquare_3x5");
}
