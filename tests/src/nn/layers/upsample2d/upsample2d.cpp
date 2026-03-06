#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/upsample2d/operations_generic.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE::index_t;

static constexpr T EPSILON = 1e-10;

template <typename UPSAMPLE_CONFIG, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI CHANNELS>
void test_upsample2d_case(const std::string& test_case_name) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_TYPE = rlt::nn::layers::upsample2d::Layer<UPSAMPLE_CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER_TYPE layer;
    typename LAYER_TYPE::template Buffer<true> buffer;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);

    constexpr TI OUTPUT_HEIGHT = LAYER_TYPE::OUTPUT_HEIGHT;
    constexpr TI OUTPUT_WIDTH = LAYER_TYPE::OUTPUT_WIDTH;

    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS>;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input, d_input, d_input_expected;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> expected_output, d_output;

    rlt::malloc(device, input);
    rlt::malloc(device, d_input);
    rlt::malloc(device, d_input_expected);
    rlt::malloc(device, expected_output);
    rlt::malloc(device, d_output);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/upsample2d_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;

    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);
    auto group = rlt::get_group(device, file, test_case_name);

    rlt::load(device, input, group, "input");
    rlt::load(device, expected_output, group, "output");
    rlt::load(device, d_output, group, "d_output");
    rlt::load(device, d_input_expected, group, "d_input");

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

    // Cleanup
    rlt::free(device, layer);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, d_input);
    rlt::free(device, d_input_expected);
    rlt::free(device, expected_output);
    rlt::free(device, d_output);
}

// ======================== Test cases ========================

using SCALE2_CONFIG = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 2, 2>;
using SCALE3_CONFIG = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 3, 3>;
using SCALE4_CONFIG = rlt::nn::layers::upsample2d::Configuration<TYPE_POLICY, TI, 4, 4>;

TEST(NN_LAYERS_UPSAMPLE2D, SCALE2_SMALL) {
    test_upsample2d_case<SCALE2_CONFIG, 2, 4, 4, 3>("scale2_small");
}
TEST(NN_LAYERS_UPSAMPLE2D, SCALE2_RECT) {
    test_upsample2d_case<SCALE2_CONFIG, 2, 3, 5, 8>("scale2_rect");
}
TEST(NN_LAYERS_UPSAMPLE2D, SCALE3) {
    test_upsample2d_case<SCALE3_CONFIG, 2, 4, 4, 4>("scale3");
}
TEST(NN_LAYERS_UPSAMPLE2D, SCALE4) {
    test_upsample2d_case<SCALE4_CONFIG, 2, 3, 3, 3>("scale4");
}
TEST(NN_LAYERS_UPSAMPLE2D, SCALE2_MULTICHANNEL) {
    test_upsample2d_case<SCALE2_CONFIG, 2, 7, 7, 64>("scale2_multichannel");
}
