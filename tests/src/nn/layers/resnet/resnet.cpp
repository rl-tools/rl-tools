#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

// Persist
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE::index_t;

static constexpr T EPSILON = 1e-8;

// ======================== ResNet-18 Model Definition ========================
// Stem: 7x7 conv, stride=2, pad=3, 64 channels, BN+ReLU
using STEM_CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<
    TYPE_POLICY, TI, 64, 7, 7, 2, 2, 3, 3,
    rlt::nn::activation_functions::ActivationFunction::RELU,
    rlt::nn::layers::conv2d::Normalization::BATCH_NORM>;

// MaxPool: 3x3, stride=2, pad=1
using MAXPOOL_CONFIG = rlt::nn::layers::max_pool2d::Configuration<TYPE_POLICY, TI, 3, 3, 2, 2, 1, 1>;

// ResNet blocks
using BLOCK_64_S1_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 64, 1>;
using BLOCK_128_S2_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 128, 2>;
using BLOCK_128_S1_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 128, 1>;
using BLOCK_256_S2_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 256, 2>;
using BLOCK_256_S1_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 256, 1>;
using BLOCK_512_S2_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 512, 2>;
using BLOCK_512_S1_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 512, 1>;

// Global average pool
using AVGPOOL_CONFIG = rlt::nn::layers::avg_pool2d::Configuration<TYPE_POLICY, TI>;

// FC: 512 -> 1000, identity activation
using FC_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1000,
    rlt::nn::activation_functions::ActivationFunction::IDENTITY>;

// Sequential chain
template<typename C, typename N = rlt::nn_models::sequential::OutputModule>
using Module = rlt::nn_models::sequential::Module<C, N>;

using MODULE_CHAIN =
    Module<rlt::nn::layers::conv2d::BindConfiguration<STEM_CONV_CONFIG>,       // 0: stem
    Module<rlt::nn::layers::max_pool2d::BindConfiguration<MAXPOOL_CONFIG>,     // 1: maxpool
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_64_S1_CONFIG>,  // 2: layer1.0
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_64_S1_CONFIG>,  // 3: layer1.1
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_128_S2_CONFIG>, // 4: layer2.0
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_128_S1_CONFIG>, // 5: layer2.1
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_256_S2_CONFIG>, // 6: layer3.0
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_256_S1_CONFIG>, // 7: layer3.1
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_512_S2_CONFIG>, // 8: layer4.0
    Module<rlt::nn::layers::resnet_block::BindConfiguration<BLOCK_512_S1_CONFIG>, // 9: layer4.1
    Module<rlt::nn::layers::avg_pool2d::BindConfiguration<AVGPOOL_CONFIG>,     // 10: avgpool
    Module<rlt::nn::layers::dense::BindConfiguration<FC_CONFIG>                // 11: fc
    >>>>>>>>>>>>;

using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 224, 224, 3>;
using CAPABILITY = rlt::nn::capability::Forward<>;
using RESNET18 = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

// ======================== Test: Full model forward pass ========================
TEST(RESNET18, FULL_FORWARD) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    // Load test data
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // Create and malloc model
    RESNET18 model;
    typename RESNET18::template Buffer<true> buffer;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    // Load model weights
    auto model_group = rlt::get_group(device, file, "model");
    bool load_success = rlt::load(device, model, model_group);
    ASSERT_TRUE(load_success) << "Failed to load model weights";

    // Load input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::malloc(device, input);
    auto test_group = rlt::get_group(device, file, "test_data");
    rlt::load(device, input, test_group, "input");

    // Evaluate
    using OUTPUT_SHAPE = typename RESNET18::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output, expected_output;
    rlt::malloc(device, output);
    rlt::malloc(device, expected_output);

    rlt::load(device, expected_output, test_group, "output");

    rlt::evaluate(device, model, input, output, buffer, rng);

    T diff = rlt::abs_diff(device, output, expected_output) / decltype(output)::SPEC::SIZE;
    std::cout << "Full model forward diff (per element): " << diff << std::endl;
    ASSERT_LT(diff, EPSILON) << "Full model forward pass mismatch";

    // Print top-5 predictions
    auto output_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device, output);
    std::cout << "Top predictions:" << std::endl;
    for(int k = 0; k < 5; k++){
        T max_val = -1e30;
        TI max_idx = 0;
        for(TI i = 0; i < 1000; i++){
            T val = rlt::get(device, output_flat, i);
            if(val > max_val){
                max_val = val;
                max_idx = i;
            }
        }
        std::cout << "  class " << max_idx << ": " << max_val << std::endl;
        rlt::set(device, output_flat, (T)-1e30, max_idx);
    }

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, expected_output);
}

// ======================== Test: Intermediate activations (stem) ========================
TEST(RESNET18, STEM_CONV) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // Create just the stem conv layer
    using STEM_CAPABILITY = rlt::nn::capability::Forward<>;
    using STEM_LAYER = rlt::nn::layers::conv2d::Layer<STEM_CONV_CONFIG, STEM_CAPABILITY, INPUT_SHAPE>;
    STEM_LAYER stem;
    typename STEM_LAYER::template Buffer<true> stem_buffer;
    rlt::malloc(device, stem);
    rlt::malloc(device, stem_buffer);

    // Load stem weights from the sequential model format
    auto model_group = rlt::get_group(device, file, "model");
    auto layers_group = rlt::get_group(device, model_group, "layers");
    auto stem_group = rlt::get_group(device, layers_group, "0");
    ASSERT_TRUE(rlt::load(device, stem, stem_group));

    // Load input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::malloc(device, input);
    auto test_group = rlt::get_group(device, file, "test_data");
    rlt::load(device, input, test_group, "input");

    // Evaluate stem
    using STEM_OUTPUT_SHAPE = typename STEM_LAYER::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEM_OUTPUT_SHAPE>> stem_output, expected_stem;
    rlt::malloc(device, stem_output);
    rlt::malloc(device, expected_stem);
    rlt::load(device, expected_stem, test_group, "after_stem");

    rlt::evaluate(device, stem, input, stem_output, stem_buffer, rng);

    T diff = rlt::abs_diff(device, stem_output, expected_stem) / decltype(stem_output)::SPEC::SIZE;
    std::cout << "Stem conv diff (per element): " << diff << std::endl;
    ASSERT_LT(diff, EPSILON) << "Stem conv output mismatch";

    rlt::free(device, stem);
    rlt::free(device, stem_buffer);
    rlt::free(device, input);
    rlt::free(device, stem_output);
    rlt::free(device, expected_stem);
}

// ======================== Test: MaxPool ========================
TEST(RESNET18, MAXPOOL) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // MaxPool layer
    using STEM_OUTPUT_SHAPE = rlt::tensor::Shape<TI, 1, 112, 112, 64>;
    using MP_CAPABILITY = rlt::nn::capability::Forward<>;
    using MP_LAYER = rlt::nn::layers::max_pool2d::Layer<MAXPOOL_CONFIG, MP_CAPABILITY, STEM_OUTPUT_SHAPE>;
    MP_LAYER mp;
    typename MP_LAYER::template Buffer<true> mp_buffer;
    rlt::malloc(device, mp);
    rlt::malloc(device, mp_buffer);

    // Load input (after_stem) and expected (after_maxpool)
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEM_OUTPUT_SHAPE>> mp_input;
    using MP_OUTPUT_SHAPE = typename MP_LAYER::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MP_OUTPUT_SHAPE>> mp_output, expected_mp;
    rlt::malloc(device, mp_input);
    rlt::malloc(device, mp_output);
    rlt::malloc(device, expected_mp);

    auto test_group = rlt::get_group(device, file, "test_data");
    rlt::load(device, mp_input, test_group, "after_stem");
    rlt::load(device, expected_mp, test_group, "after_maxpool");

    rlt::evaluate(device, mp, mp_input, mp_output, mp_buffer, rng);

    T diff = rlt::abs_diff(device, mp_output, expected_mp) / decltype(mp_output)::SPEC::SIZE;
    std::cout << "MaxPool diff (per element): " << diff << std::endl;
    ASSERT_LT(diff, EPSILON) << "MaxPool output mismatch";

    rlt::free(device, mp);
    rlt::free(device, mp_buffer);
    rlt::free(device, mp_input);
    rlt::free(device, mp_output);
    rlt::free(device, expected_mp);
}

// ======================== Test: Individual ResNet blocks ========================
template<typename BLOCK_CONFIG, TI HEIGHT, TI WIDTH, TI IN_CHANNELS>
void test_resnet_block(const std::string& layer_group_name, const std::string& expected_output_name) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    using BLOCK_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, HEIGHT, WIDTH, IN_CHANNELS>;
    using BLOCK_CAPABILITY = rlt::nn::capability::Forward<>;
    using BLOCK_TYPE = rlt::nn::layers::resnet_block::Layer<BLOCK_CONFIG, BLOCK_CAPABILITY, BLOCK_INPUT_SHAPE>;
    BLOCK_TYPE block;
    typename BLOCK_TYPE::template Buffer<true> block_buffer;
    rlt::malloc(device, block);
    rlt::malloc(device, block_buffer);

    // Load block weights
    auto model_group = rlt::get_group(device, file, "model");
    auto layers_group = rlt::get_group(device, model_group, "layers");
    auto block_group = rlt::get_group(device, layers_group, layer_group_name);
    ASSERT_TRUE(rlt::load(device, block, block_group)) << "Failed to load block " << layer_group_name;

    // Load input and expected output
    auto test_group = rlt::get_group(device, file, "test_data");

    // Determine input activation name
    std::string input_name;
    if(layer_group_name == "2") input_name = "after_maxpool";
    else if(layer_group_name == "3") input_name = "after_layer1_block0";
    else if(layer_group_name == "4") input_name = "after_layer1_block1";
    else if(layer_group_name == "5") input_name = "after_layer2_block0";
    else if(layer_group_name == "6") input_name = "after_layer2_block1";
    else if(layer_group_name == "7") input_name = "after_layer3_block0";
    else if(layer_group_name == "8") input_name = "after_layer3_block1";
    else if(layer_group_name == "9") input_name = "after_layer4_block0";

    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_INPUT_SHAPE>> block_input;
    rlt::malloc(device, block_input);
    rlt::load(device, block_input, test_group, input_name);

    using BLOCK_OUTPUT_SHAPE = typename BLOCK_TYPE::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, BLOCK_OUTPUT_SHAPE>> block_output, expected_block;
    rlt::malloc(device, block_output);
    rlt::malloc(device, expected_block);
    rlt::load(device, expected_block, test_group, expected_output_name);

    rlt::evaluate(device, block, block_input, block_output, block_buffer, rng);

    T diff = rlt::abs_diff(device, block_output, expected_block) / decltype(block_output)::SPEC::SIZE;
    std::cout << "Block " << layer_group_name << " diff (per element): " << diff << std::endl;
    ASSERT_LT(diff, EPSILON) << "Block " << layer_group_name << " output mismatch";

    rlt::free(device, block);
    rlt::free(device, block_buffer);
    rlt::free(device, block_input);
    rlt::free(device, block_output);
    rlt::free(device, expected_block);
}

TEST(RESNET18, BLOCK_LAYER1_0) { test_resnet_block<BLOCK_64_S1_CONFIG, 56, 56, 64>("2", "after_layer1_block0"); }
TEST(RESNET18, BLOCK_LAYER1_1) { test_resnet_block<BLOCK_64_S1_CONFIG, 56, 56, 64>("3", "after_layer1_block1"); }
TEST(RESNET18, BLOCK_LAYER2_0) { test_resnet_block<BLOCK_128_S2_CONFIG, 56, 56, 64>("4", "after_layer2_block0"); }
TEST(RESNET18, BLOCK_LAYER2_1) { test_resnet_block<BLOCK_128_S1_CONFIG, 28, 28, 128>("5", "after_layer2_block1"); }
TEST(RESNET18, BLOCK_LAYER3_0) { test_resnet_block<BLOCK_256_S2_CONFIG, 28, 28, 128>("6", "after_layer3_block0"); }
TEST(RESNET18, BLOCK_LAYER3_1) { test_resnet_block<BLOCK_256_S1_CONFIG, 14, 14, 256>("7", "after_layer3_block1"); }
TEST(RESNET18, BLOCK_LAYER4_0) { test_resnet_block<BLOCK_512_S2_CONFIG, 14, 14, 256>("8", "after_layer4_block0"); }
TEST(RESNET18, BLOCK_LAYER4_1) { test_resnet_block<BLOCK_512_S1_CONFIG, 7, 7, 512>("9", "after_layer4_block1"); }

// ======================== Test: Full backward pass ========================
// Helper to check conv2d parameter gradients
template<typename CONV_LAYER, typename GROUP>
void check_conv_gradients(DEVICE& device, CONV_LAYER& layer, GROUP& grad_group, const std::string& prefix, T epsilon) {
    using TI = DEVICE::index_t;
    // d_weights
    {
        using WEIGHTS_SHAPE = typename decltype(layer.weights.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, WEIGHTS_SHAPE>> expected;
        rlt::malloc(device, expected);
        rlt::load(device, expected, grad_group, prefix + "_d_weights");
        T diff = rlt::abs_diff(device, layer.weights.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    " << prefix << " d_weights diff: " << diff << std::endl;
        EXPECT_LT(diff, epsilon) << prefix << " d_weights mismatch";
        rlt::free(device, expected);
    }
    // d_gamma
    {
        using GAMMA_SHAPE = typename decltype(layer.norm.gamma.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, GAMMA_SHAPE>> expected;
        rlt::malloc(device, expected);
        rlt::load(device, expected, grad_group, prefix + "_d_gamma");
        T diff = rlt::abs_diff(device, layer.norm.gamma.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    " << prefix << " d_gamma diff: " << diff << std::endl;
        EXPECT_LT(diff, epsilon) << prefix << " d_gamma mismatch";
        rlt::free(device, expected);
    }
    // d_beta
    {
        using BETA_SHAPE = typename decltype(layer.norm.beta.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, BETA_SHAPE>> expected;
        rlt::malloc(device, expected);
        rlt::load(device, expected, grad_group, prefix + "_d_beta");
        T diff = rlt::abs_diff(device, layer.norm.beta.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    " << prefix << " d_beta diff: " << diff << std::endl;
        EXPECT_LT(diff, epsilon) << prefix << " d_beta mismatch";
        rlt::free(device, expected);
    }
}

// Helper to check resnet_block gradients
template<typename BLOCK_LAYER, typename GROUP>
void check_block_gradients(DEVICE& device, BLOCK_LAYER& block, GROUP& grad_group, const std::string& block_prefix, T epsilon) {
    check_conv_gradients(device, block.conv1, grad_group, block_prefix + "_conv1", epsilon);
    check_conv_gradients(device, block.conv2, grad_group, block_prefix + "_conv2", epsilon);
    if constexpr(std::remove_reference_t<BLOCK_LAYER>::SPEC::HAS_DOWNSAMPLE) {
        check_conv_gradients(device, block.downsample.conv, grad_group, block_prefix + "_downsample", epsilon);
    }
}

using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
using RESNET18_GRAD = rlt::nn_models::sequential::Build<GRAD_CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

TEST(RESNET18, FULL_BACKWARD) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    static constexpr T GRAD_EPSILON = 1e-7;  // Looser tolerance for accumulated gradients

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // Create and malloc model with Gradient capability
    RESNET18_GRAD model;
    typename RESNET18_GRAD::template Buffer<true> buffer;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    // Load model weights
    auto model_group = rlt::get_group(device, file, "model");
    bool load_success = rlt::load(device, model, model_group);
    ASSERT_TRUE(load_success) << "Failed to load model weights";

    // Load input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::malloc(device, input);
    auto test_group = rlt::get_group(device, file, "test_data");
    rlt::load(device, input, test_group, "input");

    // ===================== Forward pass (Evaluation mode for BN) =====================
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device, model, input, buffer, rng, eval_mode);

    // Verify forward output
    using OUTPUT_SHAPE = typename RESNET18_GRAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> expected_output;
    rlt::malloc(device, expected_output);
    rlt::load(device, expected_output, test_group, "output");
    auto model_output_view = rlt::output(device, model);
    auto model_output_tensor = rlt::to_tensor(device, model_output_view);
    T fwd_diff = rlt::abs_diff(device, model_output_tensor, expected_output) / decltype(expected_output)::SPEC::SIZE;
    std::cout << "Forward diff (per element): " << fwd_diff << std::endl;
    ASSERT_LT(fwd_diff, EPSILON) << "Forward pass mismatch before backward";
    rlt::free(device, expected_output);

    // ===================== Backward pass =====================
    // d_output = ones (loss = sum of outputs)
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;
    rlt::malloc(device, d_output);
    rlt::set_all(device, d_output, (T)1);

    // d_input for the full model
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input;
    rlt::malloc(device, d_input);

    rlt::zero_gradient(device, model);
    rlt::backward_full(device, model, input, d_output, d_input, buffer, eval_mode);

    // ===================== Verify d_input =====================
    auto grad_group = rlt::get_group(device, file, "gradient_data");
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_expected;
    rlt::malloc(device, d_input_expected);
    rlt::load(device, d_input_expected, grad_group, "d_input");
    T d_input_diff = rlt::abs_diff(device, d_input, d_input_expected) / decltype(d_input_expected)::SPEC::SIZE;
    std::cout << "d_input diff (per element): " << d_input_diff << std::endl;
    EXPECT_LT(d_input_diff, GRAD_EPSILON) << "d_input mismatch";
    rlt::free(device, d_input_expected);

    // ===================== Verify parameter gradients =====================
    std::cout << "\nChecking parameter gradients:" << std::endl;

    // Layer 0: Stem conv
    std::cout << "  Stem:" << std::endl;
    check_conv_gradients(device, model.content, grad_group, "stem", GRAD_EPSILON);

    // Layer 1: MaxPool - no parameters

    // Layer 2-3: Layer1 blocks (64ch, stride=1)
    std::cout << "  Layer1 Block0:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.content, grad_group, "layer1_block0", GRAD_EPSILON);
    std::cout << "  Layer1 Block1:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.content, grad_group, "layer1_block1", GRAD_EPSILON);

    // Layer 4-5: Layer2 blocks (128ch)
    std::cout << "  Layer2 Block0:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.content, grad_group, "layer2_block0", GRAD_EPSILON);
    std::cout << "  Layer2 Block1:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.next_module.content, grad_group, "layer2_block1", GRAD_EPSILON);

    // Layer 6-7: Layer3 blocks (256ch)
    std::cout << "  Layer3 Block0:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.next_module.next_module.content, grad_group, "layer3_block0", GRAD_EPSILON);
    std::cout << "  Layer3 Block1:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.next_module.next_module.next_module.content, grad_group, "layer3_block1", GRAD_EPSILON);

    // Layer 8-9: Layer4 blocks (512ch)
    std::cout << "  Layer4 Block0:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.content, grad_group, "layer4_block0", GRAD_EPSILON);
    std::cout << "  Layer4 Block1:" << std::endl;
    check_block_gradients(device, model.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.content, grad_group, "layer4_block1", GRAD_EPSILON);

    // Layer 10: AvgPool - no parameters

    // Layer 11: FC (Dense)
    std::cout << "  FC:" << std::endl;
    auto& fc_layer = model.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.next_module.content;
    {
        using FC_W_SHAPE = typename decltype(fc_layer.weights.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, FC_W_SHAPE>> expected;
        rlt::malloc(device, expected);
        rlt::load(device, expected, grad_group, "fc_d_weights");
        T diff = rlt::abs_diff(device, fc_layer.weights.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    fc d_weights diff: " << diff << std::endl;
        EXPECT_LT(diff, GRAD_EPSILON) << "fc d_weights mismatch";
        rlt::free(device, expected);
    }
    {
        using FC_B_SHAPE = typename decltype(fc_layer.biases.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, FC_B_SHAPE>> expected;
        rlt::malloc(device, expected);
        rlt::load(device, expected, grad_group, "fc_d_biases");
        T diff = rlt::abs_diff(device, fc_layer.biases.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    fc d_biases diff: " << diff << std::endl;
        EXPECT_LT(diff, GRAD_EPSILON) << "fc d_biases mismatch";
        rlt::free(device, expected);
    }

    // Cleanup
    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, d_output);
    rlt::free(device, d_input);
}
