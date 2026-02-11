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
