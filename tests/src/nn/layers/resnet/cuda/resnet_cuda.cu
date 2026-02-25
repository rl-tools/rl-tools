// ResNet-18 CUDA test: forward comparison and backward pass against timm reference

// CPU operations
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>

#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>

#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>

// NN operations (CPU)
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

// NN operations (CUDA)
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>

// Persist (CPU only - load then copy to GPU)
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

// Model definition
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

using INPUT_SHAPE = rlt::nn_models::resnet18::INPUT_SHAPE<TYPE_POLICY, TI>;
using CAPABILITY = rlt::nn::capability::Forward<>;

using RESNET18_CPU = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI, CAPABILITY>;
using RESNET18_CUDA = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI_CUDA, CAPABILITY>;

using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
using MODULE_CHAIN = rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI>;
using MODULE_CHAIN_CUDA = rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI_CUDA>;
using RESNET18_GRAD_CPU = rlt::nn_models::sequential::Build<GRAD_CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 224, 224, 3>;
using RESNET18_GRAD_CUDA = rlt::nn_models::sequential::Build<GRAD_CAPABILITY, MODULE_CHAIN_CUDA, INPUT_SHAPE_CUDA>;

TEST(NN_LAYERS_RESNET_CUDA, FORWARD_COMPARISON){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;

    // Load model on CPU
    RESNET18_CPU model_cpu;
    typename RESNET18_CPU::template Buffer<true> buffer_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(data_path_stub) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    auto model_group = rlt::get_group(device_cpu, file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu, model_group));

    // Create and fill input with test data
    using INPUT_SHAPE_CPU = rlt::tensor::Shape<TI, 1, 224, 224, 3>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);

    // Load test input from HDF5
    auto test_data_group = rlt::get_group(device_cpu, file, "test_data");
    ASSERT_TRUE(rlt::load(device_cpu, input_cpu, test_data_group, "input"));

    // Run CPU inference
    using OUTPUT_SHAPE = typename RESNET18_CPU::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cpu;
    rlt::malloc(device_cpu, output_cpu);
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu, eval_mode);

    // Copy model to GPU
    RESNET18_CUDA model_cuda;
    typename RESNET18_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Copy input to GPU
    using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 224, 224, 3>;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // Run GPU inference
    using OUTPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 1000>;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OUTPUT_SHAPE_CUDA>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda, eval_mode);

    // Copy GPU output back to CPU for comparison
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    // Compare outputs
    auto output_cpu_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, output_cpu);
    auto output_cuda_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, output_cuda_host);

    T max_abs_diff = 0;
    T max_rel_diff = 0;
    for(TI i = 0; i < 1000; i++){
        T cpu_val = rlt::get(device_cpu, output_cpu_flat, i);
        T cuda_val = rlt::get(device_cpu, output_cuda_flat, i);
        T abs_diff = std::abs(cpu_val - cuda_val);
        T rel_diff = std::abs(cpu_val) > 1e-6 ? abs_diff / std::abs(cpu_val) : abs_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }

    std::cout << "CPU vs CUDA comparison:" << std::endl;
    std::cout << "  Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "  Max relative difference: " << max_rel_diff << std::endl;

    // Print top-5 for both
    struct Pred { TI idx; T logit; };
    auto print_top5 = [](const char* label, auto& device, auto& output_flat){
        std::vector<Pred> preds(1000);
        for(TI i = 0; i < 1000; i++){
            preds[i] = {i, rlt::get(device, output_flat, i)};
        }
        std::sort(preds.begin(), preds.end(), [](const Pred& a, const Pred& b){ return a.logit > b.logit; });
        std::cout << label << " top-5: ";
        for(int k = 0; k < 5; k++){
            std::cout << preds[k].idx << "(" << preds[k].logit << ") ";
        }
        std::cout << std::endl;
    };
    print_top5("CPU ", device_cpu, output_cpu_flat);
    print_top5("CUDA", device_cpu, output_cuda_flat);

    // For float precision with cuDNN, allow some tolerance (cuDNN may use different algorithms)
    EXPECT_LT(max_abs_diff, 0.1) << "CPU vs CUDA results differ too much";

    // Cleanup
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cuda, rng_cuda);
}

// ======================== Backward pass helpers ========================

template<typename CONV_LAYER, typename GROUP>
void check_conv_gradients_cuda(DEVICE_CPU& device, CONV_LAYER& layer, GROUP& grad_group, const std::string& prefix, T epsilon) {
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

template<typename BLOCK_LAYER, typename GROUP>
void check_block_gradients_cuda(DEVICE_CPU& device, BLOCK_LAYER& block, GROUP& grad_group, const std::string& block_prefix, T epsilon) {
    check_conv_gradients_cuda(device, block.conv1, grad_group, block_prefix + "_conv1", epsilon);
    check_conv_gradients_cuda(device, block.conv2, grad_group, block_prefix + "_conv2", epsilon);
    if constexpr(std::remove_reference_t<BLOCK_LAYER>::SPEC::HAS_DOWNSAMPLE) {
        check_conv_gradients_cuda(device, block.downsample.conv, grad_group, block_prefix + "_downsample", epsilon);
    }
}

// ======================== Test: CUDA full backward pass against timm ========================

TEST(NN_LAYERS_RESNET_CUDA, FULL_BACKWARD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);

    static constexpr T GRAD_EPSILON = 5.0;

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    std::cout << "Loading test data from: " << data_file_path << std::endl;
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // Create CPU model with Gradient capability (for loading weights and comparing results)
    RESNET18_GRAD_CPU model_cpu;
    typename RESNET18_GRAD_CPU::template Buffer<true> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);

    // Load model weights on CPU
    auto model_group = rlt::get_group(device_cpu, file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu, model_group)) << "Failed to load model weights";

    // Create CUDA model and copy weights
    RESNET18_GRAD_CUDA model_cuda;
    typename RESNET18_GRAD_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Load input on CPU
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    auto test_group = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, input_cpu, test_group, "input");

    // Copy input to GPU
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // Forward pass on GPU (stores intermediate activations needed for backward)
    RNG_CPU rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cuda, model_cuda, input_cuda, buffer_cuda, rng_cuda, eval_mode);

    // Verify forward output matches reference
    using OUTPUT_SHAPE = typename RESNET18_GRAD_CUDA::OUTPUT_SHAPE;
    using OUTPUT_SHAPE_CPU = typename RESNET18_GRAD_CPU::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> expected_output;
    rlt::malloc(device_cpu, expected_output);
    rlt::load(device_cpu, expected_output, test_group, "output");

    auto model_output_cuda = rlt::output(device_cuda, model_cuda);
    auto model_output_cuda_tensor = rlt::to_tensor(device_cuda, model_output_cuda);
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> fwd_output_host;
    rlt::malloc(device_cpu, fwd_output_host);
    rlt::copy(device_cuda, device_cpu, model_output_cuda_tensor, fwd_output_host);

    T fwd_diff = rlt::abs_diff(device_cpu, fwd_output_host, expected_output) / decltype(expected_output)::SPEC::SIZE;
    std::cout << "CUDA forward diff (per element): " << fwd_diff << std::endl;
    EXPECT_LT(fwd_diff, 5e-3) << "CUDA forward pass mismatch before backward";
    rlt::free(device_cpu, expected_output);
    rlt::free(device_cpu, fwd_output_host);

    // Backward pass on GPU: d_output = ones (loss = sum)
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OUTPUT_SHAPE>> d_output_cuda;
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::set_all(device_cuda, d_output_cuda, (T)1);

    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> d_input_cuda;
    rlt::malloc(device_cuda, d_input_cuda);

    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::backward_full(device_cuda, model_cuda, input_cuda, d_output_cuda, d_input_cuda, buffer_cuda, eval_mode);

    // Copy model back to CPU (transfers gradients)
    rlt::copy(device_cuda, device_cpu, model_cuda, model_cpu);

    // Copy d_input back to CPU
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_cpu;
    rlt::malloc(device_cpu, d_input_cpu);
    rlt::copy(device_cuda, device_cpu, d_input_cuda, d_input_cpu);

    // ===================== Verify d_input =====================
    auto grad_group = rlt::get_group(device_cpu, file, "gradient_data");
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_expected;
    rlt::malloc(device_cpu, d_input_expected);
    rlt::load(device_cpu, d_input_expected, grad_group, "d_input");
    T d_input_diff = rlt::abs_diff(device_cpu, d_input_cpu, d_input_expected) / decltype(d_input_expected)::SPEC::SIZE;
    std::cout << "CUDA d_input diff (per element): " << d_input_diff << std::endl;
    EXPECT_LT(d_input_diff, GRAD_EPSILON) << "CUDA d_input mismatch";
    rlt::free(device_cpu, d_input_expected);

    // ===================== Verify parameter gradients =====================
    std::cout << "\nChecking CUDA parameter gradients against timm reference:" << std::endl;

    std::cout << "  Stem:" << std::endl;
    check_conv_gradients_cuda(device_cpu, rlt::get_layer<0>(model_cpu), grad_group, "stem", GRAD_EPSILON);

    std::cout << "  Layer1 Block0:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<2>(model_cpu), grad_group, "layer1_block0", GRAD_EPSILON);
    std::cout << "  Layer1 Block1:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<3>(model_cpu), grad_group, "layer1_block1", GRAD_EPSILON);

    std::cout << "  Layer2 Block0:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<4>(model_cpu), grad_group, "layer2_block0", GRAD_EPSILON);
    std::cout << "  Layer2 Block1:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<5>(model_cpu), grad_group, "layer2_block1", GRAD_EPSILON);

    std::cout << "  Layer3 Block0:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<6>(model_cpu), grad_group, "layer3_block0", GRAD_EPSILON);
    std::cout << "  Layer3 Block1:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<7>(model_cpu), grad_group, "layer3_block1", GRAD_EPSILON);

    std::cout << "  Layer4 Block0:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<8>(model_cpu), grad_group, "layer4_block0", GRAD_EPSILON);
    std::cout << "  Layer4 Block1:" << std::endl;
    check_block_gradients_cuda(device_cpu, rlt::get_layer<9>(model_cpu), grad_group, "layer4_block1", GRAD_EPSILON);

    std::cout << "  FC:" << std::endl;
    auto& fc_layer = rlt::get_layer<11>(model_cpu);
    {
        using FC_W_SHAPE = typename decltype(fc_layer.weights.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, FC_W_SHAPE>> expected;
        rlt::malloc(device_cpu, expected);
        rlt::load(device_cpu, expected, grad_group, "fc_d_weights");
        T diff = rlt::abs_diff(device_cpu, fc_layer.weights.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    fc d_weights diff: " << diff << std::endl;
        EXPECT_LT(diff, GRAD_EPSILON) << "fc d_weights mismatch";
        rlt::free(device_cpu, expected);
    }
    {
        using FC_B_SHAPE = typename decltype(fc_layer.biases.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, FC_B_SHAPE>> expected;
        rlt::malloc(device_cpu, expected);
        rlt::load(device_cpu, expected, grad_group, "fc_d_biases");
        T diff = rlt::abs_diff(device_cpu, fc_layer.biases.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << "    fc d_biases diff: " << diff << std::endl;
        EXPECT_LT(diff, GRAD_EPSILON) << "fc d_biases mismatch";
        rlt::free(device_cpu, expected);
    }

    // Cleanup
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cpu, d_input_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cuda, d_input_cuda);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// ======================== Test: CPU vs CUDA backward comparison ========================
// Runs backward on both CPU and CUDA, compares gradients directly

TEST(NN_LAYERS_RESNET_CUDA, BACKWARD_CPU_VS_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/resnet18_test_data.h5";
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    // CPU model
    RESNET18_GRAD_CPU model_cpu;
    typename RESNET18_GRAD_CPU::template Buffer<true> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);

    auto model_group = rlt::get_group(device_cpu, file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu, model_group));

    // CUDA model (copy from CPU)
    RESNET18_GRAD_CUDA model_cuda;
    typename RESNET18_GRAD_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Load input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    auto test_group = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, input_cpu, test_group, "input");

    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // RNGs
    RNG_CPU rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;

    // Forward on both
    rlt::forward(device_cpu, model_cpu, input_cpu, buffer_cpu, rng_cpu, eval_mode);
    rlt::forward(device_cuda, model_cuda, input_cuda, buffer_cuda, rng_cuda, eval_mode);

    // d_output = ones
    using OUTPUT_SHAPE_CPU = typename RESNET18_GRAD_CPU::OUTPUT_SHAPE;
    using OUTPUT_SHAPE_CUDA_T = typename RESNET18_GRAD_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> d_output_cpu;
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::set_all(device_cpu, d_output_cpu, (T)1);

    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OUTPUT_SHAPE_CUDA_T>> d_output_cuda;
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::set_all(device_cuda, d_output_cuda, (T)1);

    // d_input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_cpu;
    rlt::malloc(device_cpu, d_input_cpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> d_input_cuda;
    rlt::malloc(device_cuda, d_input_cuda);

    // Backward on both
    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::backward_full(device_cpu, model_cpu, input_cpu, d_output_cpu, d_input_cpu, buffer_cpu, eval_mode);

    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::backward_full(device_cuda, model_cuda, input_cuda, d_output_cuda, d_input_cuda, buffer_cuda, eval_mode);

    // Copy CUDA results back to CPU for comparison
    RESNET18_GRAD_CPU model_cuda_host;
    rlt::malloc(device_cpu, model_cuda_host);
    rlt::copy(device_cuda, device_cpu, model_cuda, model_cuda_host);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_cuda_host;
    rlt::malloc(device_cpu, d_input_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_input_cuda, d_input_cuda_host);

    // Compare d_input
    T d_input_diff = rlt::abs_diff(device_cpu, d_input_cpu, d_input_cuda_host) / decltype(d_input_cpu)::SPEC::SIZE;
    std::cout << "CPU vs CUDA d_input diff (per element): " << d_input_diff << std::endl;
    EXPECT_LT(d_input_diff, 0.1) << "CPU vs CUDA d_input mismatch";

    // Compare model gradients
    T model_grad_diff = rlt::abs_diff(device_cpu, model_cpu, model_cuda_host);
    std::cout << "CPU vs CUDA total model abs_diff: " << model_grad_diff << std::endl;
    EXPECT_LT(model_grad_diff / 11e6, 1.0) << "CPU vs CUDA model gradients mismatch";

    // Cleanup
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cpu, d_output_cpu);
    rlt::free(device_cpu, d_input_cpu);
    rlt::free(device_cpu, model_cuda_host);
    rlt::free(device_cpu, d_input_cuda_host);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cuda, d_input_cuda);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}
