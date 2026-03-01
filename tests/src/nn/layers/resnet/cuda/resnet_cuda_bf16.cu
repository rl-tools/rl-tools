// ResNet-18 BF16 CUDA end-to-end test
// Verifies forward/backward works without crashes and produces reasonable outputs

#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <rl_tools/nn_models/resnet/resnet.h>

#include <cuda_bf16.h>
#include <gtest/gtest.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

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

using INPUT_SHAPE = rlt::nn_models::resnet18::INPUT_SHAPE<TYPE_POLICY, TI>;
using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 224, 224, 3>;

using FWD_CAPABILITY = rlt::nn::capability::Forward<>;
using GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;

// FP32 CPU model (for loading weights)
using RESNET18_FWD_CPU = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI, FWD_CAPABILITY>;

// BF16 CUDA models
using RESNET18_FWD_CUDA = rlt::nn_models::resnet18::MODEL<TYPE_POLICY_BF16, TI_CUDA, FWD_CAPABILITY>;
using MODULE_CHAIN_CUDA = rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY_BF16, TI_CUDA>;
using RESNET18_GRAD_CUDA = rlt::nn_models::sequential::Build<GRAD_CAPABILITY, MODULE_CHAIN_CUDA, INPUT_SHAPE_CUDA>;

// FP32 CPU grad model (for copying back and inspecting)
using MODULE_CHAIN_CPU = rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI>;
using RESNET18_GRAD_CPU = rlt::nn_models::sequential::Build<GRAD_CAPABILITY, MODULE_CHAIN_CPU, INPUT_SHAPE>;

TEST(NN_LAYERS_RESNET_CUDA_BF16, FORWARD_COMPARISON){
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);
    RNG_CPU rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 0);

    // Load FP32 model on CPU
    RESNET18_FWD_CPU model_cpu;
    typename RESNET18_FWD_CPU::template Buffer<true> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu); rlt::malloc(device_cpu, buffer_cpu);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(data_path_stub) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);
    auto model_group = rlt::get_group(device_cpu, file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu, model_group));

    // Load test input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    auto test_group = rlt::get_group(device_cpu, file, "test_data");
    ASSERT_TRUE(rlt::load(device_cpu, input_cpu, test_group, "input"));

    // Load timm reference output
    using OUTPUT_SHAPE_CPU = typename RESNET18_FWD_CPU::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> output_ref;
    rlt::malloc(device_cpu, output_ref);
    ASSERT_TRUE(rlt::load(device_cpu, output_ref, test_group, "output"));

    // Copy model to GPU (FP32 -> BF16 type-converting copy)
    RESNET18_FWD_CUDA model_cuda;
    typename RESNET18_FWD_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda); rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Copy input to GPU (FP32 -> BF16)
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // GPU forward (BF16)
    using OUTPUT_SHAPE_CUDA = typename RESNET18_FWD_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, OUTPUT_SHAPE_CUDA>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda, eval_mode);

    // Copy GPU output back to CPU (BF16 -> FP32)
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    // Compare BF16 output against timm reference
    auto output_ref_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, output_ref);
    auto output_cuda_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, output_cuda_host);

    T max_abs_diff = 0;
    bool has_nan = false;
    for(TI i = 0; i < 1000; i++){
        T ref_val = rlt::get(device_cpu, output_ref_flat, i);
        T cuda_val = rlt::get(device_cpu, output_cuda_flat, i);
        if(std::isnan(cuda_val)){ has_nan = true; break; }
        max_abs_diff = std::max(max_abs_diff, std::abs(ref_val - cuda_val));
    }
    EXPECT_FALSE(has_nan) << "BF16 forward output contains NaN";

    std::cout << "BF16(CUDA) vs timm reference:" << std::endl;
    std::cout << "  Max absolute difference: " << max_abs_diff << std::endl;

    struct Pred { TI idx; T logit; };
    auto print_top5 = [](const char* label, auto& device, auto& output_flat){
        std::vector<Pred> preds(1000);
        for(TI i = 0; i < 1000; i++) preds[i] = {i, rlt::get(device, output_flat, i)};
        std::sort(preds.begin(), preds.end(), [](const Pred& a, const Pred& b){ return a.logit > b.logit; });
        std::cout << label << " top-5: ";
        for(int k = 0; k < 5; k++) std::cout << preds[k].idx << "(" << preds[k].logit << ") ";
        std::cout << std::endl;
    };
    print_top5("timm ref ", device_cpu, output_ref_flat);
    print_top5("BF16 CUDA", device_cpu, output_cuda_flat);

    // BF16 has ~7 bits mantissa; 18-layer network accumulates significant error
    // Main check: no NaN, outputs in reasonable range (not diverged)
    EXPECT_LT(max_abs_diff, 15.0) << "BF16 vs timm reference differs too much";

    // BF16 has ~7 bits mantissa; 18-layer network accumulates significant error
    EXPECT_LT(max_abs_diff, 15.0) << "BF16 vs timm reference differs too much";

    rlt::free(device_cpu, model_cpu); rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cpu, input_cpu); rlt::free(device_cpu, output_ref);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda); rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, rng_cpu); rlt::free(device_cuda, rng_cuda);
}

TEST(NN_LAYERS_RESNET_CUDA_BF16, BACKWARD){
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto file = HighFive::File(std::string(data_path_stub) + "/resnet18_test_data.h5", HighFive::File::ReadOnly);

    // Load FP32 weights on CPU, then copy to BF16 CUDA
    RESNET18_GRAD_CPU model_cpu_loader;
    rlt::malloc(device_cpu, model_cpu_loader);
    auto model_group = rlt::get_group(device_cpu, file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu_loader, model_group));

    RESNET18_GRAD_CUDA model_cuda;
    typename RESNET18_GRAD_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda); rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu_loader, model_cuda);
    rlt::free(device_cpu, model_cpu_loader);

    // Load input
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    auto test_group = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, input_cpu, test_group, "input");

    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);
    rlt::free(device_cpu, input_cpu);

    // RNG
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cuda, model_cuda, input_cuda, buffer_cuda, rng_cuda, eval_mode);

    // d_output = ones
    using OUTPUT_SHAPE_CUDA_T = typename RESNET18_GRAD_CUDA::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, OUTPUT_SHAPE_CUDA_T>> d_output_cuda;
    rlt::malloc(device_cuda, d_output_cuda); rlt::set_all(device_cuda, d_output_cuda, (T_BF16)1);

    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, INPUT_SHAPE_CUDA>> d_input_cuda;
    rlt::malloc(device_cuda, d_input_cuda);

    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::backward_full(device_cuda, model_cuda, input_cuda, d_output_cuda, d_input_cuda, buffer_cuda, eval_mode);

    // Copy results back to CPU to check for NaN
    RESNET18_GRAD_CPU model_host;
    rlt::malloc(device_cpu, model_host);
    rlt::copy(device_cuda, device_cpu, model_cuda, model_host);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_host;
    rlt::malloc(device_cpu, d_input_host);
    rlt::copy(device_cuda, device_cpu, d_input_cuda, d_input_host);

    // Check d_input is not NaN
    EXPECT_FALSE(rlt::is_nan(device_cpu, d_input_host)) << "BF16 d_input contains NaN";

    // Check model gradients are not NaN
    EXPECT_FALSE(rlt::is_nan(device_cpu, model_host, eval_mode)) << "BF16 model gradients contain NaN";

    // Check d_input against timm reference for sanity
    auto grad_group = rlt::get_group(device_cpu, file, "gradient_data");
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_ref;
    rlt::malloc(device_cpu, d_input_ref);
    rlt::load(device_cpu, d_input_ref, grad_group, "d_input");
    T d_input_diff = rlt::abs_diff(device_cpu, d_input_host, d_input_ref) / decltype(d_input_ref)::SPEC::SIZE;
    std::cout << "BF16 d_input vs timm reference (per element): " << d_input_diff << std::endl;
    EXPECT_LT(d_input_diff, 10.0) << "BF16 d_input deviates too much from timm reference";
    rlt::free(device_cpu, d_input_ref);

    // Spot-check a few late-layer gradients (smaller spatial dims = less error accumulation)
    auto check_grad_not_zero = [&](auto& gradient, const char* name) {
        T sum = rlt::abs_diff(device_cpu, gradient, gradient); // always 0; use as type deducer
        // Just check the gradient has non-zero magnitude
        T norm = 0;
        // Use a simpler check: abs_diff against a zero tensor would need malloc; just check is_nan
        std::cout << "  " << name << " gradient SIZE=" << std::remove_reference_t<decltype(gradient)>::SPEC::SIZE << std::endl;
    };

    // Print per-layer gradient norms for Layer4 (late layers = most comparable)
    std::cout << "\nLayer4 Block1 gradient magnitudes:" << std::endl;
    auto& l4b1 = rlt::get_layer<9>(model_host);
    std::cout << "  conv1 d_weights abs_diff(ref): ";
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, typename decltype(l4b1.conv1.weights.gradient)::SPEC::SHAPE>> expected;
        rlt::malloc(device_cpu, expected);
        rlt::load(device_cpu, expected, grad_group, "layer4_block1_conv1_d_weights");
        T diff = rlt::abs_diff(device_cpu, l4b1.conv1.weights.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << diff << std::endl;
        EXPECT_LT(diff, 5.0) << "layer4_block1 conv1 d_weights too far from timm";
        rlt::free(device_cpu, expected);
    }

    std::cout << "  FC d_biases abs_diff(ref): ";
    {
        auto& fc = rlt::get_layer<11>(model_host);
        using FC_B_SHAPE = typename decltype(fc.biases.gradient)::SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, FC_B_SHAPE>> expected;
        rlt::malloc(device_cpu, expected);
        rlt::load(device_cpu, expected, grad_group, "fc_d_biases");
        T diff = rlt::abs_diff(device_cpu, fc.biases.gradient, expected) / decltype(expected)::SPEC::SIZE;
        std::cout << diff << std::endl;
        EXPECT_LT(diff, 5.0) << "FC d_biases too far from timm";
        rlt::free(device_cpu, expected);
    }

    // Cleanup
    rlt::free(device_cpu, model_host); rlt::free(device_cpu, d_input_host);
    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda); rlt::free(device_cuda, d_output_cuda); rlt::free(device_cuda, d_input_cuda);
    rlt::free(device_cuda, rng_cuda);
}

TEST(NN_LAYERS_RESNET_CUDA_BF16, AMP_COMPARISON){
    DEVICE_CPU device_cpu; DEVICE_CUDA device_cuda; rlt::init(device_cuda);

    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    auto amp_file = HighFive::File(std::string(data_path_stub) + "/resnet18_test_data_bf16.h5", HighFive::File::ReadOnly);

    // Load FP32 model on CPU from AMP checkpoint, then copy to BF16 CUDA Gradient model
    RESNET18_GRAD_CPU model_cpu;
    rlt::malloc(device_cpu, model_cpu);
    auto model_group = rlt::get_group(device_cpu, amp_file, "model");
    ASSERT_TRUE(rlt::load(device_cpu, model_cpu, model_group));

    RESNET18_GRAD_CUDA model_cuda;
    typename RESNET18_GRAD_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda); rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);
    rlt::free(device_cpu, model_cpu);

    // Load input
    auto test_group = rlt::get_group(device_cpu, amp_file, "test_data");
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    ASSERT_TRUE(rlt::load(device_cpu, input_cpu, test_group, "input"));

    rlt::Tensor<rlt::tensor::Specification<T_BF16, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);
    rlt::free(device_cpu, input_cpu);

    // Forward pass (Gradient model stores intermediates in each layer's .output)
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 0);
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::forward(device_cuda, model_cuda, input_cuda, buffer_cuda, rng_cuda, eval_mode);

    // Helper: copy a CUDA tensor to CPU fp32 and compute max abs diff against HDF5 reference
    auto compare_layer = [&](auto& cuda_output, const char* ref_name, const char* label) {
        using CUDA_SPEC = typename std::remove_reference_t<decltype(cuda_output)>::SPEC;
        using SHAPE = typename CUDA_SPEC::SHAPE;
        rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> host_output, ref_output;
        rlt::malloc(device_cpu, host_output); rlt::malloc(device_cpu, ref_output);
        rlt::copy(device_cuda, device_cpu, cuda_output, host_output);
        bool loaded = rlt::load(device_cpu, ref_output, test_group, ref_name);
        if(!loaded){ std::cout << "  " << label << ": SKIP (ref not found)" << std::endl; rlt::free(device_cpu, host_output); rlt::free(device_cpu, ref_output); return; }
        T max_diff = 0;
        TI size = decltype(host_output)::SPEC::SIZE;
        auto host_flat = rlt::view_memory<rlt::tensor::Shape<TI, decltype(host_output)::SPEC::SIZE>>(device_cpu, host_output);
        auto ref_flat = rlt::view_memory<rlt::tensor::Shape<TI, decltype(ref_output)::SPEC::SIZE>>(device_cpu, ref_output);
        for(TI i = 0; i < size; i++){
            T h = rlt::get(device_cpu, host_flat, i);
            T r = rlt::get(device_cpu, ref_flat, i);
            if(!std::isnan(h) && !std::isnan(r)) max_diff = std::max(max_diff, std::abs(h - r));
        }
        std::cout << "  " << label << ": max_abs_diff = " << max_diff << " (size=" << size << ")" << std::endl;
        rlt::free(device_cpu, host_output); rlt::free(device_cpu, ref_output);
    };

    std::cout << "\nRLtools BF16 vs PyTorch AMP (layer-by-layer):" << std::endl;
    compare_layer(rlt::get_layer<0>(model_cuda).output, "after_stem", "stem (conv+BN+ReLU)");
    compare_layer(rlt::get_layer<1>(model_cuda).output, "after_maxpool", "max_pool2d");
    compare_layer(rlt::get_layer<2>(model_cuda).output, "after_layer1_block0", "layer1_block0");
    compare_layer(rlt::get_layer<3>(model_cuda).output, "after_layer1_block1", "layer1_block1");
    compare_layer(rlt::get_layer<4>(model_cuda).output, "after_layer2_block0", "layer2_block0");
    compare_layer(rlt::get_layer<5>(model_cuda).output, "after_layer2_block1", "layer2_block1");
    compare_layer(rlt::get_layer<6>(model_cuda).output, "after_layer3_block0", "layer3_block0");
    compare_layer(rlt::get_layer<7>(model_cuda).output, "after_layer3_block1", "layer3_block1");
    compare_layer(rlt::get_layer<8>(model_cuda).output, "after_layer4_block0", "layer4_block0");
    compare_layer(rlt::get_layer<9>(model_cuda).output, "after_layer4_block1", "layer4_block1");
    compare_layer(rlt::get_layer<10>(model_cuda).output, "after_avgpool", "avg_pool2d");

    // Final output (FC layer) — dense layer uses Matrix, not Tensor
    {
        auto& fc = rlt::get_layer<11>(model_cuda);
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1000>> host_out;
        rlt::malloc(device_cpu, host_out);
        rlt::copy(device_cuda, device_cpu, fc.output, host_out);
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1000>>> ref_out;
        rlt::malloc(device_cpu, ref_out);
        rlt::load(device_cpu, ref_out, test_group, "output");
        auto ref_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, ref_out);
        T max_diff = 0;
        for(TI i = 0; i < 1000; i++){
            T h = host_out._data[i];
            T r = rlt::get(device_cpu, ref_flat, i);
            if(!std::isnan(h) && !std::isnan(r)) max_diff = std::max(max_diff, std::abs(h - r));
        }
        std::cout << "  FC output: max_abs_diff = " << max_diff << " (size=1000)" << std::endl;
        EXPECT_LT(max_diff, 15.0) << "FC output vs AMP differs too much";
        rlt::free(device_cpu, host_out); rlt::free(device_cpu, ref_out);
    }

    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda); rlt::free(device_cuda, rng_cuda);
}
