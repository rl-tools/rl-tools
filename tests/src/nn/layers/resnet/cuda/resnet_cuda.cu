// ResNet-18 CUDA inference test: compare CPU vs GPU (cuDNN) results

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
    auto test_group = rlt::get_group(device_cpu, file, "test_data");
    rlt::load(device_cpu, input_cpu, test_group, "input");

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
