// CUDA tests for nn_models::parallel — compares CPU vs CUDA for evaluate, forward, backward
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

// =========================================================================
// Device types
// =========================================================================
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
using RNG_CPU = typename DEVICE_CPU::SPEC::RANDOM::ENGINE<>;
using RNG_CUDA = typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<>;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE_CPU::index_t;

// =========================================================================
// Model configuration
// =========================================================================
static constexpr TI BATCH_SIZE = 32;
static constexpr TI INPUT_DIM_A = 17;
static constexpr TI INPUT_DIM_B = 12;
static constexpr TI EMBED_DIM = 16;
static constexpr TI HIDDEN_DIM = 16;
static constexpr TI OUTPUT_DIM = 4;

using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM_A>;
using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM_B>;

// Branch A: Dense(EMBED_DIM)
using DENSE_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, EMBED_DIM, rlt::nn::activation_functions::RELU>;
using DENSE_A = rlt::nn::layers::dense::BindConfiguration<DENSE_A_CONFIG>;
using BRANCH_A = rlt::nn_models::sequential::Module<DENSE_A>;

// Branch B: Standardize → Dense(EMBED_DIM)
using STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
using STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZE_CONFIG>;
using DENSE_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, EMBED_DIM, rlt::nn::activation_functions::RELU>;
using DENSE_B = rlt::nn::layers::dense::BindConfiguration<DENSE_B_CONFIG>;
using BRANCH_B = rlt::nn_models::sequential::Module<STANDARDIZE, DENSE_B>;

// Head: MLP(HIDDEN_DIM, OUTPUT_DIM)
using HEAD_MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using HEAD_MLP = rlt::nn_models::mlp::BindConfiguration<HEAD_MLP_CONFIG>;

// Parallel model (no head)
using CAPABILITY_FWD = rlt::nn::capability::Forward<>;
using CAPABILITY_GRAD = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;

using PARALLEL_NO_HEAD = rlt::nn_models::parallel::Build<CAPABILITY_GRAD, BRANCH_A, BRANCH_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;
using PARALLEL_WITH_HEAD = rlt::nn_models::parallel::Build<CAPABILITY_GRAD, BRANCH_A, BRANCH_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD_MLP>;

// Head with mlp_unconditional_stddev (like the training binary)
using HEAD_USTD_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using HEAD_USTD = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<HEAD_USTD_CONFIG>;
using PARALLEL_USTD_HEAD = rlt::nn_models::parallel::Build<CAPABILITY_GRAD, BRANCH_A, BRANCH_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD_USTD>;

static constexpr T FWD_EPSILON = 1e-5;
static constexpr T BWD_EPSILON = 5e-4;

// =========================================================================
// Helper: fill tensor with deterministic data (CPU only)
// =========================================================================
template <typename DEVICE, typename SPEC, typename RNG>
void fill_random(DEVICE& device, rlt::Tensor<SPEC>& tensor, RNG& rng){
    using SHAPE = typename SPEC::SHAPE;
    constexpr auto SIZE = rlt::product(SHAPE{});
    T* ptr = rlt::data(tensor);
    for(typename DEVICE::index_t i = 0; i < SIZE; i++){
        ptr[i] = rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
    }
}

// =========================================================================
// Test: evaluate (no head)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, EVALUATE_NO_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    // CPU model
    PARALLEL_NO_HEAD model_cpu;
    typename PARALLEL_NO_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    // CUDA model (copy from CPU)
    PARALLEL_NO_HEAD model_cuda;
    typename PARALLEL_NO_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Input tensors
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cuda;
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    // Output tensors
    using OUTPUT_SHAPE = typename PARALLEL_NO_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cuda;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);

    // Evaluate
    rlt::evaluate(device_cpu, model_cpu, input_a_cpu, input_b_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_a_cuda, input_b_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    // Compare
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);
    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host) / rlt::product(OUTPUT_SHAPE{});
    std::cout << "Parallel EVALUATE (no head) CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    // Cleanup
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: evaluate (with head)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, EVALUATE_WITH_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 1);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 1);

    PARALLEL_WITH_HEAD model_cpu;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    PARALLEL_WITH_HEAD model_cuda;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    using OUTPUT_SHAPE = typename PARALLEL_WITH_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    rlt::evaluate(device_cpu, model_cpu, input_a_cpu, input_b_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_a_cuda, input_b_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);
    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host) / rlt::product(OUTPUT_SHAPE{});
    std::cout << "Parallel EVALUATE (with head) CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: forward (with head, gradient capability)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, FORWARD_WITH_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 2);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 2);

    PARALLEL_WITH_HEAD model_cpu;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    PARALLEL_WITH_HEAD model_cuda;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    rlt::forward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    using OUTPUT_SHAPE = typename PARALLEL_WITH_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, rlt::output(device_cuda, model_cuda), output_cuda_host);
    T diff = rlt::abs_diff(device_cpu, rlt::output(device_cpu, model_cpu), output_cuda_host) / rlt::product(OUTPUT_SHAPE{});
    std::cout << "Parallel FORWARD (with head) CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: backward (with head, gradient-only — no d_input)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, BACKWARD_WITH_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 3);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 3);

    PARALLEL_WITH_HEAD model_cpu;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    PARALLEL_WITH_HEAD model_cuda;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    // Forward on both devices
    rlt::forward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    // Create d_output (random gradient signal)
    using OUTPUT_SHAPE = typename PARALLEL_WITH_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output_cpu, d_output_cuda;
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cuda, d_output_cuda);
    fill_random(device_cpu, d_output_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // Zero gradients
    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::zero_gradient(device_cuda, model_cuda);
    cudaDeviceSynchronize();

    // Backward (gradient-only, no d_input)
    rlt::backward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, d_output_cpu, buffer_cpu);
    rlt::backward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, d_output_cuda, buffer_cuda);
    cudaDeviceSynchronize();

    // Compare model gradients by copying CUDA model back to CPU and computing abs_diff
    PARALLEL_WITH_HEAD model_cuda_host;
    rlt::malloc(device_cpu, model_cuda_host);
    rlt::copy(device_cuda, device_cpu, model_cuda, model_cuda_host);
    T diff = rlt::abs_diff(device_cpu, model_cpu, model_cuda_host);
    std::cout << "Parallel BACKWARD (with head) model abs_diff CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, BWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, model_cuda_host);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, d_output_cpu);
    rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: backward_full (with head, returns d_input)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, BACKWARD_FULL_WITH_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 4);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 4);

    PARALLEL_WITH_HEAD model_cpu;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    PARALLEL_WITH_HEAD model_cuda;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    rlt::forward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    using OUTPUT_SHAPE = typename PARALLEL_WITH_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output_cpu, d_output_cuda;
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cuda, d_output_cuda);
    fill_random(device_cpu, d_output_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // d_input tensors
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> d_input_a_cpu, d_input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> d_input_b_cpu, d_input_b_cuda;
    rlt::malloc(device_cpu, d_input_a_cpu);
    rlt::malloc(device_cpu, d_input_b_cpu);
    rlt::malloc(device_cuda, d_input_a_cuda);
    rlt::malloc(device_cuda, d_input_b_cuda);

    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::zero_gradient(device_cuda, model_cuda);
    cudaDeviceSynchronize();

    rlt::backward_full(device_cpu, model_cpu, input_a_cpu, input_b_cpu, d_output_cpu, d_input_a_cpu, d_input_b_cpu, buffer_cpu);
    rlt::backward_full(device_cuda, model_cuda, input_a_cuda, input_b_cuda, d_output_cuda, d_input_a_cuda, d_input_b_cuda, buffer_cuda);
    cudaDeviceSynchronize();

    // Compare d_input_a
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> d_input_a_cuda_host;
    rlt::malloc(device_cpu, d_input_a_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_input_a_cuda, d_input_a_cuda_host);
    T diff_a = rlt::abs_diff(device_cpu, d_input_a_cpu, d_input_a_cuda_host) / rlt::product(INPUT_SHAPE_A{});
    std::cout << "Parallel BACKWARD_FULL d_input_a CPU vs CUDA: " << diff_a << std::endl;
    EXPECT_LT(diff_a, BWD_EPSILON);

    // Compare d_input_b
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> d_input_b_cuda_host;
    rlt::malloc(device_cpu, d_input_b_cuda_host);
    rlt::copy(device_cuda, device_cpu, d_input_b_cuda, d_input_b_cuda_host);
    T diff_b = rlt::abs_diff(device_cpu, d_input_b_cpu, d_input_b_cuda_host) / rlt::product(INPUT_SHAPE_B{});
    std::cout << "Parallel BACKWARD_FULL d_input_b CPU vs CUDA: " << diff_b << std::endl;
    EXPECT_LT(diff_b, BWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, d_output_cpu);
    rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cpu, d_input_a_cpu);
    rlt::free(device_cpu, d_input_b_cpu);
    rlt::free(device_cuda, d_input_a_cuda);
    rlt::free(device_cuda, d_input_b_cuda);
    rlt::free(device_cpu, d_input_a_cuda_host);
    rlt::free(device_cpu, d_input_b_cuda_host);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: get_last_layer / mlp_unconditional_stddev head
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, GET_LAST_LAYER_USTD_HEAD){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 5);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 5);

    PARALLEL_USTD_HEAD model_cpu;
    typename PARALLEL_USTD_HEAD::Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    // Verify get_last_layer returns the head (which has log_std)
    auto& head = rlt::get_last_layer(model_cpu);
    rlt::set_all(device_cpu, head.log_std.parameters, (T)-0.5);
    T log_std_val = rlt::get(device_cpu, head.log_std.parameters, (TI)0);
    EXPECT_NEAR(log_std_val, -0.5, 1e-6);

    // Copy to CUDA, evaluate, compare
    PARALLEL_USTD_HEAD model_cuda;
    typename PARALLEL_USTD_HEAD::Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    using OUTPUT_SHAPE = typename PARALLEL_USTD_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    rlt::evaluate(device_cpu, model_cpu, input_a_cpu, input_b_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_a_cuda, input_b_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);
    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host) / rlt::product(OUTPUT_SHAPE{});
    std::cout << "Parallel EVALUATE (ustd head) CPU vs CUDA: " << diff << std::endl;
    EXPECT_LT(diff, FWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}

// =========================================================================
// Test: optimizer step (zero_gradient, backward, update cycle)
// =========================================================================
TEST(NN_MODELS_PARALLEL_CUDA, OPTIMIZER_STEP){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 6);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 6);

    using ADAM_PARAMS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMS>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

    PARALLEL_WITH_HEAD model_cpu, model_cuda;
    typename PARALLEL_WITH_HEAD::Buffer<> buffer_cpu, buffer_cuda;
    OPTIMIZER optimizer_cpu, optimizer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::malloc(device_cpu, optimizer_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::malloc(device_cuda, optimizer_cuda);

    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);
    rlt::init(device_cpu, optimizer_cpu);
    rlt::init(device_cuda, optimizer_cuda);
    rlt::reset_optimizer_state(device_cpu, optimizer_cpu, model_cpu);
    rlt::reset_optimizer_state(device_cuda, optimizer_cuda, model_cuda);
    rlt::copy(device_cpu, device_cuda, optimizer_cpu, optimizer_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A>> input_a_cpu, input_a_cuda;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B>> input_b_cpu, input_b_cuda;
    rlt::malloc(device_cpu, input_a_cpu);
    rlt::malloc(device_cpu, input_b_cpu);
    rlt::malloc(device_cuda, input_a_cuda);
    rlt::malloc(device_cuda, input_b_cuda);
    fill_random(device_cpu, input_a_cpu, rng_cpu);
    fill_random(device_cpu, input_b_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, input_a_cpu, input_a_cuda);
    rlt::copy(device_cpu, device_cuda, input_b_cpu, input_b_cuda);

    using OUTPUT_SHAPE = typename PARALLEL_WITH_HEAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output_cpu, d_output_cuda;
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cuda, d_output_cuda);
    fill_random(device_cpu, d_output_cpu, rng_cpu);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // Forward + backward + step on both devices
    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::forward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::backward(device_cpu, model_cpu, input_a_cpu, input_b_cpu, d_output_cpu, buffer_cpu);
    rlt::backward(device_cuda, model_cuda, input_a_cuda, input_b_cuda, d_output_cuda, buffer_cuda);
    cudaDeviceSynchronize();
    rlt::step(device_cpu, optimizer_cpu, model_cpu);
    rlt::step(device_cuda, optimizer_cuda, model_cuda);
    cudaDeviceSynchronize();

    // Compare models after one optimizer step
    PARALLEL_WITH_HEAD model_cuda_host;
    rlt::malloc(device_cpu, model_cuda_host);
    rlt::copy(device_cuda, device_cpu, model_cuda, model_cuda_host);
    T diff = rlt::abs_diff(device_cpu, model_cpu, model_cuda_host);
    std::cout << "Parallel OPTIMIZER_STEP model abs_diff after 1 step: " << diff << std::endl;
    EXPECT_LT(diff, BWD_EPSILON);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cpu, optimizer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, optimizer_cuda);
    rlt::free(device_cpu, model_cuda_host);
    rlt::free(device_cpu, input_a_cpu);
    rlt::free(device_cpu, input_b_cpu);
    rlt::free(device_cuda, input_a_cuda);
    rlt::free(device_cuda, input_b_cuda);
    rlt::free(device_cpu, d_output_cpu);
    rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cuda, rng_cuda);
}
