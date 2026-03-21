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
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <gtest/gtest.h>
#include <iostream>

namespace rlt = rl_tools;

using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
using RNG_CPU = typename DEVICE_CPU::SPEC::RANDOM::ENGINE<>;
using RNG_CUDA = typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE_CPU::index_t;

static constexpr TI BATCH_SIZE = 2048;
static constexpr TI INPUT_DIM = 50;
static constexpr TI OUTPUT_DIM = 1;
static constexpr TI HIDDEN_DIM = 64;
static constexpr TI NUM_LAYERS = 2;

using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>;

using STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
using STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZE_CONFIG>;

using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM,
    rlt::nn::activation_functions::ActivationFunction::FAST_TANH, rlt::nn::activation_functions::IDENTITY>;
using MLP_CONFIG_RELU = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM,
    rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;

using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZE, rlt::nn_models::sequential::Module<MLP>>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_TENSOR_MANUAL_CONVERT){
    // Tensors -> manually matrix_view -> call matrix-based evaluate
    static constexpr TI TEST_BATCH = 32;
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TEST_BATCH, INPUT_DIM>;
    using MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG_RELU, CAPABILITY, MLP_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MLP_MODEL model_cpu, model_cuda;
    typename MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Use TENSOR input but manually convert to matrix view
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_INPUT_SHAPE>> input_cpu, input_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    {
        auto m = rlt::matrix_view(device_cpu, input_cpu);
        for(TI i = 0; i < TEST_BATCH; i++)
            for(TI j = 0; j < INPUT_DIM; j++)
                rlt::set(m, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    }
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    using MLP_OUTPUT_SHAPE = typename MLP_MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    // CPU: manually convert tensor -> matrix -> evaluate
    {
        auto m_in = rlt::matrix_view(device_cpu, input_cpu);
        auto m_out = rlt::matrix_view(device_cpu, output_cpu);
        rlt::evaluate(device_cpu, model_cpu, m_in, m_out, buffer_cpu, rng_cpu);
    }
    // CUDA: manually convert tensor -> matrix -> evaluate
    {
        auto m_in = rlt::matrix_view(device_cuda, input_cuda);
        auto m_out = rlt::matrix_view(device_cuda, output_cuda);
        rlt::evaluate(device_cuda, model_cuda, m_in, m_out, buffer_cuda, rng_cuda);
        cudaDeviceSynchronize();
    }

    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);
    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (TEST_BATCH * OUTPUT_DIM);
    std::cout << "MLP tensor-manual-convert (batch=" << TEST_BATCH << ") CPU vs CUDA: " << diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu); rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu); rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu); rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_MATRIX_INPUT){
    // Call MLP evaluate directly with Matrix input (no tensor proxy)
    static constexpr TI TEST_BATCH = 32;
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TEST_BATCH, INPUT_DIM>;
    using MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG_RELU, CAPABILITY, MLP_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MLP_MODEL model_cpu, model_cuda;
    typename MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Use MATRIX input directly (bypass tensor proxy)
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, INPUT_DIM>> input_cpu, input_cuda;
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, OUTPUT_DIM>> output_cpu, output_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);

    for(TI i = 0; i < TEST_BATCH; i++)
        for(TI j = 0; j < INPUT_DIM; j++)
            rlt::set(input_cpu, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, OUTPUT_DIM>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (TEST_BATCH * OUTPUT_DIM);
    std::cout << "MLP matrix-input (batch=" << TEST_BATCH << ") CPU vs CUDA: " << diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu); rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu); rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu); rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_BACKWARD){
    static constexpr TI TEST_BATCH = 32;
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TEST_BATCH, INPUT_DIM>;
    using MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG_RELU, CAPABILITY, MLP_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MLP_MODEL model_cpu, model_cuda;
    typename MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Input and d_output
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, INPUT_DIM>> input_cpu, input_cuda;
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, OUTPUT_DIM>> d_output_cpu, d_output_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cuda, d_output_cuda);
    for(TI i = 0; i < TEST_BATCH; i++){
        for(TI j = 0; j < INPUT_DIM; j++)
            rlt::set(input_cpu, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
        for(TI j = 0; j < OUTPUT_DIM; j++)
            rlt::set(d_output_cpu, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-0.01, (T)0.01, rng_cpu));
    }
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);
    rlt::copy(device_cpu, device_cuda, d_output_cpu, d_output_cuda);

    // CPU forward + backward (matrix inputs)
    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::forward(device_cpu, model_cpu, input_cpu, buffer_cpu, rng_cpu);
    rlt::backward(device_cpu, model_cpu, input_cpu, d_output_cpu, buffer_cpu);

    // CUDA forward + backward (matrix inputs)
    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::forward(device_cuda, model_cuda, input_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::backward(device_cuda, model_cuda, input_cuda, d_output_cuda, buffer_cuda);
    cudaDeviceSynchronize();

    // Compare models (includes gradients)
    MLP_MODEL model_cuda_host;
    rlt::malloc(device_cpu, model_cuda_host);
    rlt::copy(device_cuda, device_cpu, model_cuda, model_cuda_host);
    T diff = rlt::abs_diff(device_cpu, model_cpu, model_cuda_host);
    std::cout << "MLP backward (matrix) abs_diff: " << diff << std::endl;
    EXPECT_LT(diff, 1e-3);

    // Now test with TENSOR inputs (through tensor proxy)
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda); // reset GPU model
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_INPUT_SHAPE>> t_input_cpu, t_input_cuda;
    using MLP_OUTPUT_SHAPE = typename MLP_MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_OUTPUT_SHAPE>> t_d_output_cpu, t_d_output_cuda;
    rlt::malloc(device_cpu, t_input_cpu);
    rlt::malloc(device_cuda, t_input_cuda);
    rlt::malloc(device_cpu, t_d_output_cpu);
    rlt::malloc(device_cuda, t_d_output_cuda);
    {
        auto t_in_mat = rlt::matrix_view(device_cpu, t_input_cpu);
        rlt::copy(device_cpu, device_cpu, input_cpu, t_in_mat);
        auto t_dout_mat = rlt::matrix_view(device_cpu, t_d_output_cpu);
        rlt::copy(device_cpu, device_cpu, d_output_cpu, t_dout_mat);
    }
    rlt::copy(device_cpu, device_cuda, t_input_cpu, t_input_cuda);
    rlt::copy(device_cpu, device_cuda, t_d_output_cpu, t_d_output_cuda);

    // CPU forward + backward (tensor inputs)
    rlt::zero_gradient(device_cpu, model_cpu);
    rlt::forward(device_cpu, model_cpu, t_input_cpu, buffer_cpu, rng_cpu);
    rlt::backward(device_cpu, model_cpu, t_input_cpu, t_d_output_cpu, buffer_cpu);

    // CUDA forward + backward (tensor inputs — goes through tensor proxy)
    rlt::zero_gradient(device_cuda, model_cuda);
    rlt::forward(device_cuda, model_cuda, t_input_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::backward(device_cuda, model_cuda, t_input_cuda, t_d_output_cuda, buffer_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, model_cuda, model_cuda_host);
    T diff_tensor = rlt::abs_diff(device_cpu, model_cpu, model_cuda_host);
    std::cout << "MLP backward (tensor proxy) abs_diff: " << diff_tensor << std::endl;
    EXPECT_LT(diff_tensor, 1e-3);

    rlt::free(device_cpu, model_cpu); rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda); rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, model_cuda_host);
    rlt::free(device_cpu, input_cpu); rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, d_output_cpu); rlt::free(device_cuda, d_output_cuda);
    rlt::free(device_cpu, t_input_cpu); rlt::free(device_cuda, t_input_cuda);
    rlt::free(device_cpu, t_d_output_cpu); rlt::free(device_cuda, t_d_output_cuda);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MANUAL_DENSE_CHAIN){
    // Manually chain two dense layers (like MLP does) to isolate the issue
    static constexpr TI TEST_BATCH = 32;
    using DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::FAST_TANH>;
    using DENSE1_INPUT_SHAPE = rlt::tensor::Shape<TI, TEST_BATCH, INPUT_DIM>;
    using DENSE1 = rlt::nn::layers::dense::Layer<DENSE1_CONFIG, rlt::nn::capability::Forward<>, DENSE1_INPUT_SHAPE>;
    using DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, rlt::nn::activation_functions::IDENTITY>;
    using DENSE2_INPUT_SHAPE = rlt::tensor::Shape<TI, TEST_BATCH, HIDDEN_DIM>;
    using DENSE2 = rlt::nn::layers::dense::Layer<DENSE2_CONFIG, rlt::nn::capability::Forward<>, DENSE2_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    DENSE1 d1_cpu, d1_cuda;
    DENSE2 d2_cpu, d2_cuda;
    typename DENSE1::template Buffer<> d1_buf_cpu, d1_buf_cuda;
    typename DENSE2::template Buffer<> d2_buf_cpu, d2_buf_cuda;
    rlt::malloc(device_cpu, d1_cpu); rlt::malloc(device_cpu, d2_cpu);
    rlt::malloc(device_cpu, d1_buf_cpu); rlt::malloc(device_cpu, d2_buf_cpu);
    rlt::init_weights(device_cpu, d1_cpu, rng_cpu);
    rlt::init_weights(device_cpu, d2_cpu, rng_cpu);
    rlt::malloc(device_cuda, d1_cuda); rlt::malloc(device_cuda, d2_cuda);
    rlt::malloc(device_cuda, d1_buf_cuda); rlt::malloc(device_cuda, d2_buf_cuda);
    rlt::copy(device_cpu, device_cuda, d1_cpu, d1_cuda);
    rlt::copy(device_cpu, device_cuda, d2_cpu, d2_cuda);

    // Input
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, INPUT_DIM>> input_cpu, input_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    for(TI i = 0; i < TEST_BATCH; i++)
        for(TI j = 0; j < INPUT_DIM; j++)
            rlt::set(input_cpu, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // Intermediate buffer (like tick)
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, HIDDEN_DIM>> inter_cpu, inter_cuda;
    rlt::malloc(device_cpu, inter_cpu);
    rlt::malloc(device_cuda, inter_cuda);

    // Output
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, OUTPUT_DIM>> output_cpu, output_cuda;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);

    // CPU: dense1 → inter → dense2 → output
    rlt::evaluate(device_cpu, d1_cpu, input_cpu, inter_cpu, d1_buf_cpu, rng_cpu);
    rlt::evaluate(device_cpu, d2_cpu, inter_cpu, output_cpu, d2_buf_cpu, rng_cpu);

    // CUDA: dense1 → inter → dense2 → output (NO intermediate sync!)
    rlt::evaluate(device_cuda, d1_cuda, input_cuda, inter_cuda, d1_buf_cuda, rng_cuda);
    rlt::evaluate(device_cuda, d2_cuda, inter_cuda, output_cuda, d2_buf_cuda, rng_cuda);
    cudaDeviceSynchronize();

    // Compare intermediate
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, HIDDEN_DIM>> inter_cuda_host;
    rlt::malloc(device_cpu, inter_cuda_host);
    rlt::copy(device_cuda, device_cpu, inter_cuda, inter_cuda_host);
    T inter_diff = rlt::abs_diff(device_cpu, inter_cpu, inter_cuda_host);
    std::cout << "Manual chain inter abs_diff: " << inter_diff << " (per-element: " << inter_diff / (TEST_BATCH * HIDDEN_DIM) << ")" << std::endl;

    // Compare output
    rlt::Matrix<rlt::matrix::Specification<T, TI, TEST_BATCH, OUTPUT_DIM>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);
    T out_diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = out_diff / (TEST_BATCH * OUTPUT_DIM);
    std::cout << "Manual chain output abs_diff: " << out_diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, d1_cpu); rlt::free(device_cpu, d2_cpu);
    rlt::free(device_cpu, d1_buf_cpu); rlt::free(device_cpu, d2_buf_cpu);
    rlt::free(device_cuda, d1_cuda); rlt::free(device_cuda, d2_cuda);
    rlt::free(device_cuda, d1_buf_cuda); rlt::free(device_cuda, d2_buf_cuda);
    rlt::free(device_cpu, input_cpu); rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, inter_cpu); rlt::free(device_cuda, inter_cuda);
    rlt::free(device_cpu, inter_cuda_host);
    rlt::free(device_cpu, output_cpu); rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_RELU_BATCH1){
    static constexpr TI TEST_BATCH = 1;
    using TEST_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TEST_BATCH, INPUT_DIM>;
    using TEST_MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG_RELU, CAPABILITY, TEST_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    TEST_MLP_MODEL model_cpu, model_cuda;
    typename TEST_MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, TEST_INPUT_SHAPE>> input_cpu, input_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    {
        auto m = rlt::matrix_view(device_cpu, input_cpu);
        for(TI i = 0; i < TEST_BATCH; i++)
            for(TI j = 0; j < INPUT_DIM; j++)
                rlt::set(m, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    }
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    using TEST_OUTPUT_SHAPE = typename TEST_MLP_MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, TEST_OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (TEST_BATCH * OUTPUT_DIM);
    std::cout << "MLP RELU (batch=" << TEST_BATCH << ") CPU vs CUDA: " << diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_ONLY_SMALL){
    static constexpr TI SMALL_BATCH = 32;
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, SMALL_BATCH, INPUT_DIM>;
    using MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG_RELU, CAPABILITY, MLP_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MLP_MODEL model_cpu, model_cuda;
    typename MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_INPUT_SHAPE>> input_cpu, input_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    {
        auto m = rlt::matrix_view(device_cpu, input_cpu);
        for(TI i = 0; i < SMALL_BATCH; i++)
            for(TI j = 0; j < INPUT_DIM; j++)
                rlt::set(m, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    }
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    using MLP_OUTPUT_SHAPE = typename MLP_MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (SMALL_BATCH * OUTPUT_DIM);
    std::cout << "MLP-only (batch=" << SMALL_BATCH << ") EVALUATE CPU vs CUDA: " << diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, MLP_ONLY){
    // Test just the MLP (no standardize wrapper) on CPU vs CUDA
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>;
    using MLP_MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<MLP_CONFIG, CAPABILITY, MLP_INPUT_SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MLP_MODEL model_cpu, model_cuda;
    typename MLP_MODEL::template Buffer<> buffer_cpu, buffer_cuda;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_INPUT_SHAPE>> input_cpu, input_cuda;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, input_cuda);
    {
        auto m = rlt::matrix_view(device_cpu, input_cpu);
        for(TI i = 0; i < BATCH_SIZE; i++)
            for(TI j = 0; j < INPUT_DIM; j++)
                rlt::set(m, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
    }
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    using MLP_OUTPUT_SHAPE = typename MLP_MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_OUTPUT_SHAPE>> output_cpu, output_cuda, output_cuda_host;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cuda_host);

    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (BATCH_SIZE * OUTPUT_DIM);
    std::cout << "MLP-only EVALUATE CPU vs CUDA: " << diff << " (per-element: " << per_element << ")" << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}

TEST(NN_MODELS_SEQUENTIAL_STANDARDIZE_MLP_CUDA, EVALUATE){
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);

    RNG_CPU rng_cpu;
    RNG_CUDA rng_cuda;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    MODEL model_cpu;
    typename MODEL::template Buffer<> buffer_cpu;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::init_weights(device_cpu, model_cpu, rng_cpu);

    // Set standardize to non-trivial values
    {
        auto& std_layer = rlt::get_layer<0>(model_cpu);
        for(TI i = 0; i < INPUT_DIM; i++){
            rlt::set(device_cpu, std_layer.mean.parameters, (T)(i * 0.1), i);
            rlt::set(device_cpu, std_layer.precision.parameters, (T)(1.0 / (1.0 + i * 0.05)), i);
        }
    }

    MODEL model_cuda;
    typename MODEL::template Buffer<> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);

    // Round-trip model check
    {
        MODEL model_roundtrip;
        rlt::malloc(device_cpu, model_roundtrip);
        rlt::copy(device_cuda, device_cpu, model_cuda, model_roundtrip);
        T model_diff = rlt::abs_diff(device_cpu, model_cpu, model_roundtrip);
        std::cout << "Model round-trip abs_diff: " << model_diff << std::endl;
        EXPECT_EQ(model_diff, 0);
        rlt::free(device_cpu, model_roundtrip);
    }

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    {
        auto input_matrix = rlt::matrix_view(device_cpu, input_cpu);
        for(TI i = 0; i < BATCH_SIZE; i++){
            for(TI j = 0; j < INPUT_DIM; j++){
                rlt::set(input_matrix, i, j, rlt::random::uniform_real_distribution(device_cpu.random, (T)-1, (T)1, rng_cpu));
            }
        }
    }

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cpu, output_cuda;
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cuda, output_cuda);

    rlt::evaluate(device_cpu, model_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda);
    cudaDeviceSynchronize();

    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_cuda_host;
    rlt::malloc(device_cpu, output_cuda_host);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_host);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_host);
    T per_element = diff / (BATCH_SIZE * OUTPUT_DIM);
    auto output_cpu_matrix = rlt::matrix_view(device_cpu, output_cpu);
    auto output_cuda_matrix = rlt::matrix_view(device_cpu, output_cuda_host);
    std::cout << "Sequential(Standardize, MLP) EVALUATE CPU vs CUDA: " << diff
              << " (per-element: " << per_element << ")"
              << " CPU[0]=" << rlt::get(output_cpu_matrix, 0, 0)
              << " CUDA[0]=" << rlt::get(output_cuda_matrix, 0, 0)
              << std::endl;
    EXPECT_LT(per_element, 1e-5);

    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cpu, output_cuda_host);
}
