#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/lamb/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/lamb/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 1, 4>;
using NETWORK_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 4, 2, 32, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using OPTIMIZER_SPEC = rlt::nn::optimizers::lamb::Specification<TYPE_POLICY, TI>;
using OPTIMIZER = rlt::nn::optimizers::Lamb<OPTIMIZER_SPEC>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using NetworkType = rlt::nn_models::mlp::NeuralNetwork<NETWORK_CONFIG, CAPABILITY, INPUT_SHAPE>;

TEST(RL_TOOLS_NN_OPTIMIZERS_LAMB, OVERFIT_SINGLE_SAMPLE){
    DEVICE device;
    OPTIMIZER optimizer;
    NetworkType network;
    typename NetworkType::Buffer<> buffers;
    rlt::malloc(device, optimizer);
    rlt::malloc(device, network);
    rlt::malloc(device, buffers);
    rlt::init(device, optimizer);

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    rlt::init_weights(device, network, rng);
    rlt::reset_optimizer_state(device, optimizer, network);

    constexpr TI INPUT_DIM = rlt::get_last(typename NetworkType::INPUT_SHAPE{});
    constexpr TI OUTPUT_DIM = rlt::get_last(typename NetworkType::OUTPUT_SHAPE{});

    T input_data[INPUT_DIM] = {0.5, -0.3, 0.8, -0.1};
    T target_data[OUTPUT_DIM] = {1.0, -1.0, 0.5, 0.2};

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, INPUT_DIM>> input_matrix;
    input_matrix._data = input_data;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> target_matrix;
    target_matrix._data = target_data;

    T initial_loss = 0;
    T final_loss = 0;
    constexpr TI N_ITERATIONS = 500;
    for(TI i = 0; i < N_ITERATIONS; i++){
        rlt::zero_gradient(device, network);
        bool rng_flag = false;
        rlt::forward(device, network, input_matrix, buffers, rng_flag);
        T loss = rlt::nn::loss_functions::mse::evaluate(device, network.output_layer.output, target_matrix);
        if(i == 0){
            initial_loss = loss;
        }
        if(i == N_ITERATIONS - 1){
            final_loss = loss;
        }
        T d_loss_d_output_data[OUTPUT_DIM];
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> d_loss_d_output;
        d_loss_d_output._data = d_loss_d_output_data;
        rlt::nn::loss_functions::mse::gradient(device, network.output_layer.output, target_matrix, d_loss_d_output);
        rlt::backward(device, network, input_matrix, d_loss_d_output, buffers);
        rlt::step(device, optimizer, network);
    }
    EXPECT_GT(initial_loss, 0.01);
    EXPECT_LT(final_loss, 1e-5);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_LAMB, LOSS_DECREASES_MONOTONICALLY_INITIAL_PHASE){
    DEVICE device;
    OPTIMIZER optimizer;
    NetworkType network;
    typename NetworkType::Buffer<> buffers;
    rlt::malloc(device, optimizer);
    rlt::malloc(device, network);
    rlt::malloc(device, buffers);
    rlt::init(device, optimizer);

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 123);
    rlt::init_weights(device, network, rng);
    rlt::reset_optimizer_state(device, optimizer, network);

    constexpr TI INPUT_DIM = rlt::get_last(typename NetworkType::INPUT_SHAPE{});
    constexpr TI OUTPUT_DIM = rlt::get_last(typename NetworkType::OUTPUT_SHAPE{});

    T input_data[INPUT_DIM] = {1.0, 0.0, -1.0, 0.5};
    T target_data[OUTPUT_DIM] = {0.0, 1.0, 0.0, 1.0};

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, INPUT_DIM>> input_matrix;
    input_matrix._data = input_data;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> target_matrix;
    target_matrix._data = target_data;

    T prev_loss = 1e10;
    TI non_decreasing_count = 0;
    constexpr TI N_ITERATIONS = 50;
    for(TI i = 0; i < N_ITERATIONS; i++){
        rlt::zero_gradient(device, network);
        bool rng_flag = false;
        rlt::forward(device, network, input_matrix, buffers, rng_flag);
        T loss = rlt::nn::loss_functions::mse::evaluate(device, network.output_layer.output, target_matrix);
        if(loss >= prev_loss){
            non_decreasing_count++;
        }
        prev_loss = loss;
        T d_loss_d_output_data[OUTPUT_DIM];
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> d_loss_d_output;
        d_loss_d_output._data = d_loss_d_output_data;
        rlt::nn::loss_functions::mse::gradient(device, network.output_layer.output, target_matrix, d_loss_d_output);
        rlt::backward(device, network, input_matrix, d_loss_d_output, buffers);
        rlt::step(device, optimizer, network);
    }
    EXPECT_LE(non_decreasing_count, 5);
}
