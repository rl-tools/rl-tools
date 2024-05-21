#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn_models/mlp/operations_cpu.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>


constexpr bool const_declaration = true;


TEST(RL_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST){
    using DEVICE = rlt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM());
    rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> m;
    rlt::malloc(device, m);
    rlt::randn(device, m, rng);
    rlt::print(device, m);
    auto output = rlt::save_code(device, m, "matrix_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_rl_tools_container_persist_matrix.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(RL_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_DENSE_LAYER){
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using DTYPE = float;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<DTYPE, TI>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    OPTIMIZER optimizer;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM());
    using LAYER_SPEC = rlt::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CAPABILITY_ADAM = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam>;
    rlt::nn::layers::dense::Layer<CAPABILITY_ADAM, LAYER_SPEC> layer;
    rlt::malloc(device, layer);
    rlt::init_kaiming(device, layer, rng);
    rlt::zero_gradient(device, layer);
    rlt::reset_forward_state(device, layer);
    rlt::reset_optimizer_state(device, optimizer, layer);
    rlt::randn(device, layer.weights.gradient, rng);
    rlt::randn(device, layer.weights.gradient_first_order_moment, rng);
    rlt::randn(device, layer.weights.gradient_second_order_moment, rng);
    rlt::randn(device, layer.biases.gradient, rng);
    rlt::randn(device, layer.biases.gradient_first_order_moment, rng);
    rlt::randn(device, layer.biases.gradient_second_order_moment, rng);
    auto output = rlt::save_code(device, layer, "layer_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open("data/test_rl_tools_nn_layers_dense_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(RL_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_MLP){
    using DEVICE = rlt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = rlt::nn_models::mlp::Specification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY, 1, rlt::MatrixDynamicTag, true, rlt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using CAPABILITY_ADAM = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam>;
    rlt::nn_models::mlp::NeuralNetwork<CAPABILITY_ADAM, SPEC> mlp;
    rlt::malloc(device, mlp);
    rlt::init_weights(device, mlp, rng);
    auto output = rlt::save_code(device, mlp, "mlp_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_rl_tools_nn_models_mlp_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}
