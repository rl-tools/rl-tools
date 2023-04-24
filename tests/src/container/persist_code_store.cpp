#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/optimizers/adam/persist_code.h>
#include <backprop_tools/nn/parameters/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn/layers/dense/persist_code.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/persist_code.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>


constexpr bool const_declaration = true;


TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> m;
    bpt::malloc(device, m);
    bpt::randn(device, m, rng);
    bpt::print(device, m);
    auto output = bpt::save(device, m, "matrix_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_backprop_tools_container_persist_matrix.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_DENSE_LAYER){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::nn::layers::dense::LayerBackwardGradient<bpt::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>> layer;
    bpt::malloc(device, layer);
    bpt::init_kaiming(device, layer, rng);
    bpt::zero_gradient(device, layer);
    bpt::reset_forward_state(device, layer);
    bpt::reset_optimizer_state(device, layer, optimizer);
    bpt::randn(device, layer.weights.gradient, rng);
    bpt::randn(device, layer.weights.gradient_first_order_moment, rng);
    bpt::randn(device, layer.weights.gradient_second_order_moment, rng);
    bpt::randn(device, layer.biases.gradient, rng);
    bpt::randn(device, layer.biases.gradient_first_order_moment, rng);
    bpt::randn(device, layer.biases.gradient_second_order_moment, rng);
    auto output = bpt::save(device, layer, "layer_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open("data/test_backprop_tools_nn_layers_dense_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_MLP){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = bpt::nn_models::mlp::InferenceSpecification<bpt::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, 1, bpt::MatrixDynamicTag, true, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    bpt::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    bpt::malloc(device, mlp);
    bpt::init_weights(device, mlp, rng);
    auto output = bpt::save(device, mlp, "mlp_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_backprop_tools_nn_models_mlp_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}
