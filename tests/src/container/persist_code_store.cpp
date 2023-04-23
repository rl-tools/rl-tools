#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/optimizers/adam/persist_code.h>
#include <backprop_tools/nn/parameters/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn/layers/dense/persist_code.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/persist_code.h>

namespace lic = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>


constexpr bool const_declaration = true;


TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> m;
    lic::malloc(device, m);
    lic::randn(device, m, rng);
    lic::print(device, m);
    auto output = lic::save(device, m, "matrix_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_backprop_tools_container_persist_matrix.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_DENSE_LAYER){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    using OPTIMIZER_PARAMETERS = lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::nn::layers::dense::LayerBackwardGradient<lic::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::parameters::Adam>> layer;
    lic::malloc(device, layer);
    lic::init_kaiming(device, layer, rng);
    lic::zero_gradient(device, layer);
    lic::reset_forward_state(device, layer);
    lic::reset_optimizer_state(device, layer, optimizer);
    lic::randn(device, layer.weights.gradient, rng);
    lic::randn(device, layer.weights.gradient_first_order_moment, rng);
    lic::randn(device, layer.weights.gradient_second_order_moment, rng);
    lic::randn(device, layer.biases.gradient, rng);
    lic::randn(device, layer.biases.gradient_first_order_moment, rng);
    lic::randn(device, layer.biases.gradient_second_order_moment, rng);
    auto output = lic::save(device, layer, "layer_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open("data/test_backprop_tools_nn_layers_dense_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_STORE, TEST_MLP){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = lic::nn_models::mlp::InferenceSpecification<lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1, lic::MatrixDynamicTag, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    lic::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    lic::malloc(device, mlp);
    lic::init_weights(device, mlp, rng);
    auto output = lic::save(device, mlp, "mlp_1", const_declaration);
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_backprop_tools_nn_models_mlp_persist_code.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}
