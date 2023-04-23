#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/mlp/operations_cpu.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "../../../data/test_backprop_tools_container_persist_matrix.h"

constexpr bool const_declaration = false;

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> orig;
    bpt::malloc(device, orig);
    bpt::randn(device, orig, rng);
    std::cout << "orig: " << std::endl;
    bpt::print(device, orig);
    std::cout << "loaded: " << std::endl;
    bpt::print(device, matrix_1::container);

    auto abs_diff = bpt::abs_diff(device, orig, matrix_1::container);
    ASSERT_FLOAT_EQ(0, abs_diff);
}

#include "../../../data/test_backprop_tools_nn_layers_dense_persist_code.h"

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::nn::layers::dense::Layer<bpt::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, bpt::nn::activation_functions::ActivationFunction::RELU>> layer;
    bpt::malloc(device, layer);
    bpt::init_kaiming(device, layer, rng);
    bpt::increment(layer.weights.parameters, 2, 1, 10);
    auto abs_diff = bpt::abs_diff(device, layer, layer_1::layer);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER_ADAM){
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
    bpt::increment(layer.weights.parameters, 2, 1, 10);
    bpt::increment(layer.weights.gradient, 2, 1, 5);
    bpt::increment(layer.weights.gradient_first_order_moment, 2, 1, 2);
    bpt::increment(layer.weights.gradient_second_order_moment, 2, 1, 1);
    auto abs_diff = bpt::abs_diff(device, layer, layer_1::layer);
    ASSERT_FLOAT_EQ(10 + 5 + 2 + 1, abs_diff);
}

#include "../../../data/test_backprop_tools_nn_models_mlp_persist_code.h"

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = bpt::nn_models::mlp::InferenceSpecification<bpt::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, 1, bpt::MatrixDynamicTag, true, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    bpt::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    bpt::malloc(device, mlp);
    bpt::init_weights(device, mlp, rng);
    bpt::increment(mlp.hidden_layers[0].biases.parameters, 0, 2, 10);
    auto abs_diff = bpt::abs_diff(device, mlp, mlp_1::mlp);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP_ADAM){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = bpt::nn_models::mlp::AdamSpecification<bpt::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, 1, bpt::MatrixDynamicTag, true, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    bpt::nn_models::mlp::NeuralNetworkAdam<SPEC> mlp;
    bpt::malloc(device, mlp);
    bpt::init_weights(device, mlp, rng);
    bpt::zero_gradient(device, mlp);
    bpt::reset_forward_state(device, mlp);
    bpt::reset_optimizer_state(device, mlp, optimizer);
    bpt::increment(mlp.hidden_layers[0].biases.parameters, 0, 2, 10);
    bpt::copy(device, device, mlp.input_layer, mlp_1::input_layer::layer);
    auto abs_diff = bpt::abs_diff(device, mlp, mlp_1::mlp);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(BACKPROP_TOOLS_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP_EVALUATE){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    constexpr typename DEVICE::index_t BATCH_SIZE = 10;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, 1, bpt::MatrixDynamicTag, true, bpt::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using SPEC = bpt::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>;
    bpt::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    bpt::malloc(device, mlp);
    bpt::init_weights(device, mlp, rng);

    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::INPUT_DIM>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::OUTPUT_DIM>> output_orig, output_loaded;
    bpt::malloc(device, input);
    bpt::malloc(device, output_orig);
    bpt::malloc(device, output_loaded);
    bpt::randn(device, input, rng);
    bpt::evaluate(device, mlp, input, output_orig);
    bpt::evaluate(device, mlp_1::mlp, input, output_loaded);
    bpt::print(device, output_orig);

    auto output = bpt::save(device, input, "input", const_declaration);
    output += bpt::save(device, output_orig, "expected_output", const_declaration);

    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_backprop_tools_nn_models_mlp_evaluation.h");
    file << output;
    file.close();

    auto abs_diff = bpt::abs_diff(device, output_orig, output_loaded);
    ASSERT_FLOAT_EQ(0, abs_diff);
}
