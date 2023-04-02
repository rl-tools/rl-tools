#include <layer_in_c/operations/cpu/group_1.h>
#include <layer_in_c/operations/arm/group_1.h>
#include <layer_in_c/operations/cpu/group_2.h>
#include <layer_in_c/operations/arm/group_2.h>
#include <layer_in_c/operations/cpu/group_3.h>
#include <layer_in_c/operations/arm/group_3.h>

#include <layer_in_c/containers/persist_code.h>
#include <layer_in_c/nn/layers/dense/operations_arm.h>
#include <layer_in_c/nn/layers/dense/operations_cpu.h>
#include <layer_in_c/nn_models/mlp/operations_cpu.h>
#include <layer_in_c/nn_models/mlp/operations_generic.h>

namespace lic = layer_in_c;

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
//#include "../../../data/test_layer_in_c_nn_models_mlp_persist_code.h"

constexpr bool const_declaration = false;


template <typename DTYPE, auto INPUT_DIM, auto OUTPUT_DIM, auto N_HIDDEN_LAYERS, auto HIDDEN_DIM, lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION, auto BATCH_SIZE>
void test_mlp_evaluate() {
    using DEVICE = lic::devices::DefaultCPU;
    using DEVICE_ARM = lic::devices::DefaultARM;
    DEVICE device;
    DEVICE_ARM device_arm;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, INPUT_DIM, OUTPUT_DIM, N_HIDDEN_LAYERS, HIDDEN_DIM, HIDDEN_ACTIVATION_FUNCTION, ACTIVATION_FUNCTION, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using SPEC = lic::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>;
    lic::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    lic::malloc(device, mlp);
    lic::init_weights(device, mlp, rng);

    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::INPUT_DIM>> input;
    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::OUTPUT_DIM>> output_orig, output_arm;
    lic::malloc(device, input);
    lic::malloc(device, output_orig);
    lic::malloc(device, output_arm);
    lic::randn(device, input, rng);
    lic::evaluate(device, mlp, input, output_orig);
    lic::evaluate(device_arm, mlp, input, output_arm);
    lic::print(device, output_orig);

    auto abs_diff = lic::abs_diff(device, output_orig, output_arm);

    ASSERT_LT(abs_diff, 1e-5);
}TEST(LAYER_IN_C_NN_ARM, TEST_MLP_EVALUATE) {
    test_mlp_evaluate<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 1, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 1, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 1, 1, 2, 1, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 4, 2, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 4, 3, 1, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 4, 30, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_evaluate<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::IDENTITY, lic::nn::activation_functions::ActivationFunction::RELU, 1>();
    test_mlp_evaluate<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::RELU, 1>();
}

template <typename DTYPE, auto INPUT_DIM, auto OUTPUT_DIM, auto N_HIDDEN_LAYERS, auto HIDDEN_DIM, lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION, auto BATCH_SIZE>
void test_mlp_forward() {
    using DEVICE = lic::devices::DefaultCPU;
    using DEVICE_ARM = lic::devices::DefaultARM;
    DEVICE device;
    DEVICE_ARM device_arm;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, INPUT_DIM, OUTPUT_DIM, N_HIDDEN_LAYERS, HIDDEN_DIM, HIDDEN_ACTIVATION_FUNCTION, ACTIVATION_FUNCTION, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using SPEC = lic::nn_models::mlp::BackwardGradientSpecification<STRUCTURE_SPEC>;
    using TYPE = lic::nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>;
    using FORWARD_BACKWARD_BUFFERS = typename TYPE::template BuffersForwardBackward<BATCH_SIZE>;
    TYPE mlp_cpu, mlp_arm;
    FORWARD_BACKWARD_BUFFERS buffers;
    lic::malloc(device, mlp_cpu);
    lic::malloc(device, mlp_arm);
    lic::malloc(device, buffers);
    lic::init_weights(device, mlp_cpu, rng);
    lic::zero_gradient(device, mlp_cpu);
    lic::copy(device, device, mlp_arm, mlp_cpu);

    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::INPUT_DIM>> input;
    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::OUTPUT_DIM>> target;
    lic::malloc(device, input);
    lic::malloc(device, target);
    lic::randn(device, input, rng);
    lic::randn(device, target, rng);
    lic::forward(device, mlp_cpu, input);
    lic::forward(device_arm, mlp_arm, input);
    lic::print(device, mlp_arm.output_layer.output);

    auto abs_diff_output = lic::abs_diff(device, mlp_arm.output_layer.output, mlp_arm.output_layer.output);
    auto abs_diff_network = lic::abs_diff(device, mlp_arm, mlp_cpu);

    ASSERT_LT(abs_diff_output, 1e-5);
    ASSERT_LT(abs_diff_network, 1e-5);
}

TEST(LAYER_IN_C_NN_ARM, TEST_MLP_FORWARD){
    test_mlp_forward<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 1, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 1, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 1, 1, 2, 1, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 4, 2, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 4, 3, 1, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 4, 30, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1>();
    test_mlp_forward<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::IDENTITY, lic::nn::activation_functions::ActivationFunction::RELU, 1>();
    test_mlp_forward<double, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::RELU, 1>();
}
