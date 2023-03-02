#include <layer_in_c/operations/cpu/group_1.h>
#include <layer_in_c/operations/arm/group_1.h>
#include <layer_in_c/operations/cpu/group_2.h>
#include <layer_in_c/operations/arm/group_2.h>
#include <layer_in_c/operations/cpu/group_3.h>
#include <layer_in_c/operations/arm/group_3.h>

#include <layer_in_c/containers/persist_code.h>
#include <layer_in_c/nn/layers/dense/operations_cpu.h>
#include <layer_in_c/nn/layers/dense/operations_arm.h>
#include <layer_in_c/nn_models/mlp/operations_cpu.h>
#include <layer_in_c/nn_models/mlp/operations_generic.h>

namespace lic = layer_in_c;

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
//#include "../../../data/test_layer_in_c_nn_models_mlp_persist_code.h"

constexpr bool const_declaration = false;

TEST(LAYER_IN_C_NN_ARM, TEST_MLP_EVALUATE){
    using DEVICE = lic::devices::DefaultCPU;
    using DEVICE_ARM = lic::devices::DefaultARM;
    using DTYPE = float;
    constexpr typename DEVICE::index_t BATCH_SIZE = 1;
    DEVICE device;
    DEVICE_ARM device_arm;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
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

}
