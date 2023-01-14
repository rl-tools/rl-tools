#include <layer_in_c/operations/cpu.h>
#include <layer_in_c/nn_models/operations_cpu.h>

namespace lic = layer_in_c;




#include <gtest/gtest.h>

#define EIGEN_USE_BLAS
#include <Eigen/Eigen>
#include <chrono>


using DTYPE = float;
using DEVICE = lic::devices::DefaultCPU;
using INDEX_TYPE = DEVICE::index_t;

constexpr DEVICE::index_t BATCH_SIZE = 256;
constexpr DEVICE::index_t HIDDEN_DIM = 64;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, 64, 5, 3, 64, ACTIVATION_FUNCTION, lic::nn::activation_functions::IDENTITY>;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using InferenceSpecification = lic::nn_models::mlp::InferenceSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>>;


constexpr INDEX_TYPE ITERATIONS = 1000;


TEST(LAYER_IN_C_NN_DENSE_BENCHMARK, EIGEN_ROW_VS_COLUMN_MAJOR) {
    using NetworkType = lic::nn_models::mlp::NeuralNetwork<InferenceSpecification<DTYPE, DEVICE::index_t, lic::nn::activation_functions::RELU>>;
    constexpr auto HIDDEN_DIM = NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM;
    Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic> input = Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(BATCH_SIZE, NetworkType::INPUT_DIM);
    Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic> W = Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(HIDDEN_DIM, NetworkType::INPUT_DIM);
    Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic> output;

    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        output = W * input.transpose();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        output = W * input.transpose();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Eigen Row Major: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

}

TEST(LAYER_IN_C_NN_DENSE_BENCHMARK, BENCHMARK) {
    using NetworkType = lic::nn_models::mlp::NeuralNetwork<InferenceSpecification<DTYPE, DEVICE::index_t, lic::nn::activation_functions::RELU>>;
    NetworkType network;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device(logger);
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::init_weights(device, network, rng);


    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){

    }

}
