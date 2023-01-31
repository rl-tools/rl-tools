#include <layer_in_c/operations/cpu.h>
//#include <layer_in_c/operations/cpu_mkl.h>
#include <layer_in_c/nn_models/operations_cpu.h>

namespace lic = layer_in_c;

#include <gtest/gtest.h>

//#define EIGEN_USE_BLAS
//#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>
#include <chrono>



using DTYPE = float;
using DEVICE = lic::devices::DefaultCPU;
using INDEX_TYPE = DEVICE::index_t;

constexpr DEVICE::index_t BATCH_SIZE = 256;
constexpr DEVICE::index_t HIDDEN_DIM = 64;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, 5, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::RELU, BATCH_SIZE>;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using InferenceSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using NetworkType = lic::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE::index_t, lic::nn::activation_functions::RELU>>;


constexpr INDEX_TYPE ITERATIONS = 100;

class LAYER_IN_C_NN_DENSE_BENCHMARK : public ::testing::Test
{
protected:
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input;
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> expected_output_input_layer;
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::OUTPUT_DIM>> expected_output_output_layer;
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::OUTPUT_DIM>> output_target;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;


    NetworkType network;
    NetworkType network_mkl;
    LAYER_IN_C_NN_DENSE_BENCHMARK(): device(this->logger){
        lic::malloc(device, input);
        lic::malloc(device, expected_output_input_layer);
        lic::malloc(device, expected_output_output_layer);
        lic::malloc(device, output_target);
        auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
        lic::malloc(device, network);
        network.age = 100000;
        lic::malloc(device, network_mkl);
        lic::init_weights(device, network, rng);
        lic::copy(device, device, network_mkl, network);
        assert(network_mkl.age == network.age);

        for(INDEX_TYPE i = 0; i < BATCH_SIZE * NetworkType::INPUT_DIM; ++i)
        {
            input.data[i] = lic::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng);
        }

        for(INDEX_TYPE i = 0; i < BATCH_SIZE * NetworkType::OUTPUT_DIM; ++i)
        {
            output_target.data[i] = lic::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
            for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, NetworkType::INPUT_DIM>> input_sample = {&input.data[batch_i * NetworkType::INPUT_DIM]};
                lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, 1, HIDDEN_DIM>> output_sample = {&expected_output_input_layer.data[batch_i * HIDDEN_DIM]};
                lic::evaluate(device, network.input_layer, input_sample, output_sample);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

        lic::forward(device, network, input);

        lic::copy(device, device, expected_output_output_layer, network.output_layer.output);

        lic::reset_optimizer_state(device, network);
        lic::zero_gradient(device, network);
        {
            auto start = std::chrono::high_resolution_clock::now();
            for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
                lic::forward_backward_mse(device, network, input, output_target);
            }
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "LIC forward backward mse: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;
        }
    }
};





TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, BENCHMARK_BATCH) {
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_batch;
    lic::malloc(device, output_batch);
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::evaluate(device, network.input_layer, input, output_batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    DTYPE abs_diff = lic::abs_diff(device, output_batch, expected_output_input_layer);

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}


TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, EIGEN_ROW_VS_COLUMN_MAJOR) {
    Eigen::Map<Eigen::Matrix<DTYPE, BATCH_SIZE, NetworkType::INPUT_DIM, Eigen::RowMajor>> input(this->input.data);
    Eigen::Map<Eigen::Matrix<DTYPE, HIDDEN_DIM, NetworkType::INPUT_DIM, Eigen::RowMajor>> W((DTYPE*)network.input_layer.weights.data);
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_eigen_matrix;
    lic::malloc(device, output_eigen_matrix);
    Eigen::Map<Eigen::Matrix<DTYPE, HIDDEN_DIM, BATCH_SIZE, Eigen::ColMajor>> output(output_eigen_matrix.data);

   for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        output = W * input.transpose();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        output = W * input.transpose();
    }
    output.colwise() += Eigen::Map<Eigen::Matrix<DTYPE, HIDDEN_DIM, 1>>((DTYPE*)network.input_layer.biases.data);
    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE hidden_i = 0; hidden_i < HIDDEN_DIM; hidden_i++){
            output(hidden_i, batch_i) = lic::activation<DEVICE::SPEC::MATH, DTYPE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(output(hidden_i, batch_i));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MatMul: Eigen Row Major: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device, output_eigen_matrix, expected_output_input_layer)/NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);
}

#ifdef LAYER_IN_C_TEST_ENABLE_MKL
#include "mkl.h"
#endif

#include <stdio.h>
#include <stdlib.h>


#define min(x,y) (((x) < (y)) ? (x) : (y))

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/devices/cpu_mkl.h>
#include <layer_in_c/containers/operations_cpu_mkl.h>
TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, MKL) {
    DTYPE *A, *B, *C;
    int m, n, k;
    DTYPE alpha, beta;

    m = HIDDEN_DIM, k = HIDDEN_DIM, n = BATCH_SIZE;
    // A m x k
    // B k x n
    // C m x n
    alpha = 1.0; beta = 0.0;

    A = (DTYPE *)mkl_malloc( m*k*sizeof( DTYPE ), 64 );
    B = (DTYPE *)mkl_malloc( k*n*sizeof( DTYPE ), 64 );
    C = (DTYPE *)mkl_malloc( m*n*sizeof( DTYPE ), 64 );

    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i = 0; output_i < HIDDEN_DIM; output_i++){
            C[output_i * BATCH_SIZE + batch_i] = 0;
        }
    }

    memcpy(A, network_mkl.input_layer.weights.data, m*k*sizeof( DTYPE ));

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_mkl_matrix({B});

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_lic_matrix;
    lic::malloc(device, input_lic_matrix);
    memcpy(input_lic_matrix.data, input.data, sizeof(DTYPE) * BATCH_SIZE * NetworkType::INPUT_DIM);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, NetworkType::INPUT_DIM, BATCH_SIZE>> input_lic_matrix_transpose;
    lic::malloc(device, input_lic_matrix_transpose);
    lic::transpose(device, input_lic_matrix_transpose, input_lic_matrix);

    memcpy(B, input_lic_matrix_transpose.data, k*n*sizeof( DTYPE ));

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (float*)A, k, (float*)B, n, beta, (float*)C, n);
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (double*)A, k, (double*)B, n, beta, (double*)C, n);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i=0; output_i < HIDDEN_DIM; output_i++){
            C[batch_i + output_i * BATCH_SIZE] = lic::activation<DEVICE::SPEC::MATH, DTYPE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(C[batch_i + output_i * BATCH_SIZE] + network_mkl.input_layer.biases.data[output_i]);
        }
    }

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM, BATCH_SIZE>> output_mkl_matrix_transpose;
    output_mkl_matrix_transpose.data = C;

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_mkl_matrix;
    lic::malloc(device, output_mkl_matrix);

    lic::transpose(device, output_mkl_matrix, output_mkl_matrix_transpose);

    DTYPE abs_diff = lic::abs_diff(device, output_mkl_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}

#include <layer_in_c/nn/operations_cpu_mkl.h>
#include <layer_in_c/containers/operations_generic.h>
#include <layer_in_c/utils/generic/typing.h>

TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, MKL_LAYER) {
    using DEVICE_MKL = lic::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl(device.logger);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_matrix;
    lic::malloc(device_mkl, output_matrix);
    lic::set(device_mkl, output_matrix, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::evaluate(device_mkl, network_mkl.input_layer, input, output_matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC evaluate: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device_mkl, output_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, MKL_LAYER_FORWARD) {
    std::cout << "Layer batch size: " << decltype(network_mkl.input_layer)::SPEC::BATCH_SIZE << std::endl;
    using DEVICE_MKL = lic::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl(device.logger);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_matrix;
    lic::malloc(device_mkl, output_matrix);
    lic::set(device_mkl, output_matrix, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::forward(device_mkl, network_mkl.input_layer, input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC evaluate: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device_mkl, network_mkl.input_layer.output, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, MKL_MODEL_FORWARD) {
    std::cout << "Layer batch size: " << decltype(network_mkl.input_layer)::SPEC::BATCH_SIZE << std::endl;
    using DEVICE_MKL = lic::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl(device.logger);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::forward(device_mkl, network_mkl, input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC evaluate full: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device_mkl, network_mkl.output_layer.output, expected_output_output_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}

TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, MKL_MODEL_BACKWARD) {
    using DEVICE_MKL = lic::devices::CPU_MKL<DEVICE::SPEC>;
    DEVICE_MKL device_mkl(device.logger);

    lic::reset_optimizer_state(device_mkl, network_mkl);
    lic::zero_gradient(device_mkl, network_mkl);
    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::forward_backward_mse(device_mkl, network_mkl, input, output_target);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC forward backward mse: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

//    DTYPE abs_diff = lic::abs_diff(device_mkl, network_mkl.output_layer.d_weights, network.output_layer.d_weights) / NetworkType::NUM_WEIGHTS;

    DTYPE abs_diff = lic::abs_diff(device_mkl, network_mkl, network) / NetworkType::NUM_WEIGHTS;

    if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-4);
    }
    else{
        EXPECT_LT(abs_diff, 1e-12);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}
#endif

#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, ACCELERATE) {
    DTYPE *A, *B, *C;
    int m, n, k;
    DTYPE alpha, beta;

    m = HIDDEN_DIM, k = HIDDEN_DIM, n = BATCH_SIZE;
    // A m x k
    // B k x n
    // C m x n
    alpha = 1.0; beta = 0.0;

    A = (DTYPE *)malloc( m*k*sizeof( DTYPE ));
    B = (DTYPE *)malloc( k*n*sizeof( DTYPE ));
    C = (DTYPE *)malloc( m*n*sizeof( DTYPE ));

    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i = 0; output_i < HIDDEN_DIM; output_i++){
            C[output_i * BATCH_SIZE + batch_i] = 0;
        }
    }

    memcpy(A, network_mkl.input_layer.weights.data, m*k*sizeof( DTYPE ));

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_mkl_matrix({B});

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM>> input_lic_matrix;
    lic::malloc(device, input_lic_matrix);
    memcpy(input_lic_matrix.data, input.data, sizeof(DTYPE) * BATCH_SIZE * NetworkType::INPUT_DIM);
    DTYPE input_abs_diff = lic::abs_diff(device, input_mkl_matrix, input_lic_matrix);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, NetworkType::INPUT_DIM, BATCH_SIZE>> input_lic_matrix_transpose;
    lic::malloc(device, input_lic_matrix_transpose);
    lic::transpose(device, input_lic_matrix_transpose, input_lic_matrix);

    memcpy(B, input_lic_matrix_transpose.data, k*n*sizeof( DTYPE ));

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (float*)A, k, (float*)B, n, beta, (float*)C, n);
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (double*)A, k, (double*)B, n, beta, (double*)C, n);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "ACCELERATE: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i=0; output_i < HIDDEN_DIM; output_i++){
            C[batch_i + output_i * BATCH_SIZE] = lic::activation<DEVICE::SPEC::MATH, DTYPE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(C[batch_i + output_i * BATCH_SIZE] + network_mkl.input_layer.biases.data[output_i]);
        }
    }

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM, BATCH_SIZE>> output_mkl_matrix_transpose;
    output_mkl_matrix_transpose.data = C;

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_mkl_matrix;
    lic::malloc(device, output_mkl_matrix);

    lic::transpose(device, output_mkl_matrix, output_mkl_matrix_transpose);

    DTYPE abs_diff = lic::abs_diff(device, output_mkl_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}
#endif
#ifdef LAYER_IN_C_BACKEND_ENABLE_CUBLAS

#include <layer_in_c/devices/cublas.h>
#include <layer_in_c/containers/operations_cublas.h>
TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, CUBLAS) {
    using DEVICE_CUBLAS = lic::devices::CUBLAS<DEVICE::SPEC>;
    DEVICE_CUBLAS device_cublas(device.logger);
    NetworkType network_cublas;
    lic::malloc(device_cublas, network_cublas);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM>> output_matrix;
    lic::malloc(device_cublas, output_matrix);
    lic::set(device_cublas, output_matrix, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::evaluate(device_cublas, network_mkl.input_layer, input, output_matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL LIC evaluate: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device_cublas, output_matrix, expected_output_input_layer) / NetworkType::NUM_WEIGHTS;

    if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
        EXPECT_LT(abs_diff, 1e-6);
    }
    else{
        EXPECT_LT(abs_diff, 1e-14);
    }

    std::cout << "Absolute difference: " << abs_diff << std::endl;
}
#endif
