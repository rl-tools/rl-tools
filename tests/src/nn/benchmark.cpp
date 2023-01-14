#include <layer_in_c/operations/cpu.h>
#include <layer_in_c/nn_models/operations_cpu.h>

namespace lic = layer_in_c;




#include <gtest/gtest.h>

#define EIGEN_USE_BLAS
#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>
#include <chrono>



using DTYPE = float;
using DEVICE = lic::devices::DefaultCPU;
using INDEX_TYPE = DEVICE::index_t;

constexpr DEVICE::index_t BATCH_SIZE = 256;
constexpr DEVICE::index_t HIDDEN_DIM = 64;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, 5, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::IDENTITY>;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using InferenceSpecification = lic::nn_models::mlp::InferenceSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>>;
using NetworkType = lic::nn_models::mlp::NeuralNetwork<InferenceSpecification<DTYPE, DEVICE::index_t, lic::nn::activation_functions::IDENTITY>>;

DEVICE::SPEC::LOGGING logger;
DEVICE device(logger);

constexpr INDEX_TYPE ITERATIONS = 1000;

class LAYER_IN_C_NN_DENSE_BENCHMARK : public ::testing::Test
{
protected:
    DTYPE input_lic[BATCH_SIZE * NetworkType::INPUT_DIM];
    DTYPE output_lic[BATCH_SIZE * HIDDEN_DIM];
    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM> expected_output = {(DTYPE*)output_lic};

    NetworkType network;

    virtual void SetUp()
    {
        auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
        lic::malloc(device, network);
        lic::init_weights(device, network, rng);

        for (INDEX_TYPE i = 0; i < BATCH_SIZE * NetworkType::INPUT_DIM; ++i)
        {
            input_lic[i] = lic::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng);
        }




        auto start = std::chrono::high_resolution_clock::now();
        for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
            for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                lic::evaluate(device, network.input_layer, &input_lic[batch_i * NetworkType::INPUT_DIM], &output_lic[batch_i * HIDDEN_DIM]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    }
};





TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, BENCHMARK_BATCH) {
    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM> input_lic_matrix;
    lic::malloc(device, input_lic_matrix);
    memcpy(input_lic_matrix.data, input_lic, sizeof(DTYPE) * BATCH_SIZE * NetworkType::INPUT_DIM);
    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM> output_lic_matrix;
    lic::malloc(device, output_lic_matrix);


    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        lic::evaluate(device, network.input_layer, input_lic_matrix, output_lic_matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "LIC: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    DTYPE abs_diff = lic::abs_diff(device, output_lic_matrix, expected_output);

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}


TEST_F(LAYER_IN_C_NN_DENSE_BENCHMARK, EIGEN_ROW_VS_COLUMN_MAJOR) {
    Eigen::Map<Eigen::Matrix<DTYPE, BATCH_SIZE, NetworkType::INPUT_DIM, Eigen::RowMajor>> input(input_lic);
    Eigen::Map<Eigen::Matrix<DTYPE, HIDDEN_DIM, NetworkType::INPUT_DIM, Eigen::RowMajor>> W((DTYPE*)network.input_layer.weights.data);
    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, HIDDEN_DIM> output_eigen_matrix;
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
    std::cout << "Eigen Row Major: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;

    DTYPE abs_diff = lic::abs_diff(device, output_eigen_matrix, expected_output)/NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);
}

#include "mkl.h"

#include <stdio.h>
#include <stdlib.h>


#define min(x,y) (((x) < (y)) ? (x) : (y))

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

    memcpy(A, network.input_layer.weights.data, m*k*sizeof( DTYPE ));

    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM> input_mkl_matrix({B});

    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::INPUT_DIM> input_lic_matrix;
    lic::malloc(device, input_lic_matrix);
    memcpy(input_lic_matrix.data, input_lic, sizeof(DTYPE) * BATCH_SIZE * NetworkType::INPUT_DIM);
    DTYPE input_abs_diff = lic::abs_diff(device, input_mkl_matrix, input_lic_matrix);

    lic::Matrix<DTYPE, DEVICE::index_t, NetworkType::INPUT_DIM, BATCH_SIZE> input_lic_matrix_transpose;
    lic::malloc(device, input_lic_matrix_transpose);
    lic::transpose(device, input_lic_matrix_transpose, input_lic_matrix);

    memcpy(B, input_lic_matrix_transpose.data, k*n*sizeof( DTYPE ));

    auto start = std::chrono::high_resolution_clock::now();
    for(INDEX_TYPE iteration_i = 0; iteration_i < ITERATIONS; iteration_i++) {
        if constexpr(lic::utils::typing::is_same_v<DTYPE, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
        }
        else{
//            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MKL: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((DTYPE)ITERATIONS) << "us" << std::endl;


    for(INDEX_TYPE batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(INDEX_TYPE output_i=0; output_i < HIDDEN_DIM; output_i++){
            C[batch_i + output_i * BATCH_SIZE] = lic::activation<DEVICE::SPEC::MATH, DTYPE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_ACTIVATION_FUNCTION>(C[batch_i + output_i * BATCH_SIZE] + network.input_layer.biases.data[output_i]);
        }
    }

    lic::Matrix<DTYPE, DEVICE::index_t, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM, BATCH_SIZE> output_mkl_matrix_transpose;
    output_mkl_matrix_transpose.data = C;

    lic::Matrix<DTYPE, DEVICE::index_t, BATCH_SIZE, NetworkType::SPEC::STRUCTURE_SPEC::HIDDEN_DIM> output_mkl_matrix;
    lic::malloc(device, output_mkl_matrix);

    lic::transpose(device, output_mkl_matrix, output_mkl_matrix_transpose);

    DTYPE abs_diff = lic::abs_diff(device, output_mkl_matrix, expected_output) / NetworkType::NUM_WEIGHTS;

    std::cout << "Absolute difference: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);

}

