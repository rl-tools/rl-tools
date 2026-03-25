#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/containers/tensor/operations_cuda.h>
#include <rl_tools/random/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "../../../../../../tests/data/test_nn_layers_gru_persist_code.h"

namespace rlt = rl_tools;

#include <gtest/gtest.h>


using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using TI = typename DEVICE_GPU::index_t;

constexpr T EPSILON = 1e-10;
constexpr TI SEQUENCE_LENGTH = 1;
constexpr TI BATCH_SIZE = 3;
constexpr TI INPUT_DIM = 4;
constexpr TI HIDDEN_DIM = 5;

using GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
using GRU_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
using GRU = GRU_TEMPLATE::Layer<CAPABILITY, INPUT_SHAPE>;

#ifndef DISABLED_HELPERS
TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_HELPERS_CUBLAS_SGEMM){
    cublasHandle_t handle;
    cublasCreate(&handle);
    constexpr TI M = 10;
    constexpr TI N = 10;
    constexpr TI K = 1;
    T alpha = 1;
    T beta = 0;
    T* A = nullptr;
    T* B = nullptr;
    T* output = nullptr;
    T A_cpu[M][K];
    T B_cpu[K][N];
    T output_cpu[M][N];
    for (TI i = 0; i < M; i++){
        for (TI j = 0; j < K; j++){
            A_cpu[i][j] = i;
        }
    }
    for (TI i = 0; i < K; i++){
        for (TI j = 0; j < N; j++){
            B_cpu[i][j] = j;
        }
    }

    cudaMalloc(&A, M * K * sizeof(T));
    cudaMalloc(&B, N * K * sizeof(T));
    cudaMalloc(&output, M * N * sizeof(T));
    cudaMemcpy(A, A_cpu, M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_cpu, N * K * sizeof(T), cudaMemcpyHostToDevice);
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 10, 10, 1, &alpha, B, N, A, K, &beta, output, M);
    cudaMemcpy(output_cpu, output, M * N * sizeof(T), cudaMemcpyDeviceToHost);
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cout << "CUBLAS_STATUS_SUCCESS" << std::endl;
    }
    for (TI i = 0; i < M; i++){
        for (TI j = 0; j < N; j++){
            ASSERT_EQ(output_cpu[i][j], i * j);
            std::cout << output_cpu[i][j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(output);
    cublasDestroy(handle);
}

template <TI M, TI N, TI K>
void test_gru_helper_cuda(){
    constexpr T EPSILON = 1e-10;
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU gru_cpu, gru_gpu;
    GRU::Buffer<> gru_buffer_cpu, gru_buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, M, K>>> A_cpu, A_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, K>>> B_T_cpu, B_T_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, M, N>>> C_cpu, C_cpu_generic, C_cpu_manual, C_gpu, C_gpu_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N>>> bias_cpu, bias_gpu;

    rlt::init(device_cpu);
    rlt::init(device_gpu);

    rlt::malloc(device_cpu, A_cpu);
    rlt::malloc(device_gpu, A_gpu);
    rlt::malloc(device_cpu, B_T_cpu);
    rlt::malloc(device_gpu, B_T_gpu);
    rlt::malloc(device_cpu, C_cpu);
    rlt::malloc(device_cpu, C_cpu_generic);
    rlt::malloc(device_cpu, C_cpu_manual);
    rlt::malloc(device_gpu, C_gpu);
    rlt::malloc(device_cpu, C_gpu_cpu);
    rlt::malloc(device_cpu, bias_cpu);
    rlt::malloc(device_gpu, bias_gpu);

    rlt::randn(device_cpu, A_cpu, rng_cpu);
    rlt::randn(device_cpu, B_T_cpu, rng_cpu);
    rlt::randn(device_cpu, bias_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, A_cpu, A_gpu);
    rlt::copy(device_cpu, device_gpu, B_T_cpu, B_T_gpu);
    rlt::copy(device_cpu, device_gpu, bias_cpu, bias_gpu);


    auto A_cpu_matrix_view = rlt::matrix_view(device_cpu, A_cpu);
    auto B_T_cpu_matrix_view = rlt::matrix_view(device_cpu, B_T_cpu);
    auto B_cpu_matrix_view = rlt::view_transpose(device_cpu, B_T_cpu_matrix_view);
    auto C_cpu_generic_matrix_view = rlt::matrix_view(device_cpu, C_cpu_generic);
    auto C_cpu_manual_matrix_view = rlt::matrix_view(device_cpu, C_cpu_manual);

    for (TI step=0; step <= 1; step++){
        if (step == 0){
            rlt::multiply(device_cpu, A_cpu_matrix_view, B_cpu_matrix_view, C_cpu_generic_matrix_view);
        }
        else{
            for (TI i = 0; i < M; i++){
                for (TI j = 0; j < N; j++){
                    rlt::set(C_cpu_generic_matrix_view, i, j, rlt::get(device_cpu, bias_cpu, j));
                }
            }
            rlt::multiply_accumulate(device_cpu, A_cpu_matrix_view, B_cpu_matrix_view, C_cpu_generic_matrix_view);
        }
        for (TI i = 0; i < M; i++){
            for (TI j = 0; j < N; j++){
                T sum = 0;
                if (step == 1){
                    sum = rlt::get(device_cpu, bias_cpu, j);
                }
                for (TI k = 0; k < K; k++){
                    sum += get(device_cpu, A_cpu, i, k) * get(device_cpu, B_T_cpu, j, k);
                }
                rlt::set(C_cpu_manual_matrix_view, i, j, sum);
            }
        }
        T abs_diff_generic_manual = rlt::abs_diff(device_cpu, C_cpu_generic, C_cpu_manual);
        if (step == 0){
            std::cout << "abs_diff_generic_manual: " << abs_diff_generic_manual << std::endl;
            rlt::utils::assert_exit(device_cpu, abs_diff_generic_manual < EPSILON, "abs_diff_generic_manual >= EPSILON");
        }
        else{
            std::cout << "abs_diff_generic_manual_accumulate: " << abs_diff_generic_manual << std::endl;
            rlt::utils::assert_exit(device_cpu, abs_diff_generic_manual < EPSILON, "abs_diff_generic_manual >= EPSILON");
        }
    }
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias(device_cpu, B_T_cpu, A_cpu, bias_cpu, C_cpu);

    T abs_diff_helper_generic = rlt::abs_diff(device_cpu, C_cpu, C_cpu_generic);
    std::cout << "abs_diff_helper_generic: " << abs_diff_helper_generic << std::endl;
    rlt::utils::assert_exit(device_cpu, abs_diff_helper_generic < EPSILON, "abs_diff_helper_generic >= EPSILON");

    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias(device_gpu, B_T_gpu, A_gpu, bias_gpu, C_gpu);
    rlt::copy(device_gpu, device_cpu, C_gpu, C_gpu_cpu);
    T abs_diff_helper_gpu = rlt::abs_diff(device_cpu, C_cpu, C_gpu_cpu);
    std::cout << "abs_diff_helper_gpu: " << abs_diff_helper_gpu << std::endl;
    rlt::utils::assert_exit(device_cpu, abs_diff_helper_gpu < EPSILON, "abs_diff_helper_gpu >= EPSILON");


    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_cpu, B_T_cpu, A_cpu, bias_cpu, C_cpu);
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_gpu, B_T_gpu, A_gpu, bias_gpu, C_gpu);
    rlt::copy(device_gpu, device_cpu, C_gpu, C_gpu_cpu);
    T abs_diff_helper_gpu_accumulate = rlt::abs_diff(device_cpu, C_cpu, C_gpu_cpu);
    std::cout << "abs_diff_helper_gpu: " << abs_diff_helper_gpu_accumulate << std::endl;
    rlt::utils::assert_exit(device_cpu, abs_diff_helper_gpu_accumulate < EPSILON, "abs_diff_helper_gpu >= EPSILON");

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, K>>> B_broadcast_cpu, B_broadcast_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, M>>> bias_broadcast_cpu, bias_broadcast_gpu;
    rlt::malloc(device_cpu, B_broadcast_cpu);
    rlt::malloc(device_gpu, B_broadcast_gpu);
    rlt::malloc(device_cpu, bias_broadcast_cpu);
    rlt::malloc(device_gpu, bias_broadcast_gpu);
    rlt::randn(device_cpu, B_broadcast_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, B_broadcast_cpu, B_broadcast_gpu);
    rlt::randn(device_cpu, bias_broadcast_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, bias_broadcast_cpu, bias_broadcast_gpu);
    auto C_T_cpu_view = rlt::permute(device_cpu, C_cpu, rlt::tensor::PermutationSpec<1, 0>{});
    rlt::nn::layers::gru::helper::matrix_multiply_broadcast_transpose_bias(device_cpu, A_cpu, B_broadcast_cpu, bias_broadcast_cpu, C_T_cpu_view);
    auto C_T_gpu_view = rlt::permute(device_gpu, C_gpu, rlt::tensor::PermutationSpec<1, 0>{});
    auto C_T_gpu_cpu_view = rlt::permute(device_gpu, C_gpu_cpu, rlt::tensor::PermutationSpec<1, 0>{});
    rlt::nn::layers::gru::helper::matrix_multiply_broadcast_transpose_bias(device_gpu, A_gpu, B_broadcast_gpu, bias_broadcast_gpu, C_T_gpu_view);
    rlt::copy(device_gpu, device_cpu, C_gpu, C_gpu_cpu);

    T abs_diff_broadcast = rlt::abs_diff(device_cpu, C_cpu, C_gpu_cpu);
    std::cout << "abs_diff_broadcast: " << abs_diff_broadcast << std::endl;
    rlt::print(device_cpu, C_T_cpu_view);
    rlt::print(device_cpu, C_T_gpu_cpu_view);
    rlt::utils::assert_exit(device_cpu, abs_diff_broadcast < EPSILON, "abs_diff_broadcast >= EPSILON");
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_HELPERS_CUDA){
    test_gru_helper_cuda<3, 4, 5>();
    test_gru_helper_cuda<1, 1, 1>();
    test_gru_helper_cuda<10, 1, 1>();
    test_gru_helper_cuda<1, 10, 1>();
    test_gru_helper_cuda<1, 1, 10>();
    test_gru_helper_cuda<1, 10, 10>();
    test_gru_helper_cuda<10, 10, 10>();
    test_gru_helper_cuda<10, 10, 1>();
    test_gru_helper_cuda<100, 10, 100>();
}

#endif

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_SIGMOID_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 5>>> t_cpu, t_gpu, t_gpu_cpu;
    rlt::malloc(device_cpu, t_cpu);
    rlt::malloc(device_gpu, t_gpu);
    rlt::malloc(device_cpu, t_gpu_cpu);
    for(TI i = 0; i < 3; i++) for(TI j = 0; j < 5; j++) rlt::set(device_cpu, t_cpu, (T)(i*5+j)*0.1 - 0.7, i, j);
    rlt::copy(device_cpu, device_gpu, t_cpu, t_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 5>>> t_cpu_copy, t_gpu_copy, t_gpu_copy_cpu;
    rlt::malloc(device_cpu, t_cpu_copy);
    rlt::malloc(device_gpu, t_gpu_copy);
    rlt::malloc(device_cpu, t_gpu_copy_cpu);
    rlt::copy(device_cpu, device_cpu, t_cpu, t_cpu_copy);
    rlt::copy(device_cpu, device_gpu, t_cpu, t_gpu_copy);

    rlt::fast_sigmoid(device_cpu, t_cpu);
    rlt::fast_sigmoid(device_gpu, t_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, t_gpu, t_gpu_cpu);
    T diff_fast_sigmoid = rlt::abs_diff(device_cpu, t_cpu, t_gpu_cpu);
    std::cout << "fast_sigmoid CPU vs GPU abs_diff: " << diff_fast_sigmoid << std::endl;

    rlt::sigmoid(device_cpu, t_cpu_copy);
    rlt::sigmoid(device_gpu, t_gpu_copy);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, t_gpu_copy, t_gpu_copy_cpu);
    T diff_sigmoid = rlt::abs_diff(device_cpu, t_cpu_copy, t_gpu_copy_cpu);
    std::cout << "sigmoid CPU vs GPU abs_diff: " << diff_sigmoid << std::endl;

    T diff = diff_fast_sigmoid + diff_sigmoid;
    rlt::free(device_cpu, t_cpu_copy);
    rlt::free(device_gpu, t_gpu_copy);
    rlt::free(device_cpu, t_gpu_copy_cpu);
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, t_cpu);
    rlt::free(device_gpu, t_gpu);
    rlt::free(device_cpu, t_gpu_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_COPY_VIEW_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 15>>> full_cpu, full_gpu, full_gpu_cpu;
    rlt::malloc(device_cpu, full_cpu);
    rlt::malloc(device_gpu, full_gpu);
    rlt::malloc(device_cpu, full_gpu_cpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, full_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, full_cpu, full_gpu);
    auto src_cpu = rlt::view_range(device_cpu, full_cpu, 2 * (TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    auto dst_cpu = rlt::view_range(device_cpu, full_cpu, 0, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    auto src_gpu = rlt::view_range(device_gpu, full_gpu, 2 * (TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    auto dst_gpu = rlt::view_range(device_gpu, full_gpu, 0, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    rlt::copy(device_cpu, device_cpu, src_cpu, dst_cpu);
    rlt::copy(device_gpu, device_gpu, src_gpu, dst_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, full_gpu, full_gpu_cpu);
    T diff = rlt::abs_diff(device_cpu, full_cpu, full_gpu_cpu);
    std::cout << "copy view abs_diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, full_cpu);
    rlt::free(device_gpu, full_gpu);
    rlt::free(device_cpu, full_gpu_cpu);
    rlt::free(device_cpu, rng_cpu);
}

template <typename DEVICE, typename SPEC>
RL_TOOLS_FUNCTION_PLACEMENT void indirect_fast_sigmoid(DEVICE& device, rlt::Tensor<SPEC>& t){
    rlt::fast_sigmoid(device, t);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_INDIRECT_FAST_SIGMOID_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 5>>> t_cpu, t_gpu, t_gpu_cpu;
    rlt::malloc(device_cpu, t_cpu);
    rlt::malloc(device_gpu, t_gpu);
    rlt::malloc(device_cpu, t_gpu_cpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, t_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, t_cpu, t_gpu);
    indirect_fast_sigmoid(device_cpu, t_cpu);
    indirect_fast_sigmoid(device_gpu, t_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, t_gpu, t_gpu_cpu);
    T diff = rlt::abs_diff(device_cpu, t_cpu, t_gpu_cpu);
    std::cout << "indirect fast_sigmoid abs_diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, t_cpu);
    rlt::free(device_gpu, t_gpu);
    rlt::free(device_cpu, t_gpu_cpu);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_FAST_SIGMOID_VIEW_RANGE_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3 * HIDDEN_DIM>>> full_cpu, full_gpu, full_gpu_cpu;
    rlt::malloc(device_cpu, full_cpu);
    rlt::malloc(device_gpu, full_gpu);
    rlt::malloc(device_cpu, full_gpu_cpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, full_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, full_cpu, full_gpu);
    auto rz_cpu = rlt::view_range(device_cpu, full_cpu, (TI)0, rlt::tensor::ViewSpec<1, 2 * HIDDEN_DIM>{});
    auto rz_gpu = rlt::view_range(device_gpu, full_gpu, (TI)0, rlt::tensor::ViewSpec<1, 2 * HIDDEN_DIM>{});
    rlt::fast_sigmoid(device_cpu, rz_cpu);
    rlt::fast_sigmoid(device_gpu, rz_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, full_gpu, full_gpu_cpu);
    T diff = rlt::abs_diff(device_cpu, full_cpu, full_gpu_cpu);
    std::cout << "fast_sigmoid on view_range abs_diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, full_cpu);
    rlt::free(device_gpu, full_gpu);
    rlt::free(device_cpu, full_gpu_cpu);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_MULTIPLY_ACCUMULATE_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, HIDDEN_DIM>>> a_cpu, a_gpu, b_cpu, b_gpu, c_cpu, c_gpu, c_gpu_cpu;
    rlt::malloc(device_cpu, a_cpu);
    rlt::malloc(device_gpu, a_gpu);
    rlt::malloc(device_cpu, b_cpu);
    rlt::malloc(device_gpu, b_gpu);
    rlt::malloc(device_cpu, c_cpu);
    rlt::malloc(device_gpu, c_gpu);
    rlt::malloc(device_cpu, c_gpu_cpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, a_cpu, rng_cpu);
    rlt::randn(device_cpu, b_cpu, rng_cpu);
    rlt::randn(device_cpu, c_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, a_cpu, a_gpu);
    rlt::copy(device_cpu, device_gpu, b_cpu, b_gpu);
    rlt::copy(device_cpu, device_gpu, c_cpu, c_gpu);
    rlt::multiply_accumulate(device_cpu, a_cpu, b_cpu, c_cpu);
    rlt::multiply_accumulate(device_gpu, a_gpu, b_gpu, c_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, c_gpu, c_gpu_cpu);
    T diff = rlt::abs_diff(device_cpu, c_cpu, c_gpu_cpu);
    std::cout << "multiply_accumulate abs_diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, a_cpu);
    rlt::free(device_gpu, a_gpu);
    rlt::free(device_cpu, b_cpu);
    rlt::free(device_gpu, b_gpu);
    rlt::free(device_cpu, c_cpu);
    rlt::free(device_gpu, c_gpu);
    rlt::free(device_cpu, c_gpu_cpu);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_MULTIPLY_BROADCAST_ACCUMULATE_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, HIDDEN_DIM>>> factor_cpu, factor_gpu, result_cpu, result_gpu, result_gpu_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, HIDDEN_DIM>>> broadcast_cpu, broadcast_gpu;
    rlt::malloc(device_cpu, factor_cpu);
    rlt::malloc(device_gpu, factor_gpu);
    rlt::malloc(device_cpu, result_cpu);
    rlt::malloc(device_gpu, result_gpu);
    rlt::malloc(device_cpu, result_gpu_cpu);
    rlt::malloc(device_cpu, broadcast_cpu);
    rlt::malloc(device_gpu, broadcast_gpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, factor_cpu, rng_cpu);
    rlt::randn(device_cpu, result_cpu, rng_cpu);
    rlt::randn(device_cpu, broadcast_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, factor_cpu, factor_gpu);
    rlt::copy(device_cpu, device_gpu, result_cpu, result_gpu);
    rlt::copy(device_cpu, device_gpu, broadcast_cpu, broadcast_gpu);
    rlt::multiply_broadcast_accumulate(device_cpu, factor_cpu, broadcast_cpu, result_cpu);
    rlt::multiply_broadcast_accumulate(device_gpu, factor_gpu, broadcast_gpu, result_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, result_gpu, result_gpu_cpu);
    T diff = rlt::abs_diff(device_cpu, result_cpu, result_gpu_cpu);
    std::cout << "multiply_broadcast_accumulate abs_diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);
    rlt::free(device_cpu, factor_cpu);
    rlt::free(device_gpu, factor_gpu);
    rlt::free(device_cpu, result_cpu);
    rlt::free(device_gpu, result_gpu);
    rlt::free(device_cpu, result_gpu_cpu);
    rlt::free(device_cpu, broadcast_cpu);
    rlt::free(device_gpu, broadcast_gpu);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, TENSOR_COPY_VIEW_THEN_ZERO_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3*HIDDEN_DIM>>> pa_cpu, pa_gpu, pa_gpu_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, HIDDEN_DIM>>> npp_cpu, npp_gpu;
    rlt::malloc(device_cpu, pa_cpu);
    rlt::malloc(device_gpu, pa_gpu);
    rlt::malloc(device_cpu, pa_gpu_cpu);
    rlt::malloc(device_cpu, npp_cpu);
    rlt::malloc(device_gpu, npp_gpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::randn(device_cpu, pa_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, pa_cpu, pa_gpu);

    auto n_cpu = rlt::view_range(device_cpu, pa_cpu, 2*(TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    auto n_gpu = rlt::view_range(device_gpu, pa_gpu, 2*(TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    rlt::copy(device_cpu, device_cpu, n_cpu, npp_cpu);
    rlt::copy(device_gpu, device_gpu, n_gpu, npp_gpu);
    rlt::set_all(device_cpu, n_cpu, (T)0);
    rlt::set_all(device_gpu, n_gpu, (T)0);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, pa_gpu, pa_gpu_cpu);
    T diff_pa = rlt::abs_diff(device_cpu, pa_cpu, pa_gpu_cpu);
    std::cout << "pa after copy+zero view: " << diff_pa << std::endl;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, HIDDEN_DIM>>> npp_gpu_cpu;
    rlt::malloc(device_cpu, npp_gpu_cpu);
    rlt::copy(device_gpu, device_cpu, npp_gpu, npp_gpu_cpu);
    T diff_npp = rlt::abs_diff(device_cpu, npp_cpu, npp_gpu_cpu);
    std::cout << "npp after copy: " << diff_npp << std::endl;

    EXPECT_LT(diff_pa, 1e-10);
    EXPECT_LT(diff_npp, 1e-10);

    rlt::free(device_cpu, pa_cpu);
    rlt::free(device_gpu, pa_gpu);
    rlt::free(device_cpu, pa_gpu_cpu);
    rlt::free(device_cpu, npp_cpu);
    rlt::free(device_gpu, npp_gpu);
    rlt::free(device_cpu, npp_gpu_cpu);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_MANUAL_STEP){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    using EVAL_CAP = rlt::nn::capability::Forward<>;
    using EVAL_GRU = GRU_TEMPLATE::Layer<EVAL_CAP, INPUT_SHAPE>;
    EVAL_GRU gru_cpu, gru_gpu;
    typename EVAL_GRU::template Buffer<> buf_cpu, buf_gpu;
    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, buf_cpu);
    rlt::malloc(device_gpu, buf_gpu);
    rlt::init_weights(device_cpu, gru_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, gru_cpu, gru_gpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>>> input_cpu, input_gpu;
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::randn(device_cpu, input_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, input_cpu, input_gpu);

    auto& pa_cpu = buf_cpu.post_activation;
    auto& pa_gpu = buf_gpu.post_activation;
    auto& npp_cpu = buf_cpu.n_pre_pre_activation;
    auto& npp_gpu = buf_gpu.n_pre_pre_activation;

    rlt::nn::layers::gru::helper::matrix_multiply_broadcast_transpose_bias(device_cpu, gru_cpu.weights_hidden.parameters, gru_cpu.initial_hidden_state.parameters, gru_cpu.biases_hidden.parameters, pa_cpu);
    rlt::nn::layers::gru::helper::matrix_multiply_broadcast_transpose_bias(device_gpu, gru_gpu.weights_hidden.parameters, gru_gpu.initial_hidden_state.parameters, gru_gpu.biases_hidden.parameters, pa_gpu);
    cudaDeviceSynchronize();
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3*HIDDEN_DIM>>> pa_gpu_cpu;
        rlt::malloc(device_cpu, pa_gpu_cpu);
        rlt::copy(device_gpu, device_cpu, pa_gpu, pa_gpu_cpu);
        T diff = rlt::abs_diff(device_cpu, pa_cpu, pa_gpu_cpu);
        std::cout << "after matrix_multiply_broadcast_transpose_bias: " << diff << std::endl;
        EXPECT_LT(diff, 1e-10);
        rlt::free(device_cpu, pa_gpu_cpu);
    }

    auto n_pa_cpu = rlt::view_range(device_cpu, pa_cpu, 2*(TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    auto n_pa_gpu = rlt::view_range(device_gpu, pa_gpu, 2*(TI)HIDDEN_DIM, rlt::tensor::ViewSpec<1, HIDDEN_DIM>{});
    rlt::copy(device_cpu, device_cpu, n_pa_cpu, npp_cpu);
    rlt::copy(device_gpu, device_gpu, n_pa_gpu, npp_gpu);
    rlt::set_all(device_cpu, n_pa_cpu, (T)0);
    rlt::set_all(device_gpu, n_pa_gpu, (T)0);
    cudaDeviceSynchronize();
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3*HIDDEN_DIM>>> pa_gpu_cpu;
        rlt::malloc(device_cpu, pa_gpu_cpu);
        rlt::copy(device_gpu, device_cpu, pa_gpu, pa_gpu_cpu);
        T diff = rlt::abs_diff(device_cpu, pa_cpu, pa_gpu_cpu);
        std::cout << "after copy+set_all: " << diff << std::endl;
        EXPECT_LT(diff, 1e-10);
        rlt::free(device_cpu, pa_gpu_cpu);
    }

    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_cpu, gru_cpu.weights_input.parameters, input_cpu, gru_cpu.biases_input.parameters, pa_cpu);
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_gpu, gru_gpu.weights_input.parameters, input_gpu, gru_gpu.biases_input.parameters, pa_gpu);
    cudaDeviceSynchronize();
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3*HIDDEN_DIM>>> pa_gpu_cpu;
        rlt::malloc(device_cpu, pa_gpu_cpu);
        rlt::copy(device_gpu, device_cpu, pa_gpu, pa_gpu_cpu);
        T diff = rlt::abs_diff(device_cpu, pa_cpu, pa_gpu_cpu);
        std::cout << "after matrix_multiply_transpose_bias_accumulate: " << diff << std::endl;
        EXPECT_LT(diff, 1e-10);
        rlt::free(device_cpu, pa_gpu_cpu);
    }

    auto rz_cpu = rlt::view_range(device_cpu, pa_cpu, (TI)0, rlt::tensor::ViewSpec<1, 2 * HIDDEN_DIM>{});
    auto rz_gpu = rlt::view_range(device_gpu, pa_gpu, (TI)0, rlt::tensor::ViewSpec<1, 2 * HIDDEN_DIM>{});
    rlt::fast_sigmoid(device_cpu, rz_cpu);
    rlt::fast_sigmoid(device_gpu, rz_gpu);
    cudaDeviceSynchronize();
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, 3*HIDDEN_DIM>>> pa_gpu_cpu;
        rlt::malloc(device_cpu, pa_gpu_cpu);
        rlt::copy(device_gpu, device_cpu, pa_gpu, pa_gpu_cpu);
        T diff = rlt::abs_diff(device_cpu, pa_cpu, pa_gpu_cpu);
        std::cout << "after fast_sigmoid on rz view: " << diff << std::endl;
        EXPECT_LT(diff, 1e-10);
        rlt::free(device_cpu, pa_gpu_cpu);
    }

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, buf_cpu);
    rlt::free(device_gpu, buf_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_EVALUATE_SEQ1){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    using EVAL_CAP = rlt::nn::capability::Forward<>;
    using EVAL_GRU = GRU_TEMPLATE::Layer<EVAL_CAP, INPUT_SHAPE>;
    EVAL_GRU gru_cpu, gru_gpu, gru_gpu_cpu;
    typename EVAL_GRU::template Buffer<> buffer_cpu, buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename EVAL_GRU::INPUT_SHAPE>> input_cpu, input_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename EVAL_GRU::OUTPUT_SHAPE>> output_cpu, output_gpu, output_gpu_cpu;

    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_gpu_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::malloc(device_gpu, buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_gpu, output_gpu);
    rlt::malloc(device_cpu, output_gpu_cpu);

    rlt::init_weights(device_cpu, gru_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, gru_cpu, gru_gpu);
    rlt::randn(device_cpu, input_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, input_cpu, input_gpu);

    rlt::evaluate(device_cpu, gru_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    rlt::evaluate(device_gpu, gru_gpu, input_gpu, output_gpu, buffer_gpu, rng_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device_cpu, output_gpu, output_gpu_cpu);

    T diff = rlt::abs_diff(device_cpu, output_cpu, output_gpu_cpu);
    std::cout << "evaluate seq1 output abs_diff: " << diff << std::endl;
    rlt::print(device_cpu, output_cpu);
    rlt::print(device_cpu, output_gpu_cpu);
    ASSERT_LT(diff, 1e-6);

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, gru_gpu_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_gpu, buffer_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_gpu, output_gpu);
    rlt::free(device_cpu, output_gpu_cpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_FORWARD_SEQ1){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU gru_cpu, gru_gpu, gru_gpu_cpu;
    GRU::Buffer<> gru_buffer_cpu, gru_buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU::INPUT_SHAPE>> input_cpu, input_gpu;

    rlt::init(device_cpu);
    rlt::init(device_gpu);

    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_gpu_cpu);
    rlt::malloc(device_cpu, gru_buffer_cpu);
    rlt::malloc(device_gpu, gru_buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);

    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    rlt::init_weights(device_cpu, gru_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, gru_cpu, gru_gpu);
    rlt::randn(device_cpu, input_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, input_cpu, input_gpu);

    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_gpu_cpu);
    T abs_diff_weights = rlt::abs_diff(device_cpu, gru_cpu.weights_input.parameters, gru_gpu_cpu.weights_input.parameters);
    std::cout << "weights_input after copy abs_diff: " << abs_diff_weights << std::endl;
    ASSERT_LT(abs_diff_weights, 1e-10);

    rlt::forward(device_cpu, gru_cpu, input_cpu, gru_buffer_cpu, rng_cpu);
    rlt::forward(device_gpu, gru_gpu, input_gpu, gru_buffer_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_gpu_cpu);

    T abs_diff_output = rlt::abs_diff(device_cpu, gru_cpu.output, gru_gpu_cpu.output);
    std::cout << "forward seq1 output abs_diff: " << abs_diff_output << std::endl;
    ASSERT_LT(abs_diff_output, 1e-6);

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, gru_gpu_cpu);
    rlt::free(device_cpu, gru_buffer_cpu);
    rlt::free(device_gpu, gru_buffer_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
}

template <TI T_SEQUENCE_LENGTH, TI T_BATCH_SIZE, TI T_INPUT_DIM, TI T_HIDDEN_DIM>
void test_gru_cuda_forward(){
    using GRU_CFG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, T_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
    using GRU_BIND = rlt::nn::layers::gru::BindConfiguration<GRU_CFG>;
    using CAP = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using SHAPE = rlt::tensor::Shape<TI, T_SEQUENCE_LENGTH, T_BATCH_SIZE, T_INPUT_DIM>;
    using GRU_TYPE = typename GRU_BIND::template Layer<CAP, SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU_TYPE gru_cpu, gru_gpu, gru_gpu_cpu;
    typename GRU_TYPE::template Buffer<> buffer_cpu, buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> input_cpu, input_gpu;

    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_gpu_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::malloc(device_gpu, buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    rlt::init_weights(device_gpu, gru_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_cpu);
    rlt::randn(device_gpu, input_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, input_gpu, input_cpu);

    rlt::forward(device_cpu, gru_cpu, input_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_gpu, gru_gpu, input_gpu, buffer_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_gpu_cpu);

    T abs_diff_output = rlt::abs_diff(device_cpu, gru_cpu.output, gru_gpu_cpu.output);
    std::cout << "forward<" << T_SEQUENCE_LENGTH << "," << T_BATCH_SIZE << "," << T_INPUT_DIM << "," << T_HIDDEN_DIM << "> output abs_diff: " << abs_diff_output << std::endl;
    EXPECT_LT(abs_diff_output, 1e-5);

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, gru_gpu_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_gpu, buffer_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_FORWARD_MULTI_SEQ){
    test_gru_cuda_forward<1, 3, 4, 5>();
    test_gru_cuda_forward<4, 3, 4, 5>();
    test_gru_cuda_forward<16, 8, 4, 5>();
    test_gru_cuda_forward<16, 32, 19, 64>();
}

template <TI T_SEQUENCE_LENGTH, TI T_BATCH_SIZE, TI T_INPUT_DIM, TI T_HIDDEN_DIM>
void test_gru_cuda_backward(){
    using GRU_CFG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, T_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
    using GRU_BIND = rlt::nn::layers::gru::BindConfiguration<GRU_CFG>;
    using CAP = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using SHAPE = rlt::tensor::Shape<TI, T_SEQUENCE_LENGTH, T_BATCH_SIZE, T_INPUT_DIM>;
    using GRU_TYPE = typename GRU_BIND::template Layer<CAP, SHAPE>;
    using OUTPUT_SHAPE = typename GRU_TYPE::OUTPUT_SHAPE;

    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU_TYPE gru_cpu, gru_gpu, gru_gpu_cpu;
    typename GRU_TYPE::template Buffer<> buffer_cpu, buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> input_cpu, input_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output_cpu, d_output_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> d_input_cpu, d_input_gpu, d_input_gpu_cpu;

    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_gpu_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::malloc(device_gpu, buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_gpu, d_output_gpu);
    rlt::malloc(device_cpu, d_input_cpu);
    rlt::malloc(device_gpu, d_input_gpu);
    rlt::malloc(device_cpu, d_input_gpu_cpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    rlt::init_weights(device_gpu, gru_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_cpu);
    rlt::randn(device_gpu, input_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, input_gpu, input_cpu);
    rlt::randn(device_gpu, d_output_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, d_output_gpu, d_output_cpu);

    rlt::forward(device_cpu, gru_cpu, input_cpu, buffer_cpu, rng_cpu);
    rlt::forward(device_gpu, gru_gpu, input_gpu, buffer_gpu, rng_gpu);

    rlt::zero_gradient(device_cpu, gru_cpu);
    rlt::zero_gradient(device_gpu, gru_gpu);
    rlt::backward_full(device_cpu, gru_cpu, input_cpu, d_output_cpu, d_input_cpu, buffer_cpu);
    rlt::backward_full(device_gpu, gru_gpu, input_gpu, d_output_gpu, d_input_gpu, buffer_gpu);

    rlt::copy(device_gpu, device_cpu, d_input_gpu, d_input_gpu_cpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_gpu_cpu);

    T abs_diff_d_input = rlt::abs_diff(device_cpu, d_input_cpu, d_input_gpu_cpu);
    std::cout << "backward<" << T_SEQUENCE_LENGTH << "," << T_BATCH_SIZE << "," << T_INPUT_DIM << "," << T_HIDDEN_DIM << "> d_input abs_diff: " << abs_diff_d_input << std::endl;
    EXPECT_LT(abs_diff_d_input, 1e-5);

    T abs_diff_weights_input = rlt::abs_diff(device_cpu, gru_cpu.weights_input.gradient, gru_gpu_cpu.weights_input.gradient);
    std::cout << "backward<" << T_SEQUENCE_LENGTH << "," << T_BATCH_SIZE << "," << T_INPUT_DIM << "," << T_HIDDEN_DIM << "> weights_input gradient abs_diff: " << abs_diff_weights_input << std::endl;
    EXPECT_LT(abs_diff_weights_input, 1e-5);

    T abs_diff_weights_hidden = rlt::abs_diff(device_cpu, gru_cpu.weights_hidden.gradient, gru_gpu_cpu.weights_hidden.gradient);
    std::cout << "backward<" << T_SEQUENCE_LENGTH << "," << T_BATCH_SIZE << "," << T_INPUT_DIM << "," << T_HIDDEN_DIM << "> weights_hidden gradient abs_diff: " << abs_diff_weights_hidden << std::endl;
    EXPECT_LT(abs_diff_weights_hidden, 1e-5);

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, gru_gpu_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_gpu, buffer_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
    rlt::free(device_cpu, d_output_cpu);
    rlt::free(device_gpu, d_output_gpu);
    rlt::free(device_cpu, d_input_cpu);
    rlt::free(device_gpu, d_input_gpu);
    rlt::free(device_cpu, d_input_gpu_cpu);
}

template <TI T_SEQUENCE_LENGTH, TI T_BATCH_SIZE, TI T_INPUT_DIM, TI T_HIDDEN_DIM>
void test_gru_cuda_forward_reset_mode(){
    using GRU_CFG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, T_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
    using GRU_BIND = rlt::nn::layers::gru::BindConfiguration<GRU_CFG>;
    using CAP = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using SHAPE = rlt::tensor::Shape<TI, T_SEQUENCE_LENGTH, T_BATCH_SIZE, T_INPUT_DIM>;
    using GRU_TYPE = typename GRU_BIND::template Layer<CAP, SHAPE>;

    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    rlt::init(device_cpu);
    rlt::init(device_gpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    GRU_TYPE gru_cpu, gru_gpu, gru_gpu_cpu;
    typename GRU_TYPE::template Buffer<> buffer_cpu, buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> input_cpu, input_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, T_SEQUENCE_LENGTH, T_BATCH_SIZE, 1>>> reset_cpu, reset_gpu;

    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_gpu_cpu);
    rlt::malloc(device_cpu, buffer_cpu);
    rlt::malloc(device_gpu, buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::malloc(device_cpu, reset_cpu);
    rlt::malloc(device_gpu, reset_gpu);

    rlt::init_weights(device_cpu, gru_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, gru_cpu, gru_gpu);
    rlt::randn(device_cpu, input_cpu, rng_cpu);
    rlt::copy(device_cpu, device_gpu, input_cpu, input_gpu);

    rlt::set_all(device_cpu, reset_cpu, (T)0);
    for(TI s = 0; s < T_SEQUENCE_LENGTH; s++){
        for(TI b = 0; b < T_BATCH_SIZE; b++){
            if((s * T_BATCH_SIZE + b) % 3 == 0){
                rlt::set(device_cpu, reset_cpu, (T)1, s, b, (TI)0);
            }
        }
    }

    rlt::copy(device_cpu, device_gpu, reset_cpu, reset_gpu);
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, T_SEQUENCE_LENGTH, T_BATCH_SIZE, 1>>> reset_gpu_cpu;
        rlt::malloc(device_cpu, reset_gpu_cpu);
        rlt::copy(device_gpu, device_cpu, reset_gpu, reset_gpu_cpu);
        T reset_diff = rlt::abs_diff(device_cpu, reset_cpu, reset_gpu_cpu);
        std::cout << "reset container abs_diff: " << reset_diff << std::endl;
        EXPECT_LT(reset_diff, 1e-10);
        rlt::free(device_cpu, reset_gpu_cpu);
    }

    using RESET_MODE_CPU_SPEC = rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(reset_cpu)>;
    using RESET_MODE_CPU = rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, RESET_MODE_CPU_SPEC>;
    rlt::Mode<RESET_MODE_CPU> reset_mode_cpu;
    reset_mode_cpu.reset_container = reset_cpu;

    using RESET_MODE_GPU_SPEC = rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(reset_gpu)>;
    using RESET_MODE_GPU = rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, RESET_MODE_GPU_SPEC>;
    rlt::Mode<RESET_MODE_GPU> reset_mode_gpu;
    reset_mode_gpu.reset_container = reset_gpu;

    rlt::forward(device_cpu, gru_cpu, input_cpu, buffer_cpu, rng_cpu, reset_mode_cpu);
    rlt::forward(device_gpu, gru_gpu, input_gpu, buffer_gpu, rng_gpu, reset_mode_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_gpu_cpu);

    T abs_diff_output = rlt::abs_diff(device_cpu, gru_cpu.output, gru_gpu_cpu.output);
    std::cout << "forward_reset<" << T_SEQUENCE_LENGTH << "," << T_BATCH_SIZE << "," << T_INPUT_DIM << "," << T_HIDDEN_DIM << "> output abs_diff: " << abs_diff_output << std::endl;
    EXPECT_LT(abs_diff_output, 1e-5);

    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_cpu, gru_cpu);
    rlt::free(device_gpu, gru_gpu);
    rlt::free(device_cpu, gru_gpu_cpu);
    rlt::free(device_cpu, buffer_cpu);
    rlt::free(device_gpu, buffer_gpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_gpu, input_gpu);
    rlt::free(device_cpu, reset_cpu);
    rlt::free(device_gpu, reset_gpu);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_FORWARD_RESET_MODE){
    test_gru_cuda_forward_reset_mode<2, 3, 4, 5>();
    test_gru_cuda_forward_reset_mode<4, 3, 4, 5>();
    test_gru_cuda_forward_reset_mode<16, 8, 4, 5>();
    test_gru_cuda_forward_reset_mode<16, 32, 19, 64>();
}

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_BACKWARD){
    test_gru_cuda_backward<1, 3, 4, 5>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA_BACKWARD_MULTI_SEQ){
    test_gru_cuda_backward<4, 3, 4, 5>();
    test_gru_cuda_backward<16, 8, 4, 5>();
    test_gru_cuda_backward<16, 32, 19, 64>();
}