#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/random/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "../../../../../data/test_nn_layers_gru_persist_code.h"

namespace rlt = rl_tools;

#include <gtest/gtest.h>


using T = double;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using TI = typename DEVICE_GPU::index_t;

constexpr TI SEQUENCE_LENGTH = 10;
constexpr TI BATCH_SIZE = 3;
constexpr TI INPUT_DIM = 4;
constexpr TI HIDDEN_DIM = 5;

using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal, true>;
using GRU_TEMPLATE = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
using GRU = GRU_TEMPLATE::Layer<CAPABILITY, INPUT_SHAPE>;

TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_HELPERS_CUDA){
    constexpr T EPSILON = 1e-10;
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU gru_cpu, gru_gpu;
    GRU::Buffer<> gru_buffer_cpu, gru_buffer_gpu;
    constexpr TI M = 30;
    constexpr TI N = 40;
    constexpr TI K = 50;
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
            ASSERT_LT(abs_diff_generic_manual, EPSILON);
        }
        else{
            std::cout << "abs_diff_generic_manual_accumulate: " << abs_diff_generic_manual << std::endl;
            ASSERT_LT(abs_diff_generic_manual, EPSILON);
        }
    }
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias(device_cpu, B_T_cpu, A_cpu, bias_cpu, C_cpu);

    T abs_diff_helper_generic = rlt::abs_diff(device_cpu, C_cpu, C_cpu_generic);
    std::cout << "abs_diff_helper_generic: " << abs_diff_helper_generic << std::endl;
    ASSERT_LT(abs_diff_helper_generic, EPSILON);

    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias(device_gpu, B_T_gpu, A_gpu, bias_gpu, C_gpu);
    rlt::copy(device_gpu, device_cpu, C_gpu, C_gpu_cpu);
    T abs_diff_helper_gpu = rlt::abs_diff(device_cpu, C_cpu, C_gpu_cpu);
    std::cout << "abs_diff_helper_gpu: " << abs_diff_helper_gpu << std::endl;
    ASSERT_LT(abs_diff_helper_gpu, EPSILON);


    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_cpu, B_T_cpu, A_cpu, bias_cpu, C_cpu);
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias_accumulate(device_gpu, B_T_gpu, A_gpu, bias_gpu, C_gpu);
    rlt::copy(device_gpu, device_cpu, C_gpu, C_gpu_cpu);
    T abs_diff_helper_gpu_accumulate = rlt::abs_diff(device_cpu, C_cpu, C_gpu_cpu);
    std::cout << "abs_diff_helper_gpu: " << abs_diff_helper_gpu_accumulate << std::endl;
    ASSERT_LT(abs_diff_helper_gpu_accumulate, EPSILON);
}

// TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA){
//     DEVICE_CPU device_cpu;
//     DEVICE_GPU device_gpu;
//     DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
//     DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
//     GRU gru_cpu, gru_gpu;
//     GRU::Buffer<> gru_buffer_cpu, gru_buffer_gpu;
//     rlt::Tensor<rlt::tensor::Specification<T, TI, GRU::INPUT_SHAPE>> input_cpu, input_gpu;
//     rlt::Tensor<rlt::tensor::Specification<T, TI, GRU::OUTPUT_SHAPE>> output_cpu, output_gpu;
//
//     rlt::init(device_cpu);
//     rlt::init(device_gpu);
//
//     rlt::malloc(device_cpu, rng_cpu);
//     rlt::malloc(device_gpu, rng_gpu);
//     rlt::malloc(device_cpu, gru_cpu);
//     rlt::malloc(device_gpu, gru_gpu);
//     rlt::malloc(device_cpu, gru_buffer_cpu);
//     rlt::malloc(device_gpu, gru_buffer_gpu);
//     rlt::malloc(device_cpu, input_cpu);
//     rlt::malloc(device_gpu, input_gpu);
//     rlt::malloc(device_cpu, output_cpu);
//     rlt::malloc(device_gpu, output_gpu);
//
//     rlt::init(device_cpu, rng_cpu, 0);
//     rlt::init(device_gpu, rng_gpu, 0);
//
//     rlt::init_weights(device_gpu, gru_gpu, rng_gpu);
//     rlt::copy(device_gpu, device_cpu, gru_gpu, gru_cpu);
//     rlt::randn(device_gpu, input_gpu, rng_gpu);
//     rlt::copy(device_gpu, device_cpu, input_gpu, input_cpu);
//
//     rlt::evaluate(device_cpu, gru_cpu, input_cpu, output_cpu, gru_buffer_cpu, rng_cpu);
//     rlt::evaluate(device_gpu, gru_gpu, input_gpu, output_gpu, gru_buffer_gpu, rng_gpu);
//
//     rlt::print(device_cpu, output_cpu);
// }
