#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#ifdef RL_TOOLS_TEST_GRU_CUDA_HELPER_FIRST
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#endif
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>

#include <gtest/gtest.h>
#include <metra/metra.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using CPU = rlt::devices::DefaultCPU;
using GPU = rlt::devices::DefaultCUDA;
using TI = GPU::index_t;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

template <typename DEVICE, typename SPEC>
auto reset_mask_view(DEVICE&, rlt::Matrix<SPEC>& mask){
    return mask;
}
template <typename DEVICE, typename SPEC>
auto reset_mask_view(DEVICE& device, rlt::Tensor<SPEC>& mask){
    return rlt::matrix_view(device, mask);
}

template <typename MASK>
void test_masked_reset(){
    CPU cpu;
    GPU gpu;
    MASK mask_cpu, mask_gpu;
    using MASK_VIEW = decltype(reset_mask_view(cpu, mask_cpu));
    constexpr TI BATCH_SIZE = MASK_VIEW::COLS;
    constexpr TI HIDDEN_DIM = 5;
    using CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM>;
    using LAYER = rlt::nn::layers::gru::Layer<CONFIG, rlt::nn::capability::Forward<>, rlt::tensor::Shape<TI, 16, BATCH_SIZE, 2>>;
    rlt::init(gpu);
    LAYER layer_cpu, layer_gpu;
    typename LAYER::template State<> state_cpu, state_gpu, actual;
    typename CPU::SPEC::RANDOM::template ENGINE<> rng_cpu;
    typename GPU::SPEC::RANDOM::template ENGINE<> rng_gpu;
    rlt::malloc(cpu, rng_cpu);
    rlt::init(cpu, rng_cpu, 42);
    rlt::malloc(gpu, rng_gpu);
    rlt::init(gpu, rng_gpu, 42);
    rlt::malloc(cpu, layer_cpu);
    rlt::malloc(gpu, layer_gpu);
    rlt::malloc(cpu, state_cpu);
    rlt::malloc(gpu, state_gpu);
    rlt::malloc(cpu, actual);
    rlt::malloc(cpu, mask_cpu);
    rlt::malloc(gpu, mask_gpu);
    for(TI hidden_i = 0; hidden_i < HIDDEN_DIM; hidden_i++){
        rlt::set(cpu, layer_cpu.initial_hidden_state.parameters, (T)hidden_i / 4 + 2, hidden_i);
    }
    rlt::copy(cpu, gpu, layer_cpu.initial_hidden_state.parameters, layer_gpu.initial_hidden_state.parameters);
    using MASK_SPEC = rlt::mode::sequential::ResetMaskSpecification<MASK_VIEW>;
    using MODE = rlt::Mode<rlt::mode::sequential::ResetMask<rlt::mode::Default<>, MASK_SPEC>>;
    MODE mode_cpu, mode_gpu;
    mode_cpu.mask = reset_mask_view(cpu, mask_cpu);
    mode_gpu.mask = reset_mask_view(gpu, mask_gpu);
    T max_error = 0;
    for(TI pattern = 0; pattern < 3; pattern++){
        for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
            const bool selected = pattern == 1 || (pattern == 2 && batch_i % 2 == 0);
            rlt::set(mode_cpu.mask, 0, batch_i, selected);
            rlt::set(cpu, state_cpu.step, batch_i + 4, batch_i);
            for(TI hidden_i = 0; hidden_i < HIDDEN_DIM; hidden_i++){
                rlt::set(cpu, state_cpu.state, (T)(64 + batch_i * HIDDEN_DIM + hidden_i), batch_i, hidden_i);
            }
        }
        rlt::copy(cpu, gpu, mask_cpu, mask_gpu);
        rlt::copy(cpu, gpu, state_cpu, state_gpu);
        for(TI repeat = 0; repeat < 2; repeat++){
            rlt::reset(cpu, layer_cpu, state_cpu, rng_cpu, mode_cpu);
            rlt::reset(gpu, layer_gpu, state_gpu, rng_gpu, mode_gpu);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(gpu.stream), cudaSuccess);
            rlt::copy(gpu, cpu, state_gpu, actual);
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                const bool selected = pattern == 1 || (pattern == 2 && batch_i % 2 == 0);
                const TI expected_step = selected ? 0 : batch_i + 4;
                EXPECT_EQ(rlt::get(cpu, actual.step, batch_i), expected_step);
                EXPECT_EQ(rlt::get(cpu, state_cpu.step, batch_i), expected_step);
                for(TI hidden_i = 0; hidden_i < HIDDEN_DIM; hidden_i++){
                    const T expected = selected ? (T)hidden_i / 4 + 2 : (T)(64 + batch_i * HIDDEN_DIM + hidden_i);
                    const T observed = rlt::get(cpu, actual.state, batch_i, hidden_i);
                    EXPECT_EQ(observed, expected);
                    EXPECT_EQ(rlt::get(cpu, state_cpu.state, batch_i, hidden_i), expected);
                    max_error = rlt::math::max(cpu.math, max_error, rlt::math::abs(cpu.math, observed - expected));
                }
            }
        }
    }
    metra::log("nn/layers/gru/reset_cuda/max_abs_error", (double)max_error);
    rlt::free(cpu, mask_cpu);
    rlt::free(gpu, mask_gpu);
    rlt::free(cpu, actual);
    rlt::free(cpu, state_cpu);
    rlt::free(gpu, state_gpu);
    rlt::free(cpu, layer_cpu);
    rlt::free(gpu, layer_gpu);
    rlt::free(cpu, rng_cpu);
    rlt::free(gpu, rng_gpu);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
    EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
}

TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, MATRIX_SINGLE){
    test_masked_reset<rlt::Matrix<rlt::matrix::Specification<bool, TI, 1, 1>>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, MATRIX_MULTIBLOCK){
    test_masked_reset<rlt::Matrix<rlt::matrix::Specification<bool, TI, 1, 37>>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, MATRIX_STRIDED){
    test_masked_reset<rlt::Matrix<rlt::matrix::Specification<bool, TI, 1, 37, true, rlt::matrix::layouts::Fixed<TI, 74, 2>>>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, MATRIX_FLOAT){
    test_masked_reset<rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 37>>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, TENSOR_VIEW_SINGLE){
    test_masked_reset<rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, 1>>>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, TENSOR_VIEW_MULTIBLOCK){
    test_masked_reset<rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, 37>>>>();
}

template <typename MODE>
void test_step_counters(){
    constexpr TI N = 37, H = 5, SEQUENCE = 16;
    using CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, H>;
    using LAYER = rlt::nn::layers::gru::Layer<CONFIG, rlt::nn::capability::Forward<>, rlt::tensor::Shape<TI, SEQUENCE, N, 2>>;
    constexpr bool NO_AUTO_RESET = rlt::mode::is<MODE, rlt::nn::layers::gru::NoAutoResetMode>;
    CPU cpu;
    GPU gpu;
    rlt::init(gpu);
    LAYER layer_cpu, layer_gpu;
    typename LAYER::template State<> state_cpu, state_gpu, actual;
    rlt::malloc(cpu, layer_cpu); rlt::malloc(gpu, layer_gpu);
    rlt::malloc(cpu, state_cpu); rlt::malloc(gpu, state_gpu); rlt::malloc(cpu, actual);
    for(TI h = 0; h < H; h++) rlt::set(cpu, layer_cpu.initial_hidden_state.parameters, (T)(h + 2), h);
    rlt::copy(cpu, gpu, layer_cpu.initial_hidden_state.parameters, layer_gpu.initial_hidden_state.parameters);
    rlt::Mode<MODE> mode;
    for(TI operation = 0; operation < 3; operation++){
        for(TI i = 0; i < N; i++){
            rlt::set(cpu, state_cpu.step, i % 4 == 0 ? 0 : SEQUENCE + i % 4 - 2, i);
            for(TI h = 0; h < H; h++) rlt::set(cpu, state_cpu.state, (T)(100 + i * H + h), i, h);
        }
        rlt::copy(cpu, gpu, state_cpu, state_gpu);
        if(operation == 0){
            rlt::reset_truncate(cpu, layer_cpu, state_cpu, mode);
            rlt::reset_truncate(gpu, layer_gpu, state_gpu, mode);
        }
        else if(operation == 1){
            rlt::advance_gru_step<N>(cpu, layer_cpu, state_cpu, mode);
            rlt::advance_gru_step<N>(gpu, layer_gpu, state_gpu, mode);
        }
        else{
            rlt::advance_gru_step<13>(cpu, layer_cpu, state_cpu, mode);
            rlt::advance_gru_step<13>(gpu, layer_gpu, state_gpu, mode);
        }
        ASSERT_EQ(cudaStreamSynchronize(gpu.stream), cudaSuccess);
        rlt::copy(gpu, cpu, state_gpu, actual);
        for(TI i = 0; i < N; i++){
            TI expected_step = i % 4 == 0 ? 0 : SEQUENCE + i % 4 - 2;
            const bool selected = operation != 2 || i < 13 || NO_AUTO_RESET;
            if(operation != 0 && selected) expected_step++;
            const bool reset = !NO_AUTO_RESET && selected && expected_step >= SEQUENCE;
            if(reset) expected_step = 0;
            EXPECT_EQ(rlt::get(cpu, actual.step, i), expected_step);
            EXPECT_EQ(rlt::get(cpu, state_cpu.step, i), expected_step);
            for(TI h = 0; h < H; h++){
                const T expected = reset ? (T)(h + 2) : (T)(100 + i * H + h);
                EXPECT_EQ(rlt::get(cpu, actual.state, i, h), expected);
                EXPECT_EQ(rlt::get(cpu, state_cpu.state, i, h), expected);
            }
        }
    }
    rlt::free(cpu, layer_cpu); rlt::free(gpu, layer_gpu);
    rlt::free(cpu, state_cpu); rlt::free(gpu, state_gpu); rlt::free(cpu, actual);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
    EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
    metra::log("nn/layers/gru/step_cuda/mismatches", ::testing::Test::HasFailure() ? 1.0 : 0.0);
}

TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, STEP_COUNTERS){
    test_step_counters<rlt::mode::Default<>>();
}
TEST(RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA, STEP_COUNTERS_NO_AUTO_RESET){
    test_step_counters<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>();
}
