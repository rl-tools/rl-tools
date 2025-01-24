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


using T = float;
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


TEST(RL_TOOLS_NN_LAYERS_GRU, GRU_CUDA){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    DEVICE_GPU::SPEC::RANDOM::ENGINE<> rng_gpu;
    GRU gru_cpu, gru_gpu;
    GRU::Buffer<> gru_buffer_cpu, gru_buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU::INPUT_SHAPE>> input_cpu, input_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU::OUTPUT_SHAPE>> output_cpu, output_gpu;

    rlt::init(device_cpu);
    rlt::init(device_gpu);

    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::malloc(device_cpu, gru_cpu);
    rlt::malloc(device_gpu, gru_gpu);
    rlt::malloc(device_cpu, gru_buffer_cpu);
    rlt::malloc(device_gpu, gru_buffer_gpu);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_gpu, input_gpu);
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_gpu, output_gpu);

    rlt::init(device_cpu, rng_cpu, 0);
    rlt::init(device_gpu, rng_gpu, 0);

    rlt::init_weights(device_gpu, gru_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, gru_gpu, gru_cpu);
    rlt::randn(device_gpu, input_gpu, rng_gpu);
    rlt::copy(device_gpu, device_cpu, input_gpu, input_cpu);

    rlt::evaluate(device_cpu, gru_cpu, input_cpu, output_cpu, gru_buffer_cpu, rng_cpu);
    // rlt::evaluate(device_gpu, gru_gpu, input_gpu, output_gpu, gru_buffer_gpu, rng_gpu);

    rlt::print(device_cpu, output_cpu);
}
