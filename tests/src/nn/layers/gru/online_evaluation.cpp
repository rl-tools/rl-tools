#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = double;
using TI = typename DEVICE::index_t;
constexpr TI SEQUENCE_LENGTH = 16;
constexpr TI BATCH_SIZE = 32;
constexpr TI INPUT_DIM = 8;
constexpr TI HIDDEN_DIM = 24;

#include <gtest/gtest.h>

TEST(RL_TOOLS_NN_LAYERS_GRU, ONLINE_EVALUATION){
//    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using CAPABILITY = rlt::nn::layer_capability::Forward;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM>;
    using GRU = rlt::nn::layers::gru::Layer<CAPABILITY, GRU_SPEC>;

    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    GRU gru;
    GRU::Buffer<BATCH_SIZE> buffer;
    rlt::nn::Mode<rlt::nn::layers::gru::StepByStepMode> mode, mode_reset;
    mode.reset = false;
    mode_reset.reset = true;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, HIDDEN_DIM>>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, gru);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, gru, rng);
    rlt::randn(device, input, rng);

    std::cout << "Post activation: " << std::endl;
    rlt::print(device, decltype(buffer.post_activation)::SPEC::SHAPE{});
    std::cout << std::endl;

    rlt::evaluate(device, gru, input, output, buffer, rng, mode_reset);
}

