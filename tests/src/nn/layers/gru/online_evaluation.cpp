#ifdef MUX
#include <rl_tools/operations/cpu_mux.h>
#else
#include <rl_tools/operations/cpu.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = rl_tools;

#ifdef MUX
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE = rlt::devices::DefaultCPU;
#endif
using T = double;
using TI = typename DEVICE::index_t;
constexpr TI SEQUENCE_LENGTH = 16;
constexpr TI BATCH_SIZE = 32;
constexpr TI INPUT_DIM = 8;
constexpr TI HIDDEN_DIM = 24;

#ifndef DISABLE_TEST
#include <gtest/gtest.h>
#endif

#ifndef DISABLE_TEST
TEST(RL_TOOLS_NN_LAYERS_GRU, ONLINE_EVALUATION){
#else
int main(){
#endif
//    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using CAPABILITY = rlt::nn::layer_capability::Forward;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM>;
    using GRU = rlt::nn::layers::gru::Layer<CAPABILITY, GRU_SPEC>;

    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    GRU gru;
    GRU::Buffer<BATCH_SIZE> buffer;
    rlt::nn::Mode<rlt::nn::layers::gru::StepByStepMode> mode_no_reset, mode_reset;
    mode_no_reset.reset = false;
    mode_reset.reset = true;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, HIDDEN_DIM>>> output_1, output_2;

    rlt::malloc(device, input);
    rlt::malloc(device, output_1);
    rlt::malloc(device, output_2);
    rlt::malloc(device, gru);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, gru, rng);
    rlt::randn(device, input, rng);

    std::cout << "Post activation: " << std::endl;
    rlt::print(device, decltype(buffer.post_activation)::SPEC::SHAPE{});
    std::cout << std::endl;

    rlt::evaluate(device, gru, input, output_1, buffer, rng, mode_reset);
    rlt::evaluate(device, gru, input, output_2, buffer, rng, mode_reset);
    T diff_reset = rlt::abs_diff(device, output_1, output_2);
    std::cout << "Diff reset: " << diff_reset << std::endl;
#ifndef DISABLE_TEST
    ASSERT_EQ(0, diff_reset);
#endif
    rlt::evaluate(device, gru, input, output_2, buffer, rng, mode_no_reset);
    T diff_no_reset = rlt::abs_diff(device, output_1, output_2);
    std::cout << "Diff no reset: " << diff_no_reset << std::endl;
#ifndef DISABLE_TEST
    ASSERT_GT(diff_no_reset, 0);
#endif
}

