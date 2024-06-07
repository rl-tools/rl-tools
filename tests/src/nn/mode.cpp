#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/parameters/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <gtest/gtest.h>

namespace rlt = rl_tools;


TEST(RL_TOOLS_NN_MODE, LAYER) {
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = DEVICE::index_t;
    constexpr TI INPUT_DIM = 10;
    constexpr TI OUTPUT_DIM = 5;
    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
    using PARAMETER_TYPE = rlt::nn::parameters::Plain;
    using LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>;
    rlt::nn::layers::dense::Layer<rlt::nn::layer_capability::Forward, LAYER_SPEC> layer;
    decltype(layer)::template Buffer<1> buffer;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, INPUT_DIM>> input;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> output;

    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{});
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);
    rlt::init_kaiming(device, layer, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, layer, input, output, buffer, rng, rlt::nn::Mode<rlt::nn::mode::Default>{});
}
