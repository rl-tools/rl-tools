#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/parameters/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash//operations_generic.h>
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
    using LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, OUTPUT_DIM, ACTIVATION_FUNCTION>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, INPUT_DIM>;
    rlt::nn::layers::dense::Layer<LAYER_CONFIG, rlt::nn::layer_capability::Forward<>, INPUT_SHAPE> layer;
    decltype(layer)::template Buffer<1> buffer;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, INPUT_DIM>> input;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> output;

    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{});
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, layer, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, layer, input, output, buffer, rng, rlt::Mode<rlt::mode::Default<>>{});
}

template <typename BASE = bool, typename SPEC = bool>
struct Default{};

template <typename BASE = rlt::mode::Default<>>
using SAS = rlt::nn::layers::sample_and_squash::mode::ExternalNoise<BASE>;
TEST(RL_TOOLS_NN_MODE, INHERITANCE_CHAIN) {
    using DEFAULT = rlt::mode::Default<>;
    DEFAULT mode;
    SAS<> ext_noise;
    SAS<rlt::mode::Inference<DEFAULT>> inf;
    rlt::nn::layers::sample_and_squash::mode::Sample<SAS<DEFAULT>> sample;
    static_assert(!rlt::mode::is<decltype(ext_noise), rlt::nn::layers::sample_and_squash::mode::Sample>);
    static_assert(rlt::mode::is<decltype(ext_noise), rlt::nn::layers::sample_and_squash::mode::ExternalNoise>);
    static_assert(rlt::mode::is<decltype(sample), rlt::nn::layers::sample_and_squash::mode::ExternalNoise>);
    static_assert(rlt::mode::is<decltype(sample), rlt::nn::layers::sample_and_squash::mode::Sample>);
    static_assert(!rlt::mode::is<decltype(sample), rlt::mode::Inference>);
}
