// Group 1
#include <backprop_tools/devices/cpu.h>
#include <backprop_tools/math/operations_cpu.h>
#include <backprop_tools/random/operations_cpu.h>
#include <backprop_tools/logging/operations_cpu.h>
#include <backprop_tools/devices/dummy.h>
#include <backprop_tools/math/operations_dummy.h>
#include <backprop_tools/random/operations_dummy.h>
#include <backprop_tools/logging/operations_dummy.h>

// Group 2: depends on logging
#include <backprop_tools/utils/assert/operations_cpu.h>
#include <backprop_tools/utils/assert/operations_dummy.h>
// Group 3: dependent on assert
#include <backprop_tools/containers/operations_cpu.h>
#include <backprop_tools/containers/operations_dummy.h>

#include <backprop_tools/nn/operations_cpu.h>
#include <backprop_tools/utils/generic/typing.h>

#include <gtest/gtest.h>

namespace bpt = backprop_tools;
using DTYPE = float;
using index_t = unsigned;
constexpr index_t OUTER_INPUT_DIM = 10;
constexpr index_t OUTER_OUTPUT_DIM = 10;
constexpr unsigned OUTER_INPUT_DIM_2 = 10;
constexpr unsigned OUTER_OUTPUT_DIM_2 = 10;
using LayerSpec1 = bpt::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Plain>;
using LayerSpec2 = bpt::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Plain>;
using LayerSpec3 = bpt::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Plain>;

struct LayerSpec4{
    typedef DTYPE T;
    static constexpr auto INPUT_DIM = OUTER_INPUT_DIM;
    static constexpr auto OUTPUT_DIM = OUTER_OUTPUT_DIM;
    static constexpr bpt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = bpt::nn::activation_functions::ActivationFunction::IDENTITY;
    // Summary
    static constexpr auto NUM_WEIGHTS = OUTPUT_DIM * INPUT_DIM + OUTPUT_DIM;
};
using LayerSpec5 = bpt::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM_2, OUTER_OUTPUT_DIM_2, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Plain>;

static_assert(bpt::utils::typing::is_same_v<LayerSpec1, LayerSpec2>);
// these should fail
//static_assert(bpt::utils::typing::is_same_v<LayerSpec1, LayerSpec3>);
//static_assert(bpt::utils::typing::is_same_v<LayerSpec1, LayerSpec4>);
//static_assert(bpt::utils::typing::is_same_v<LayerSpec1, LayerSpec5>);



TEST(BACKPROP_TOOLS_NN_MLP_CONVERSIONS, CONVERSIONS) {
    using Device1 = bpt::devices::DefaultDummy;
    using Layer1 = bpt::nn::layers::dense::Layer<LayerSpec1>;

    Device1::SPEC::LOGGING logger;
    Device1 device1;
    device1.logger = &logger;
    Layer1 layer1;

    Layer1 layer11;

    using Device2 = bpt::devices::DefaultCPU;
    using Layer2 = bpt::nn::layers::dense::Layer<LayerSpec2>;

    Device2::SPEC::LOGGING logger2;
    Device2 device2;
    device2.logger = &logger2;
    Layer2 layer2;
    Layer2 layer22;
    Layer2 layer222;

    bpt::malloc(device1, layer1);
    bpt::malloc(device2, layer2);
    bpt::malloc(device2, layer22);
    bpt::malloc(device2, layer222);

    auto rng = bpt::random::default_engine(Device2::SPEC::RANDOM());
    bpt::init_kaiming(device2, layer2, rng);
    bpt::init_kaiming(device2, layer22, rng);
    bpt::init_kaiming(device2, layer222, rng);

    ASSERT_GT(bpt::abs_diff(device2, layer2, layer22), 0);

    bpt::copy(device2, device2, layer22, layer222);

    ASSERT_GT(bpt::abs_diff(device2, layer2, layer22), 0);
    ASSERT_EQ(bpt::abs_diff(device2, layer22, layer222), 0);

    bpt::copy(device2, device2, layer2, layer22);

    ASSERT_EQ(bpt::abs_diff(device2, layer2, layer222), 0);

    bpt::copy(device1, device1, layer1, layer2);

    ASSERT_EQ(bpt::abs_diff(device1, layer1, layer222), 0);

}