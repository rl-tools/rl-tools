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

namespace lic = backprop_tools;
using DTYPE = float;
using index_t = unsigned;
constexpr index_t OUTER_INPUT_DIM = 10;
constexpr index_t OUTER_OUTPUT_DIM = 10;
constexpr unsigned OUTER_INPUT_DIM_2 = 10;
constexpr unsigned OUTER_OUTPUT_DIM_2 = 10;
using LayerSpec1 = lic::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, lic::nn::activation_functions::ActivationFunction::IDENTITY>;
using LayerSpec2 = lic::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, lic::nn::activation_functions::ActivationFunction::IDENTITY>;
using LayerSpec3 = lic::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM, OUTER_OUTPUT_DIM, lic::nn::activation_functions::ActivationFunction::RELU>;

struct LayerSpec4{
    typedef DTYPE T;
    static constexpr auto INPUT_DIM = OUTER_INPUT_DIM;
    static constexpr auto OUTPUT_DIM = OUTER_OUTPUT_DIM;
    static constexpr lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = lic::nn::activation_functions::ActivationFunction::IDENTITY;
    // Summary
    static constexpr auto NUM_WEIGHTS = OUTPUT_DIM * INPUT_DIM + OUTPUT_DIM;
};
using LayerSpec5 = lic::nn::layers::dense::Specification<DTYPE, index_t, OUTER_INPUT_DIM_2, OUTER_OUTPUT_DIM_2, lic::nn::activation_functions::ActivationFunction::RELU>;

static_assert(lic::utils::typing::is_same_v<LayerSpec1, LayerSpec2>);
// these should fail
//static_assert(lic::utils::typing::is_same_v<LayerSpec1, LayerSpec3>);
//static_assert(lic::utils::typing::is_same_v<LayerSpec1, LayerSpec4>);
//static_assert(lic::utils::typing::is_same_v<LayerSpec1, LayerSpec5>);



TEST(BACKPROP_TOOLS_NN_MLP_CONVERSIONS, CONVERSIONS) {
    using Device1 = lic::devices::DefaultDummy;
    using Layer1 = lic::nn::layers::dense::Layer<LayerSpec1>;

    Device1::SPEC::LOGGING logger;
    Device1 device1;
    device1.logger = &logger;
    Layer1 layer1;

    Layer1 layer11;

    using Device2 = lic::devices::DefaultCPU;
    using Layer2 = lic::nn::layers::dense::Layer<LayerSpec2>;

    Device2::SPEC::LOGGING logger2;
    Device2 device2;
    device2.logger = &logger2;
    Layer2 layer2;
    Layer2 layer22;
    Layer2 layer222;

    lic::malloc(device1, layer1);
    lic::malloc(device2, layer2);
    lic::malloc(device2, layer22);
    lic::malloc(device2, layer222);

    auto rng = lic::random::default_engine(Device2::SPEC::RANDOM());
    lic::init_kaiming(device2, layer2, rng);
    lic::init_kaiming(device2, layer22, rng);
    lic::init_kaiming(device2, layer222, rng);

    ASSERT_GT(lic::abs_diff(device2, layer2, layer22), 0);

    lic::copy(device2, device2, layer22, layer222);

    ASSERT_GT(lic::abs_diff(device2, layer2, layer22), 0);
    ASSERT_EQ(lic::abs_diff(device2, layer22, layer222), 0);

    lic::copy(device2, device2, layer2, layer22);

    ASSERT_EQ(lic::abs_diff(device2, layer2, layer222), 0);

    lic::copy(device1, device1, layer1, layer2);

    ASSERT_EQ(lic::abs_diff(device1, layer1, layer222), 0);

}