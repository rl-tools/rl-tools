#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include "groups_export.h"
#include <gtest/gtest.h>

namespace rl_tools::tests::groups{
    TEST(NN_GROUPS_PERSIST_CODE, PreservesMembershipAndValues){
        using SPEC = grouped_chain::TYPE::SPEC;
        using FIRST = typename nn_models::sequential::layer_spec_t<0, SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        using THIRD = typename nn_models::sequential::layer_spec_t<2, SPEC>::CONTENT::WEIGHTS_PARAMETER_SPEC;
        static_assert(nn::group::Only<Fixed>::MATCH<FIRST>);
        static_assert(!nn::group::Only<Learned>::MATCH<FIRST>);
        static_assert(nn::group::Only<Learned>::MATCH<THIRD>);
        using HEAD = grouped_parallel::TYPE::SPEC::HEAD_TYPE;
        static_assert(nn::group::Only<Learned>::MATCH<typename HEAD::LOG_STD_PARAMETER_SPEC>);
        static_assert(!nn::group::Only<Learned>::MATCH<typename HEAD::SPEC::INPUT_LAYER::WEIGHTS_PARAMETER_SPEC>);
        devices::DefaultCPU device;
        auto network = grouped_chain::factory<>;
        auto actor = grouped_parallel::factory<>;
        EXPECT_EQ(abs_diff(device, network, grouped_chain::module), 0);
        EXPECT_EQ(abs_diff(device, actor, grouped_parallel::module), 0);
    }
}
