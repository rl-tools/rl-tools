#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>

#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/optimizers/sgd/instance/persist_code.h>

#include <gtest/gtest.h>

#include <string>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using SHAPE = rlt::tensor::Shape<TI, 2, 3>;

using TYPE_POLICY_NO_MASTER = rlt::numeric_types::Policy<float>;
using TYPE_POLICY_MASTER_EQ = rlt::numeric_types::Policy<float, rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
using TYPE_POLICY_MASTER_DIFF = rlt::numeric_types::Policy<float, rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, double>>;

template <typename TYPE_POLICY>
using SGD_PARAMETER_SPEC = rlt::nn::parameters::SGD::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true, false>;

template <typename TYPE_POLICY>
using ADAM_PARAMETER_SPEC = rlt::nn::parameters::Adam::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true, false>;

template <typename TYPE_POLICY>
using SGD_PARAMETER = rlt::nn::parameters::SGD::Instance<SGD_PARAMETER_SPEC<TYPE_POLICY>>;

template <typename TYPE_POLICY>
using ADAM_PARAMETER = rlt::nn::parameters::Adam::Instance<ADAM_PARAMETER_SPEC<TYPE_POLICY>>;

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS, USE_MASTER_TOGGLE) {
    static_assert(!SGD_PARAMETER<TYPE_POLICY_NO_MASTER>::USE_MASTER_PARAMETERS);
    static_assert(!SGD_PARAMETER<TYPE_POLICY_MASTER_EQ>::USE_MASTER_PARAMETERS);
    static_assert(SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF>::USE_MASTER_PARAMETERS);

    static_assert(!ADAM_PARAMETER<TYPE_POLICY_NO_MASTER>::USE_MASTER_PARAMETERS);
    static_assert(!ADAM_PARAMETER<TYPE_POLICY_MASTER_EQ>::USE_MASTER_PARAMETERS);
    static_assert(ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF>::USE_MASTER_PARAMETERS);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS, SGD_COPY_ABS_DIFF_MIXED_MASTER_STATE) {
    DEVICE device;
    SGD_PARAMETER<TYPE_POLICY_NO_MASTER> no_master;
    SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF> master;
    rlt::malloc(device, no_master);
    rlt::malloc(device, master);

    rlt::set_all(device, no_master.parameters, 1.5f);
    rlt::set_all(device, no_master.gradient, -2.0f);
    rlt::set_all(device, no_master.velocity, 3.0f);
    rlt::copy(device, device, no_master, master);
    rlt::copy(device, device, no_master.parameters, master.parameters);

    auto diff = rlt::abs_diff(device, no_master, master);
    EXPECT_EQ(diff, 0);
    EXPECT_EQ(rlt::abs_diff(device, master.parameters, master.master_parameters), 0);

    rlt::free(device, master);
    rlt::free(device, no_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS, ADAM_COPY_ABS_DIFF_MIXED_MASTER_STATE) {
    DEVICE device;
    ADAM_PARAMETER<TYPE_POLICY_NO_MASTER> no_master;
    ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF> master;
    rlt::malloc(device, no_master);
    rlt::malloc(device, master);

    rlt::set_all(device, no_master.parameters, 1.5f);
    rlt::set_all(device, no_master.gradient, -2.0f);
    rlt::set_all(device, no_master.gradient_first_order_moment, 0.1f);
    rlt::set_all(device, no_master.gradient_second_order_moment, 0.2f);
    rlt::copy(device, device, no_master, master);
    rlt::copy(device, device, no_master.parameters, master.parameters);

    auto diff = rlt::abs_diff(device, no_master, master);
    EXPECT_EQ(diff, 0);
    EXPECT_EQ(rlt::abs_diff(device, master.parameters, master.master_parameters), 0);

    rlt::free(device, master);
    rlt::free(device, no_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS, SGD_PERSIST_CODE_MASTER_GUARDS) {
    DEVICE device;
    SGD_PARAMETER<TYPE_POLICY_NO_MASTER> no_master;
    SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF> master;
    rlt::malloc(device, no_master);
    rlt::malloc(device, master);
    rlt::set_all(device, no_master.parameters, 0.0f);
    rlt::set_all(device, no_master.gradient, 0.0f);
    rlt::set_all(device, no_master.velocity, 0.0f);
    rlt::set_all(device, master.parameters, 0.0f);
    rlt::set_all(device, master.gradient, 0.0f);
    rlt::set_all(device, master.velocity, 0.0f);
    rlt::set_all(device, master.master_parameters, 0.0);

    const auto code_no_master = rlt::save_code_split(device, no_master, "sgd_no_master");
    const auto code_master = rlt::save_code_split(device, master, "sgd_master");
    const std::string code_no_master_text = code_no_master.header + code_no_master.body;
    const std::string code_master_text = code_master.header + code_master.body;

    EXPECT_EQ(code_no_master_text.find("master_parameters_memory"), std::string::npos);
    EXPECT_EQ(code_no_master_text.find("master_parameters_memory::container"), std::string::npos);
    EXPECT_NE(code_master_text.find("master_parameters_memory"), std::string::npos);
    EXPECT_NE(code_master_text.find("master_parameters_memory::container"), std::string::npos);

    rlt::free(device, master);
    rlt::free(device, no_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS, ADAM_PERSIST_CODE_MASTER_GUARDS) {
    DEVICE device;
    ADAM_PARAMETER<TYPE_POLICY_NO_MASTER> no_master;
    ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF> master;
    rlt::malloc(device, no_master);
    rlt::malloc(device, master);
    rlt::set_all(device, no_master.parameters, 0.0f);
    rlt::set_all(device, no_master.gradient, 0.0f);
    rlt::set_all(device, no_master.gradient_first_order_moment, 0.0f);
    rlt::set_all(device, no_master.gradient_second_order_moment, 0.0f);
    rlt::set_all(device, master.parameters, 0.0f);
    rlt::set_all(device, master.gradient, 0.0f);
    rlt::set_all(device, master.gradient_first_order_moment, 0.0f);
    rlt::set_all(device, master.gradient_second_order_moment, 0.0f);
    rlt::set_all(device, master.master_parameters, 0.0);

    const auto code_no_master = rlt::save_code_split(device, no_master, "adam_no_master");
    const auto code_master = rlt::save_code_split(device, master, "adam_master");
    const std::string code_no_master_text = code_no_master.header + code_no_master.body;
    const std::string code_master_text = code_master.header + code_master.body;

    EXPECT_EQ(code_no_master_text.find("master_parameters_memory"), std::string::npos);
    EXPECT_EQ(code_no_master_text.find("master_parameters_memory::container"), std::string::npos);
    EXPECT_NE(code_master_text.find("master_parameters_memory"), std::string::npos);
    EXPECT_NE(code_master_text.find("master_parameters_memory::container"), std::string::npos);

    rlt::free(device, master);
    rlt::free(device, no_master);
}
