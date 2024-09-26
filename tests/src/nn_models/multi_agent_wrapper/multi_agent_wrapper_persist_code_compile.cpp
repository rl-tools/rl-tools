#include <rl_tools/operations/cpu.h>

#include "../../../data/nn_models_multi_agent_wrapper_persist_code.h"

#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>


namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = double;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

TEST(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_PERSIST_CODE_COMPILE, GRADIENT){
    DEVICE device;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM>> output;
    rl_tools_export::model::MODEL::Buffer<1> buffer;

    rlt::malloc(device, output);
    rlt::malloc(device, buffer);

    bool rng = false;

    rlt::evaluate(device, rl_tools_export::model::module, rl_tools_export::input::container, output, buffer, rng);

    auto abs_diff = rlt::abs_diff(device, output, rl_tools_export::output::container);

    std::cout << "Original output:" << std::endl;
    rlt::print(device, rl_tools_export::output::container);
    std::cout << "Loaded output:" << std::endl;
    rlt::print(device, output);

    std::cout << "abs_diff: " << abs_diff << std::endl;
    ASSERT_LT(abs_diff, 1e-5);
}

