#include <rl_tools/operations/cpu.h>

#include "../../../data/nn_models_sequential_persist_code_model_sample_and_squash_forward.h"

#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev//operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>


namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = double;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_COMPILE, MODEL_SAMPLE_AND_SQUASH_FORWARD){
    DEVICE device;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM>> output;
    rl_tools_export::model::MODEL::Buffer<1> buffer;

    rlt::malloc(device, output);
    rlt::malloc(device, buffer);

    auto rng = rlt::random::default_engine(device.random, 0);

    rlt::evaluate(device, rl_tools_export::model::module, rl_tools_export::input::container, output, buffer, rng);

    auto abs_diff = rlt::abs_diff(device, output, rl_tools_export::output::container);

    std::cout << "Oiriginal output:" << std::endl;
    rlt::print(device, rl_tools_export::output::container);
    std::cout << "Loaded output:" << std::endl;
    rlt::print(device, output);

    std::cout << "abs_diff: " << abs_diff << std::endl;
    ASSERT_LT(abs_diff, 1e-5);
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE_COMPILE, MODEL_SAMPLE_AND_SQUASH_FORWARD_RESAMPLE){
    DEVICE device;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM>> output;
    rl_tools_export::model::MODEL::Buffer<1> buffer;

    rlt::malloc(device, output);
    rlt::malloc(device, buffer);

    auto rng = rlt::random::default_engine(device.random, 0);

    auto mode = rlt::nn::Mode<rlt::nn::layers::sample_and_squash::mode::Sample<rlt::nn::mode::Default<>>>{};
    rlt::evaluate(device, rl_tools_export::model::module, rl_tools_export::input::container, output, buffer, rng, mode);

    auto abs_diff = rlt::abs_diff(device, output, rl_tools_export::output::container);

    std::cout << "Oiriginal output:" << std::endl;
    rlt::print(device, rl_tools_export::output::container);
    std::cout << "Loaded output:" << std::endl;
    rlt::print(device, output);

    std::cout << "abs_diff: " << abs_diff << std::endl;
    ASSERT_GT(abs_diff, 1e-5);
}

