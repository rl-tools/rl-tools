#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "../../../../data/test_nn_layers_gru_persist_code.h"

// #include <highfive/H5File.hpp>

namespace rlt = rl_tools;

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <optional>

std::optional<std::string> get_env_var(const std::string& var) {
    const char* value = std::getenv(var.c_str());
    if (value) {
        return std::string(value);
    } else {
        return std::nullopt;
    }
}

//using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using T = double;

TEST(RL_TOOLS_NN_LAYERS_GRU, PERSIST_CODE_COMPILE){
    constexpr TI BATCH_SIZE = rlt::get<1>(input::SHAPE{});
    using GRU = gru::TYPE;
    typename GRU::Buffer<> buffer;
    using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using ADAM = rlt::nn::optimizers::Adam<ADAM_SPEC>;
    ADAM optimizer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, output::SHAPE>> output_imported;

    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);


    rlt::malloc(device, buffer);
    rlt::malloc(device, output_imported);


    const gru::TYPE module = gru::factory_function(); // MSVC fix:
    rlt::evaluate(device, module, input::container, output_imported, buffer, rng);

    std::cout << "expected output: " << std::endl;
    rlt::print(device, output::container);
    std::cout << "actual output: " << std::endl;
    rlt::print(device, output_imported);

    T abs_diff = rlt::abs_diff(device, output::container, output_imported);
    std::cout << "abs_diff: " << abs_diff << std::endl;
    EXPECT_LT(abs_diff, 1e-6);


}
