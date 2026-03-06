#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/conv2d/persist_code.h>
#include <rl_tools/nn/layers/flatten/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;

#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>

std::optional<std::string> get_env_var(const std::string& var) {
    const char* value = std::getenv(var.c_str());
    if (value) {
        return std::string(value);
    } else {
        return std::nullopt;
    }
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN_PERSIST_CODE, STORE) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    constexpr TI BATCH = 2;
    constexpr TI H = 6, W = 6, C = 4;
    constexpr TI DENSE_OUT = 8;

    // Conv2d(8, k3, s1, p0, RELU) → Flatten → Dense(8)
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH, H, W, C>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, DENSE_OUT, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

    using CAPA = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV, rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<DENSE>>>;
    using MODEL = rlt::nn_models::sequential::Build<CAPA, MODULE_CHAIN, INPUT_SHAPE>;

    MODEL model;
    typename MODEL::template Buffer<true> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output;

    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);

    rlt::init(device, rng, 0);
    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    std::string output_code;
    output_code += rlt::save_code(device, model, "model");
    output_code += rlt::save_code(device, input, "input", true);
    output_code += rlt::save_code(device, output, "output", true);

    std::ofstream file;
    std::string output_path = "tests/data/test_nn_layers_flatten_persist_code.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
    file.open(output_path, std::ios::out | std::ios::trunc);
    std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Full file path: " << std::filesystem::absolute(output_path) << std::endl;
    file << output_code;
    file.close();
}
