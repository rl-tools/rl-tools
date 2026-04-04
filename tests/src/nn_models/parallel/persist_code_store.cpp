#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/conv2d/persist_code.h>
#include <rl_tools/nn/layers/flatten/persist_code.h>
#include <rl_tools/nn/layers/unflatten/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/parallel/persist_code.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

std::optional<std::string> get_env_var(const std::string& var) {
    const char* value = std::getenv(var.c_str());
    if (value) {
        return std::string(value);
    } else {
        return std::nullopt;
    }
}

namespace PARALLEL_MODEL{
    using TYPE_POLICY = rlt::numeric_types::Policy<double>;
    constexpr TI BATCH_SIZE = 1;
    constexpr TI IMG_H = 8;
    constexpr TI IMG_W = 8;
    constexpr TI IMG_C = 3;
    constexpr TI STATE_DIM = 12;
    constexpr TI HIDDEN_DIM = 16;
    constexpr TI ACTION_DIM = 4;

    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, IMG_C>;
    using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_DIM>;

    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZE_CONFIG>;
    using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<TYPE_POLICY, TI, IMG_H, IMG_W, IMG_C>;
    using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
    using IMAGE_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using IMAGE_DENSE = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_CONFIG>;
    using IMAGE_BRANCH = rlt::nn_models::sequential::Module<FLATTEN, STANDARDIZE, UNFLATTEN, CONV, OUTPUT_FLATTEN, IMAGE_DENSE>;

    using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
    using STATE_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using STATE_DENSE = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_CONFIG>;
    using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE>;

    using HEAD_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ACTION_DIM, 2, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using HEAD = rlt::nn_models::mlp::BindConfiguration<HEAD_CONFIG>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, IMAGE_BRANCH, STATE_BRANCH, IMAGE_INPUT_SHAPE, STATE_INPUT_SHAPE, HEAD>;
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_CODE, STORE) {
    using MODEL = PARALLEL_MODEL::MODEL;
    using T = typename MODEL::TYPE_POLICY::DEFAULT;

    DEVICE device;
    MODEL model;
    MODEL::Buffer<> buffer;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::INPUT_SHAPE_A>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::INPUT_SHAPE_B>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::OUTPUT_SHAPE>> output;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, output);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);

    rlt::evaluate(device, model, input_a, input_b, output, buffer, rng);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_a_code = rlt::save_code_split(device, input_a, "input_a", true, 1);
        auto input_b_code = rlt::save_code_split(device, input_b, "input_b", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);

        auto header = model_code.header + "\n" + input_a_code.header + "\n" + input_b_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_a_code.body + "\n" + input_b_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output_string = wrapped.header + "\n" + wrapped.body;

        std::ofstream file;
        std::string output_path = "tests/data/nn_models_parallel_persist_code.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_path) << std::endl;
        file << output_string;
        file.close();
    }

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, output);
}
