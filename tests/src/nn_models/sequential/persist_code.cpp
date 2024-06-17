#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev//operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <optional>
#include <string>

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

std::optional<std::string> get_env_var(const std::string& var) {
    const char* value = std::getenv(var.c_str());
    if (value) {
        return std::string(value);
    } else {
        return std::nullopt;
    }
}

namespace MODEL_BENCHMARK{
    using T_BENCHMARK = float;
    using LAYER_PARAMETERS = rlt::nn::layers::dense::DefaultInitializer<T_BENCHMARK, TI>;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T_BENCHMARK, TI, 13, 64, rlt::nn::activation_functions::ActivationFunction::RELU, LAYER_PARAMETERS, rlt::nn::parameters::groups::Input>;
    using LAYER_1 = rlt::nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T_BENCHMARK, TI, 64, 64, rlt::nn::activation_functions::ActivationFunction::RELU, LAYER_PARAMETERS, rlt::nn::parameters::groups::Normal>;
    using LAYER_2 = rlt::nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T_BENCHMARK, TI, 64, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY, LAYER_PARAMETERS, rlt::nn::parameters::groups::Output>;
    using LAYER_3 = rlt::nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

    constexpr TI BATCH_SIZE = 1;

    using IF = rlt::nn_models::sequential::Interface<rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>>;
    using MODEL = IF::Module<LAYER_1::Layer, IF::Module<LAYER_2::Layer, IF::Module<LAYER_3::Layer>>>;
}

namespace MODEL_1{
    using T = double;
    using LAYER_PARAMETERS = rlt::nn::layers::dense::DefaultInitializer<T, TI>;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 13, 64, rlt::nn::activation_functions::ActivationFunction::RELU, LAYER_PARAMETERS, rlt::nn::parameters::groups::Input>;
    using LAYER_1 = rlt::nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 64, 64, rlt::nn::activation_functions::ActivationFunction::RELU, LAYER_PARAMETERS, rlt::nn::parameters::groups::Normal>;
    using LAYER_2 = rlt::nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 64, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY, LAYER_PARAMETERS, rlt::nn::parameters::groups::Output>;
    using LAYER_3 = rlt::nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

    constexpr TI BATCH_SIZE = 1;

    using IF = rlt::nn_models::sequential::Interface<rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>>;
    using MODEL = IF::Module<LAYER_1::Layer, IF::Module<LAYER_2::Layer, IF::Module<LAYER_3::Layer>>>;
}
namespace MODEL_2{
    using T = double;
    constexpr TI BATCH_SIZE = 1;
    using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, 13, 4, 3, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::BindSpecification<ACTOR_SPEC>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
    using ACTOR_MODULE = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork>;
    using STANDARDIZATION_LAYER_SPEC = rlt::nn::layers::standardize::Specification<T, TI, 13>;
    using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindSpecification<STANDARDIZATION_LAYER_SPEC>;
    using MODEL = typename IF::template Module<STANDARDIZATION_LAYER::template Layer, ACTOR_MODULE>;
}

namespace MODEL_MLP{
    using T = double;
    constexpr TI BATCH_SIZE = 1;
    using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, 13, 4, 3, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_TYPE = rlt::nn_models::mlp::BindSpecification<ACTOR_SPEC>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
    using MODEL = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork>;
}

namespace MODEL_SAMPLE_AND_SQUASH{
    using T = double;
    constexpr TI BATCH_SIZE = 1;
    using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, 13, 8, 3, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::BindSpecification<ACTOR_SPEC>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
    using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, 4, rlt::nn::layers::sample_and_squash::DefaultParameters<T>>;
    using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
    using SAMPLE_AND_SQUASH_MODULE = typename IF::template Module<SAMPLE_AND_SQUASH_LAYER::template Layer>;
    using MODEL = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, save_and_load) {
    using MODEL = MODEL_1::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    MODEL model;
    MODEL::Buffer<1> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);



    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_path = "tests/data/nn_models_sequential_persist_code.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, save_model_benchmark) {
    using MODEL = MODEL_BENCHMARK::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    MODEL model;
    MODEL::Buffer<1> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);



    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_path = "tests/data/nn_models_sequential_persist_code_benchmark.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_2) {
    using MODEL = MODEL_2::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    MODEL model;
    MODEL::Buffer<MODEL_2::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    {
        auto& first_layer = model.content;
        for(TI input_i=0; input_i < MODEL::INPUT_DIM; input_i++){
            rlt::set(first_layer.mean.parameters, 0, input_i, input_i);
            rlt::set(first_layer.precision.parameters, 0, input_i, input_i*2);
            rlt::set(first_layer.output, 0, input_i, input_i*3);
        }
    }

    {
        auto& last_layer = rlt::get_last_layer(model);
        for(TI output_i=0; output_i < MODEL::OUTPUT_DIM; output_i++){
            rlt::set(last_layer.log_std.parameters, 0, output_i, output_i);
            rlt::set(last_layer.log_std.gradient, 0, output_i, output_i*2);
            rlt::set(last_layer.log_std.gradient_first_order_moment, 0, output_i, output_i*3);
            rlt::set(last_layer.log_std.gradient_second_order_moment, 0, output_i, output_i*4);
        }
    }

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_2.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_2_forward) {
    using MODEL = MODEL_2::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY_FORWARD = rlt::nn::layer_capability::Forward;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY_FORWARD> model;
    MODEL::Buffer<MODEL_2::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    {
        auto& first_layer = model.content;
        for(TI input_i=0; input_i < MODEL::INPUT_DIM; input_i++){
            rlt::set(first_layer.mean.parameters, 0, input_i, input_i);
            rlt::set(first_layer.precision.parameters, 0, input_i, input_i*2);
        }
    }

    {
        auto& last_layer = rlt::get_last_layer(model);
        for(TI output_i=0; output_i < MODEL::OUTPUT_DIM; output_i++){
            rlt::set(last_layer.log_std.parameters, 0, output_i, output_i);
        }
    }

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_2_forward.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}
TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_2_gradient) {
    using MODEL = MODEL_2::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY_BACKWARD = rlt::nn::layer_capability::Backward<1>;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY_BACKWARD> model;
    MODEL::Buffer<MODEL_2::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    {
        auto& first_layer = model.content;
        for(TI input_i=0; input_i < MODEL::INPUT_DIM; input_i++){
            rlt::set(first_layer.mean.parameters, 0, input_i, input_i);
            rlt::set(first_layer.precision.parameters, 0, input_i, input_i*2);
        }
    }

    {
        auto& last_layer = rlt::get_last_layer(model);
        for(TI output_i=0; output_i < MODEL::OUTPUT_DIM; output_i++){
            rlt::set(last_layer.log_std.parameters, 0, output_i, output_i);
        }
    }

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_2_backward.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_mlp){
    using MODEL = MODEL_MLP::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
//    using CAPABILITY_BACKWARD = rlt::nn::layer_capability::Backward<1>;
//    MODEL::template CHANGE_CAPABILITY<CAPABILITY_BACKWARD> model;
    MODEL model;
    MODEL::Buffer<MODEL_MLP::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_mlp.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_mlp_forward){
    using MODEL = MODEL_MLP::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY_FORWARD = rlt::nn::layer_capability::Forward;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY_FORWARD> model;
    MODEL::Buffer<MODEL_MLP::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_mlp_forward.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_sample_and_squash_forward){
    using MODEL = MODEL_SAMPLE_AND_SQUASH::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY_FORWARD = rlt::nn::layer_capability::Forward;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY_FORWARD> model;
    MODEL::Buffer<MODEL_MLP::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_sample_and_squash_forward.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_sample_and_squash_backward){
    using MODEL = MODEL_SAMPLE_AND_SQUASH::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY = rlt::nn::layer_capability::Backward<3>;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY> model;
    MODEL::Buffer<MODEL_MLP::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_sample_and_squash_backward.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, model_sample_and_squash_gradient){
    using MODEL = MODEL_SAMPLE_AND_SQUASH::MODEL;
    using T = typename MODEL::T;

    DEVICE device;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, 3>;
    MODEL::template CHANGE_CAPABILITY<CAPABILITY> model;
    MODEL::Buffer<MODEL_MLP::BATCH_SIZE> buffer;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, model, input, output, buffer, rng);

    rlt::print(device, output);

    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_code_split(device, input, "input", true, 1);
        auto output_code = rlt::save_code_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
//        std::filesystem::create_directories("data");
        std::ofstream file;
        std::string output_file_path = "tests/data/nn_models_sequential_persist_code_model_sample_and_squash_gradient.h" + std::string((get_env_var("GITHUB_ACTIONS") ? ".disabled" : ""));
        file.open(output_file_path, std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute(output_file_path) << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}
