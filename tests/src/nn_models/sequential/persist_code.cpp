#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = float;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

namespace MODEL_1{
    using namespace rlt::nn_models::sequential::interface;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 13, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Input>;
    using LAYER_1 = rlt::nn::layers::dense::Layer<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 64, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Normal>;
    using LAYER_2 = rlt::nn::layers::dense::Layer<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 64, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Output>;
    using LAYER_3 = rlt::nn::layers::dense::Layer<LAYER_3_SPEC>;

    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_CODE, save_and_load) {
    using MODEL = MODEL_1::MODEL;

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

    rlt::evaluate(device, model, input, output, buffer);

    rlt::print(device, output);



    {
        auto model_code = rlt::save_code_split(device, model, "model", true, 1);
        auto input_code = rlt::save_split(device, input, "input", true, 1);
        auto output_code = rlt::save_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = rlt::embed_in_namespace(device, {header, body}, "rl_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
        std::filesystem::create_directories("data");
        std::ofstream file;
        file.open("tests/data/nn_models_sequential_persist_code.h", std::ios::out | std::ios::trunc);
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        std::cout << "Full file path: " << std::filesystem::absolute("data/nn_models_sequential_persist_code.h") << std::endl;
        file << output;
        file.close();
    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::Buffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}