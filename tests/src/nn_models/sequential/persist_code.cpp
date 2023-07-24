#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/layers/dense/operations_generic.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <backprop_tools/nn/parameters/persist_code.h>
#include <backprop_tools/nn/optimizers/adam/persist_code.h>
#include <backprop_tools/nn/layers/dense/persist_code.h>
#include <backprop_tools/nn_models/sequential/persist_code.h>

namespace bpt = backprop_tools;
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using T = float;
using DEVICE = bpt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

namespace MODEL_1{
    using namespace bpt::nn_models::sequential::interface;
    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 15, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Plain>;
    using LAYER_1 = bpt::nn::layers::dense::Layer<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 15, 20, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Plain>;
    using LAYER_2 = bpt::nn::layers::dense::Layer<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 5, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Plain>;
    using LAYER_3 = bpt::nn::layers::dense::Layer<LAYER_3_SPEC>;

    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}

TEST(BACKPROP_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load) {
    using MODEL = MODEL_1::MODEL;

    DEVICE device;
    MODEL model;
    MODEL::DoubleBuffer<1> buffer;

    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, MODEL::INPUT_DIM>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, MODEL::OUTPUT_DIM>> output;

    bpt::malloc(device, input);
    bpt::malloc(device, output);
    bpt::malloc(device, model);
    bpt::malloc(device, buffer);

    bpt::init_weights(device, model, rng);
    bpt::randn(device, input, rng);

    bpt::evaluate(device, model, input, output, buffer);

    bpt::print(device, output);

    {
        auto model_code = bpt::save_code_split(device, model, "model", true, 1);
        auto input_code = bpt::save_split(device, input, "input", true, 1);
        auto output_code = bpt::save_split(device, output, "output", true, 1);
        auto header = model_code.header + "\n" + input_code.header + "\n" + output_code.header;
        auto body = model_code.body + "\n" + input_code.body + "\n" + output_code.body;

        auto wrapped = bpt::embed_in_namespace(device, {header, body}, "backprop_tools_export", 0);

        auto output = wrapped.header + "\n" + wrapped.body;
//        std::cout << "output: " << output << std::endl;
        std::filesystem::create_directories("data");
        std::ofstream file;
        file.open ("data/nn_models_sequential_persist_code.h");
        file << output;
        file.close();

    }

    std::cout << "output dim " << MODEL::OUTPUT_DIM << std::endl;
    std::cout << "max hidden dim " << MODEL::DoubleBuffer<1>::SPEC::MAX_HIDDEN_DIM << std::endl;
}