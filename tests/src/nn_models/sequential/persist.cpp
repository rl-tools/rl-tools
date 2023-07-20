#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/layers/dense/operations_generic.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <backprop_tools/nn_models/sequential/persist.h>
namespace bpt = backprop_tools;
#include <gtest/gtest.h>

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
    MODEL model, model_loaded;

    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    bpt::malloc(device, model);
    bpt::malloc(device, model_loaded);

    bpt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_backprop_tools_nn_models_sequential_save.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        bpt::save(device, model, file.createGroup("sequential_model"));
    }

    {
        auto file = HighFive::File("test_backprop_tools_nn_models_sequential_save.h5", HighFive::File::ReadOnly);
        bpt::load(device, model_loaded, file.getGroup("sequential_model"));
    }

    auto abs_diff = bpt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}