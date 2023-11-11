#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/nn_models/sequential/persist.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>

using T = float;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

namespace MODEL_1{
    using namespace rlt::nn_models::sequential::interface;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 15, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Plain>;
    using LAYER_1 = rlt::nn::layers::dense::Layer<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 15, 20, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Plain>;
    using LAYER_2 = rlt::nn::layers::dense::Layer<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 20, 5, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Plain>;
    using LAYER_3 = rlt::nn::layers::dense::Layer<LAYER_3_SPEC>;

    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load) {
    using MODEL = MODEL_1::MODEL;

    DEVICE device;
    MODEL model, model_loaded;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        rlt::save(device, model, file.createGroup("sequential_model"));
    }

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save.h5", HighFive::File::ReadOnly);
        rlt::load(device, model_loaded, file.getGroup("sequential_model"));
    }

    auto abs_diff = rlt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}