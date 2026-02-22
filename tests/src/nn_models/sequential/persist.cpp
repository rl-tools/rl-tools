#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
constexpr TI BATCH_SIZE = 1;

namespace MODEL_FORWARD{
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 10>;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 15, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_1 = rlt::nn::layers::dense::BindConfiguration<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 20, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_2 = rlt::nn::layers::dense::BindConfiguration<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_3 = rlt::nn::layers::dense::BindConfiguration<LAYER_3_SPEC>;

    template <typename... T_CONTENTS>
    using Module = typename rlt::nn_models::sequential::Module<T_CONTENTS...>;
    using MODULE_CHAIN = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
}
namespace MODEL_BACKWARD{
    using MODEL = MODEL_FORWARD::MODEL::template CHANGE_CAPABILITY<rlt::nn::capability::Backward<>>;
}
namespace MODEL_GRADIENT_ADAM{
    using MODEL = MODEL_FORWARD::MODEL::template CHANGE_CAPABILITY<rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>>;
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load_forward_forward) {
    using MODEL = MODEL_FORWARD::MODEL;

    DEVICE device;
    MODEL model, model_loaded;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_forward_forward.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto sequential_model_group = rlt::create_group(device, file, "sequential_model");
        rlt::save(device, model, sequential_model_group);
    }

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_forward_forward.h5", HighFive::File::ReadOnly);
        auto sequential_model_group = rlt::get_group(device, file, "sequential_model");
        rlt::load(device, model_loaded, sequential_model_group);
    }

    auto abs_diff = rlt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load_backward_forward) {

    DEVICE device;
    MODEL_BACKWARD::MODEL model;
    MODEL_FORWARD::MODEL model_loaded;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_backward_forward.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto sequential_model_group = rlt::create_group(device, file, "sequential_model");
        rlt::save(device, model, sequential_model_group);
    }

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_backward_forward.h5", HighFive::File::ReadOnly);
        auto sequential_model_group = rlt::get_group(device, file, "sequential_model");
        rlt::load(device, model_loaded, sequential_model_group);
    }

    auto abs_diff = rlt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load_gradient_adam_forward) {

    DEVICE device;
    MODEL_GRADIENT_ADAM::MODEL model;
    MODEL_FORWARD::MODEL model_loaded;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_gradient_adam_forward.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto sequential_model_group = rlt::create_group(device, file, "sequential_model");
        rlt::save(device, model, sequential_model_group);
    }

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_gradient_adam_forward.h5", HighFive::File::ReadOnly);
        auto sequential_model_group = rlt::get_group(device, file, "sequential_model");
        rlt::load(device, model_loaded, sequential_model_group);
    }

    auto abs_diff = rlt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load_forward_gradient_adam) {

    DEVICE device;
    MODEL_FORWARD::MODEL model;
    MODEL_GRADIENT_ADAM::MODEL model_loaded;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_forward_gradient_adam.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto sequential_model_group = rlt::create_group(device, file, "sequential_model");
        rlt::save(device, model, sequential_model_group);
    }

    bool got_expected_error = false;
    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_forward_gradient_adam.h5", HighFive::File::ReadOnly);
        try{
            auto sequential_model_group = rlt::get_group(device, file, "sequential_model");
            rlt::load(device, model_loaded, sequential_model_group);
        }
        catch(HighFive::DataSetException& e){

            std::cerr << "Error while loading model: " << e.what() << std::endl;
            const std::string error = e.what();
            const bool missing_object = error.find("Object not found") != std::string::npos;
            const bool missing_expected_dataset =
                error.find("\"gradient\"") != std::string::npos ||
                error.find("\"gradient_first_order_moment\"") != std::string::npos ||
                error.find("\"gradient_second_order_moment\"") != std::string::npos;
            if(missing_object && missing_expected_dataset){
                got_expected_error = true;
            }
        }
    }
    ASSERT_EQ(got_expected_error, true);
}

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST, save_and_load_gradient_adam_gradient_adam) {

    DEVICE device;
    MODEL_GRADIENT_ADAM::MODEL model;
    MODEL_GRADIENT_ADAM::MODEL model_loaded;

    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::malloc(device, model);
    rlt::malloc(device, model_loaded);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, rlt::get_layer<0>(model).output, rng);
    rlt::randn(device, rlt::get_layer<0>(model).pre_activations, rng);
    rlt::randn(device, rlt::get_layer<0>(model).weights.gradient, rng);
    rlt::randn(device, rlt::get_layer<0>(model).weights.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<0>(model).weights.gradient_second_order_moment, rng);
    rlt::randn(device, rlt::get_layer<0>(model).biases.gradient, rng);
    rlt::randn(device, rlt::get_layer<0>(model).biases.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<0>(model).biases.gradient_second_order_moment, rng);

    rlt::randn(device, rlt::get_layer<1>(model).output, rng);
    rlt::randn(device, rlt::get_layer<1>(model).pre_activations, rng);
    rlt::randn(device, rlt::get_layer<1>(model).weights.gradient, rng);
    rlt::randn(device, rlt::get_layer<1>(model).weights.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<1>(model).weights.gradient_second_order_moment, rng);
    rlt::randn(device, rlt::get_layer<1>(model).biases.gradient, rng);
    rlt::randn(device, rlt::get_layer<1>(model).biases.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<1>(model).biases.gradient_second_order_moment, rng);

    rlt::randn(device, rlt::get_layer<2>(model).output, rng);
    rlt::randn(device, rlt::get_layer<2>(model).pre_activations, rng);
    rlt::randn(device, rlt::get_layer<2>(model).weights.gradient, rng);
    rlt::randn(device, rlt::get_layer<2>(model).weights.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<2>(model).weights.gradient_second_order_moment, rng);
    rlt::randn(device, rlt::get_layer<2>(model).biases.gradient, rng);
    rlt::randn(device, rlt::get_layer<2>(model).biases.gradient_first_order_moment, rng);
    rlt::randn(device, rlt::get_layer<2>(model).biases.gradient_second_order_moment, rng);

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_gradient_adam_gradient_adam.h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto sequential_model_group = rlt::create_group(device, file, "sequential_model");
        rlt::save(device, model, sequential_model_group);
    }

    {
        auto file = HighFive::File("test_rl_tools_nn_models_sequential_save_gradient_adam_gradient_adam.h5", HighFive::File::ReadOnly);
        auto sequential_model_group = rlt::get_group(device, file, "sequential_model");
        rlt::load(device, model_loaded, sequential_model_group);
    }

    auto abs_diff = rlt::abs_diff(device, model, model_loaded);

    ASSERT_EQ(abs_diff, 0);
}
