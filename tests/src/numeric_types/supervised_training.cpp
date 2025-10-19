struct TestTag{};

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = rl_tools;

#include <gtest/gtest.h>


using DEVICE = rlt::devices::DefaultCPU;
using TEST_USE_CASE = rlt::numeric_types::UseCase<TestTag, float>;

using TYPE_POLICY = rlt::numeric_types::Policy<float, TEST_USE_CASE>;
using T = float;
static_assert(rlt::utils::typing::is_same_v<TYPE_POLICY::GET<TestTag>, float>);

using TI = typename DEVICE::index_t;
constexpr bool DYNAMIC_ALLOCATION = true;

constexpr TI INPUT_DIM = 1;
constexpr TI BATCH_SIZE = 32;
constexpr TI OUTPUT_DIM = 1;
constexpr TI NUM_LAYERS = 3;
constexpr TI HIDDEN_DIM = 32;
constexpr TI DATASET_SIZE = 1000;

using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>;
using MLP = rlt::nn_models::mlp::NeuralNetwork<MLP_CONFIG, CAPABILITY, INPUT_SHAPE>;
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{

};
using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, DYNAMIC_ALLOCATION>;
using OPTIMIZER = rlt::nn::optimizers::Adam<ADAM_SPEC>;


using DATASET_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, DATASET_SIZE, INPUT_DIM>;
using INPUT_TYPE = typename TYPE_POLICY::template GET<rlt::nn::numeric_types::categories::Input>;
using DATASET_INPUT_SPEC = rlt::tensor::Specification<INPUT_TYPE, TI, DATASET_INPUT_SHAPE, DYNAMIC_ALLOCATION>;
using DATASET_OUTPUT_SHAPE = rlt::tensor::Shape<TI, 1, DATASET_SIZE, OUTPUT_DIM>;
using DATASET_OUTPUT_SPEC = rlt::tensor::Specification<INPUT_TYPE, TI, DATASET_OUTPUT_SHAPE, DYNAMIC_ALLOCATION>;


template <typename DEVICE, typename INPUT, typename OUTPUT>
void fill_targets(DEVICE& device, INPUT& input, OUTPUT& output){
    for (TI sample_i=0; sample_i < INPUT::SPEC::SHAPE::template GET<1>; sample_i++){
        for (TI output_i=0; output_i < OUTPUT_DIM; output_i++){
            INPUT_TYPE value = rlt::get(device, input, 0, sample_i, 0);
            value *= value;
            rlt::set(device, output, value, 0, sample_i, output_i);
        }
    }
}


TEST(RL_TOOLS_NUMERIC_TYPES_TYPE_POLICY, MAIN){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    MLP model;
    MLP::Buffer<DYNAMIC_ALLOCATION> buffers;
    OPTIMIZER optimizer;
    rlt::Tensor<DATASET_INPUT_SPEC> X_train, X_test;
    rlt::Tensor<DATASET_OUTPUT_SPEC> y_train, y_test;
    rlt::Tensor<rlt::tensor::Specification<INPUT_TYPE, TI, MLP::OUTPUT_SHAPE>> d_output;

    rlt::init(device);

    rlt::malloc(device, rng);
    rlt::malloc(device, model);
    rlt::malloc(device, buffers);
    rlt::malloc(device, optimizer);
    rlt::malloc(device, X_train);
    rlt::malloc(device, X_test);
    rlt::malloc(device, y_train);
    rlt::malloc(device, y_test);
    rlt::malloc(device, d_output);

    rlt::init(device, rng, 1);
    rlt::init_weights(device, model, rng);
    rlt::reset_optimizer_state(device, optimizer, model);

    rlt::randn(device, X_train, rng);
    rlt::randn(device, X_test, rng);
    fill_targets(device, X_train, y_train);
    fill_targets(device, X_test, y_test);

    for (TI epoch_i=0; epoch_i < 100; epoch_i++){
        for (TI batch_i=0; batch_i < DATASET_SIZE / BATCH_SIZE; batch_i++){
            auto X_batch = rlt::view_range(device, X_train, batch_i*BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            auto y_batch = rlt::view_range(device, y_train,  batch_i*BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            rlt::forward(device, model, X_batch, buffers, rng);
            auto output = rlt::output(device, model);
            T loss = rlt::nn::loss_functions::mse::evaluate(device, output, y_batch);
            rlt::nn::loss_functions::mse::gradient(device, output, y_batch, d_output);
            rlt::zero_gradient(device, model);
            rlt::backward(device, model, X_batch, d_output, buffers);
            rlt::step(device, optimizer, model);
        }

        T test_loss = 0;
        for (TI batch_i=0; batch_i < DATASET_SIZE / BATCH_SIZE; batch_i++){
            auto X_batch = rlt::view_range(device, X_test, batch_i*BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            auto y_batch = rlt::view_range(device, y_test,  batch_i*BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            rlt::forward(device, model, X_batch, buffers, rng);
            auto output = rlt::output(device, model);
            test_loss += rlt::nn::loss_functions::mse::evaluate(device, output, y_batch);
        }
        std::cout << "Test Loss: " << test_loss / (DATASET_SIZE / BATCH_SIZE) << std::endl;
    }
}
