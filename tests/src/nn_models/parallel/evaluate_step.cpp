#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>

TEST(RL_TOOLS_NN_MODELS_PARALLEL, EVALUATE_STEP_GRU_HEAD){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI SEQUENCE_LENGTH = 10;
    constexpr TI BATCH_SIZE = 4;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 4>;
    using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
    using HEAD_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<GRU, HEAD_DENSE>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using MODEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, HEAD_MODULE, BRANCH_A, BRANCH_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    MODEL model;
    typename MODEL::Buffer<> buffer;
    typename MODEL::State<> state;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, state);
    rlt::init_weights(device, model, rng);
    rlt::reset(device, model, state, rng);

    using STEP_INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;
    using STEP_INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using STEP_OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, 2>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEP_INPUT_SHAPE_A>> step_input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEP_INPUT_SHAPE_B>> step_input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEP_OUTPUT_SHAPE>> step_output;
    rlt::malloc(device, step_input_a);
    rlt::malloc(device, step_input_b);
    rlt::malloc(device, step_output);

    for(TI step = 0; step < SEQUENCE_LENGTH; step++){
        rlt::randn(device, step_input_a, rng);
        rlt::randn(device, step_input_b, rng);
        auto inputs = rlt::nn_models::parallel::pack_inputs(step_input_a, step_input_b);
        rlt::evaluate_step(device, model, inputs, state, step_output, buffer, rng);
    }

    T sum = 0;
    for(TI i = 0; i < BATCH_SIZE; i++){
        for(TI j = 0; j < 2; j++){
            sum += rlt::math::abs(device.math, rlt::get(device, step_output, i, j));
        }
    }
    ASSERT_GT(sum, 0);

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, state);
    rlt::free(device, step_input_a);
    rlt::free(device, step_input_b);
    rlt::free(device, step_output);
}
