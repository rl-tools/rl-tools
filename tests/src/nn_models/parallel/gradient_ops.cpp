#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>

namespace {
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    constexpr TI SEQUENCE_LENGTH = 3;
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
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD_MODULE, BRANCH_A, BRANCH_B>;

    // hand-rolled walk over every gradient tensor of MODEL, applied to N models in lockstep, so the
    // recursive add_gradient/copy_gradient are checked against per-tensor operations
    template<typename F, typename... MODELS>
    void for_each_gradient(F&& f, MODELS&... models){
        auto apply = [&](auto&& accessor){ f(accessor(models)...); };
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<0>(m.pipelines)).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<0>(m.pipelines)).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<1>(m.pipelines)).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<1>(m.pipelines)).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).weights_input.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).biases_input.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).weights_hidden.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).biases_hidden.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).initial_hidden_state.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<1>(m.head).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<1>(m.head).biases.gradient; });
    }
}

namespace {
    constexpr TI CONV_BATCH_SIZE = 2;
    using CONV_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, CONV_BATCH_SIZE, 6, 6, 3>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using CONV_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>>;
    using CONV_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONV_DENSE = rlt::nn::layers::dense::BindConfiguration<CONV_DENSE_CONFIG>;
    using CONV_MODULE = rlt::nn_models::sequential::Module<CONV, CONV_FLATTEN, CONV_DENSE>;
    using CONV_MODEL = rlt::nn_models::sequential::Build<CAPABILITY, CONV_MODULE, CONV_INPUT_SHAPE>;

    template<typename F, typename... MODELS>
    void for_each_conv_gradient(F&& f, MODELS&... models){
        auto apply = [&](auto&& accessor){ f(accessor(models)...); };
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<2>(m).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<2>(m).biases.gradient; });
    }
}

// conv2d gradients are 4-D tensors, which exercises the rank > 2 path of the elementwise tensor ops
TEST(RL_TOOLS_NN_MODELS_PARALLEL, ADD_GRADIENT_CONV2D){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 5);

    CONV_MODEL a, b, expected;
    rlt::malloc(device, a);
    rlt::malloc(device, b);
    rlt::malloc(device, expected);
    rlt::init_weights(device, a, rng);
    rlt::init_weights(device, b, rng);
    for_each_conv_gradient([&](auto& ta, auto& tb){ rlt::randn(device, ta, rng); rlt::randn(device, tb, rng); }, a, b);
    for_each_conv_gradient([&](auto& ta, auto& tb, auto& te){ rlt::copy(device, device, ta, te); rlt::add(device, tb, te); }, a, b, expected);

    rlt::add_gradient(device, b, a);

    for_each_conv_gradient([&](auto& ta, auto& te){ EXPECT_EQ(rlt::abs_diff(device, ta, te), (T)0); }, a, expected);

    rlt::free(device, a);
    rlt::free(device, b);
    rlt::free(device, expected);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, ADD_GRADIENT){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 7);

    MODEL a, b, b_snapshot, expected;
    rlt::malloc(device, a);
    rlt::malloc(device, b);
    rlt::malloc(device, b_snapshot);
    rlt::malloc(device, expected);
    rlt::init_weights(device, a, rng);
    rlt::init_weights(device, b, rng);
    rlt::init_weights(device, expected, rng);
    for_each_gradient([&](auto& ta, auto& tb){ rlt::randn(device, ta, rng); rlt::randn(device, tb, rng); }, a, b);
    rlt::copy(device, device, b, b_snapshot);
    for_each_gradient([&](auto& ta, auto& tb, auto& te){ rlt::copy(device, device, ta, te); rlt::add(device, tb, te); }, a, b, expected);

    rlt::add_gradient(device, b, a);

    TI count = 0;
    for_each_gradient([&](auto& ta, auto& te){ EXPECT_EQ(rlt::abs_diff(device, ta, te), (T)0); count++; }, a, expected);
    EXPECT_EQ(count, 11);
    EXPECT_EQ(rlt::abs_diff(device, b, b_snapshot), (T)0);

    rlt::free(device, a);
    rlt::free(device, b);
    rlt::free(device, b_snapshot);
    rlt::free(device, expected);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, COPY_GRADIENT){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 11);

    MODEL source, target, target_snapshot;
    rlt::malloc(device, source);
    rlt::malloc(device, target);
    rlt::malloc(device, target_snapshot);
    rlt::init_weights(device, source, rng);
    rlt::init_weights(device, target, rng);
    for_each_gradient([&](auto& ts, auto& tt){ rlt::randn(device, ts, rng); rlt::randn(device, tt, rng); }, source, target);
    rlt::copy(device, device, target, target_snapshot);

    rlt::copy_gradient(device, device, source, target);

    for_each_gradient([&](auto& ts, auto& tt){ EXPECT_EQ(rlt::abs_diff(device, ts, tt), (T)0); }, source, target);
    // everything but the gradient is untouched: zero the (now equal) gradients and compare against the snapshot
    for_each_gradient([&](auto& tt, auto& tsnap){ rlt::set_all(device, tt, (T)0); rlt::set_all(device, tsnap, (T)0); }, target, target_snapshot);
    EXPECT_EQ(rlt::abs_diff(device, target, target_snapshot), (T)0);
    auto& source_weights = rlt::nn_models::sequential::layer<0>(rlt::get<0>(source.pipelines)).weights.parameters;
    auto& target_weights = rlt::nn_models::sequential::layer<0>(rlt::get<0>(target.pipelines)).weights.parameters;
    EXPECT_GT(rlt::abs_diff(device, source_weights, target_weights), (T)0);

    rlt::free(device, source);
    rlt::free(device, target);
    rlt::free(device, target_snapshot);
    rlt::free(device, rng);
}
