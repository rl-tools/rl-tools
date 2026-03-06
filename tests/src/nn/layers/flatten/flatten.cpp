#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, SHAPE) {
    // Verify that flatten produces the correct output shape
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 2, 7, 7, 64>;
    using CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using LAYER = rlt::nn::layers::flatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;

    static_assert(LAYER::INPUT_HEIGHT == 7);
    static_assert(LAYER::INPUT_WIDTH == 7);
    static_assert(LAYER::INPUT_CHANNELS == 64);
    static_assert(LAYER::OUTPUT_DIM == 7 * 7 * 64);
    static_assert(LAYER::NUM_WEIGHTS == 0);

    using OUTPUT_SHAPE = typename LAYER::OUTPUT_SHAPE;
    static_assert(rlt::length(OUTPUT_SHAPE{}) == 2); // (2, 3136)
    static_assert(rlt::get<0>(OUTPUT_SHAPE{}) == 2);
    static_assert(rlt::get<1>(OUTPUT_SHAPE{}) == 3136);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, SHAPE_WITH_LEADING_DIMS) {
    // Input with extra leading dimensions: (1, BATCH, H, W, C)
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 4, 3, 5, 8>;
    using CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using LAYER = rlt::nn::layers::flatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;

    static_assert(LAYER::INPUT_HEIGHT == 3);
    static_assert(LAYER::INPUT_WIDTH == 5);
    static_assert(LAYER::INPUT_CHANNELS == 8);
    static_assert(LAYER::OUTPUT_DIM == 3 * 5 * 8);

    using OUTPUT_SHAPE = typename LAYER::OUTPUT_SHAPE;
    static_assert(rlt::length(OUTPUT_SHAPE{}) == 3); // (1, 4, 120)
    static_assert(rlt::get<0>(OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(OUTPUT_SHAPE{}) == 4);
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == 120);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, EVALUATE) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 3;
    constexpr TI H = 4, W = 5, C = 2;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH, H, W, C>;
    using CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using LAYER = rlt::nn::layers::flatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER layer;
    rlt::nn::layers::flatten::Buffer buffer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH, H * W * C>>> output;

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::randn(device, input, rng);

    rlt::evaluate(device, layer, input, output, buffer, rng);

    // Verify each element matches: input[b, h, w, c] == output[b, h*W*C + w*C + c]
    for(TI b = 0; b < BATCH; b++){
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                for(TI c = 0; c < C; c++){
                    T input_val = rlt::get(device, input, b, h, w, c);
                    T output_val = rlt::get(device, output, b, (TI)(h * W * C + w * C + c));
                    ASSERT_EQ(input_val, output_val) << "Mismatch at b=" << b << " h=" << h << " w=" << w << " c=" << c;
                }
            }
        }
    }

    rlt::free(device, input);
    rlt::free(device, output);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, FORWARD_GRADIENT) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 2;
    constexpr TI H = 3, W = 4, C = 5;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH, H, W, C>;
    using CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER = rlt::nn::layers::flatten::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER layer;
    rlt::nn::layers::flatten::Buffer buffer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH, H * W * C>>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input;

    rlt::malloc(device, layer);
    rlt::malloc(device, input);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input);
    rlt::randn(device, input, rng);
    rlt::randn(device, d_output, rng);

    // Forward
    rlt::forward(device, layer, input, buffer, rng);

    // Check output matches input data
    auto& out = layer.output;
    for(TI b = 0; b < BATCH; b++){
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                for(TI c = 0; c < C; c++){
                    T input_val = rlt::get(device, input, b, h, w, c);
                    T output_val = rlt::get(device, out, b, (TI)(h * W * C + w * C + c));
                    ASSERT_EQ(input_val, output_val);
                }
            }
        }
    }

    // Backward
    rlt::backward_full(device, layer, input, d_output, d_input, buffer);

    // Check d_input matches d_output: d_input[b,h,w,c] == d_output[b, h*W*C + w*C + c]
    for(TI b = 0; b < BATCH; b++){
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                for(TI c = 0; c < C; c++){
                    T d_input_val = rlt::get(device, d_input, b, h, w, c);
                    T d_output_val = rlt::get(device, d_output, b, (TI)(h * W * C + w * C + c));
                    ASSERT_EQ(d_input_val, d_output_val) << "Backward mismatch at b=" << b << " h=" << h << " w=" << w << " c=" << c;
                }
            }
        }
    }

    rlt::free(device, layer);
    rlt::free(device, input);
    rlt::free(device, d_output);
    rlt::free(device, d_input);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, SEQUENTIAL_CONV_FLATTEN_DENSE) {
    // Test flatten in a sequential model: Conv2d → Flatten → Dense
    // This is the key use case for NatureCNN-style architectures.
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 4;
    constexpr TI H = 8, W = 8, C = 3;
    constexpr TI CONV_OUT_CHANNELS = 16;
    constexpr TI DENSE_OUT = 32;

    // Conv2d: (BATCH, 8, 8, 3) → (BATCH, 6, 6, 16)  [k=3, s=1, p=0]
    // Flatten: (BATCH, 6, 6, 16) → (BATCH, 576)
    // Dense: (BATCH, 576) → (BATCH, 32)
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH, H, W, C>;

    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, CONV_OUT_CHANNELS, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;

    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;

    using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, DENSE_OUT, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV, rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<DENSE>>>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    // Verify output shape at compile time
    // Conv: (1, BATCH, 8, 8, 3) → (1, BATCH, 6, 6, 16)
    // Flatten: (1, BATCH, 6, 6, 16) → (1, BATCH, 576)
    // Dense: (1, BATCH, 576) → (1, BATCH, 32)
    using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
    static_assert(rlt::get<0>(OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(OUTPUT_SHAPE{}) == BATCH);
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == DENSE_OUT);

    MODEL model;
    typename MODEL::template Buffer<true> buffer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::randn(device, d_output, rng);

    // Forward pass should not crash and produce valid output
    rlt::forward(device, model, input, buffer, rng);
    rlt::evaluate(device, model, input, output, buffer, rng);

    ASSERT_FALSE(rlt::is_nan(device, output));

    // Backward pass
    rlt::zero_gradient(device, model);
    rlt::backward_full(device, model, input, d_output, d_input, buffer);

    ASSERT_FALSE(rlt::is_nan(device, d_input));

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, d_output);
    rlt::free(device, d_input);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, NUMERICAL_GRADIENT_CHECK) {
    // Numerical gradient check for a Conv → Flatten → Dense pipeline.
    // Perturb each input element and check that the finite-difference gradient
    // matches the analytical backward gradient.
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 1;
    constexpr TI H = 4, W = 4, C = 1;
    constexpr TI CONV_OC = 2;
    constexpr TI DENSE_OUT = 1;

    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH, H, W, C>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, CONV_OC, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, DENSE_OUT, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV, rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<DENSE>>>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    MODEL model;
    typename MODEL::template Buffer<true> buffer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input, d_input;
    using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output, output_perturbed, d_output;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, d_input);
    rlt::malloc(device, output);
    rlt::malloc(device, output_perturbed);
    rlt::malloc(device, d_output);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);

    // d_output = all ones (sum reduction)
    rlt::set_all(device, d_output, 1.0);

    // Analytical gradient
    rlt::forward(device, model, input, buffer, rng);
    rlt::zero_gradient(device, model);
    rlt::backward_full(device, model, input, d_output, d_input, buffer);

    // Numerical gradient check
    constexpr T eps = 1e-5;
    constexpr T tolerance = 1e-4;

    // Compute baseline output
    rlt::evaluate(device, model, input, output, buffer, rng);
    T baseline = rlt::get(device, output, (TI)0, (TI)0);

    for(TI h = 0; h < H; h++){
        for(TI w = 0; w < W; w++){
            for(TI c = 0; c < C; c++){
                T original = rlt::get(device, input, (TI)0, h, w, c);

                // Perturb +eps
                rlt::set(device, input, original + eps, (TI)0, h, w, c);
                rlt::evaluate(device, model, input, output_perturbed, buffer, rng);
                T out_plus = rlt::get(device, output_perturbed, (TI)0, (TI)0);

                // Perturb -eps
                rlt::set(device, input, original - eps, (TI)0, h, w, c);
                rlt::evaluate(device, model, input, output_perturbed, buffer, rng);
                T out_minus = rlt::get(device, output_perturbed, (TI)0, (TI)0);

                // Restore
                rlt::set(device, input, original, (TI)0, h, w, c);

                T numerical_grad = (out_plus - out_minus) / (2.0 * eps);
                T analytical_grad = rlt::get(device, d_input, (TI)0, h, w, c);

                T abs_error = std::abs(numerical_grad - analytical_grad);
                T scale = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
                T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;

                ASSERT_LT(rel_error, tolerance) << "Gradient mismatch at h=" << h << " w=" << w << " c=" << c
                    << " numerical=" << numerical_grad << " analytical=" << analytical_grad;
            }
        }
    }

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, d_input);
    rlt::free(device, output);
    rlt::free(device, output_perturbed);
    rlt::free(device, d_output);
}

TEST(RL_TOOLS_NN_LAYERS_FLATTEN, NATURECNN_ARCHITECTURE) {
    // Test the full NatureCNN architecture: 3 conv layers → flatten → FC
    // 84x84x3 → Conv(32,8x8,s4) → 20x20x32 → Conv(64,4x4,s2) → 9x9x64 → Conv(64,3x3,s1) → 7x7x64 → Flatten → 3136 → Dense(512) → Dense(2)
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 2;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH, 84, 84, 3>;

    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 8, 8, 4, 4, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 4, 4, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using FC_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 512, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using FC = rlt::nn::layers::dense::BindConfiguration<FC_CONFIG>;
    using OUT_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using OUT = rlt::nn::layers::dense::BindConfiguration<OUT_CONFIG>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV1,
        rlt::nn_models::sequential::Module<CONV2,
        rlt::nn_models::sequential::Module<CONV3,
        rlt::nn_models::sequential::Module<FLATTEN,
        rlt::nn_models::sequential::Module<FC,
        rlt::nn_models::sequential::Module<OUT>>>>>>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    // Verify shapes at compile time
    using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
    static_assert(rlt::get<0>(OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(OUTPUT_SHAPE{}) == BATCH);
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == 2);

    MODEL model;
    typename MODEL::template Buffer<true> buffer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output, d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::randn(device, d_output, rng);

    // Forward
    rlt::evaluate(device, model, input, output, buffer, rng);
    ASSERT_FALSE(rlt::is_nan(device, output));

    // Forward (training path)
    rlt::forward(device, model, input, buffer, rng);
    ASSERT_FALSE(rlt::is_nan(device, model));

    // Backward
    rlt::zero_gradient(device, model);
    rlt::backward_full(device, model, input, d_output, d_input, buffer);
    ASSERT_FALSE(rlt::is_nan(device, d_input));

    // Sanity check: output should change if we change input
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output2;
    rlt::malloc(device, output2);
    rlt::set(device, input, rlt::get(device, input, (TI)0, (TI)0, (TI)42, (TI)42, (TI)0) + 1.0, (TI)0, (TI)0, (TI)42, (TI)42, (TI)0);
    rlt::evaluate(device, model, input, output2, buffer, rng);
    T diff = rlt::abs_diff(device, output, output2);
    ASSERT_GT(diff, 0) << "Output should change when input changes";

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, output2);
    rlt::free(device, d_output);
    rlt::free(device, d_input);
}
