#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_STATIC_SHAPES){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    static_assert(rlt::get<0>(typename PARALLEL::OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(typename PARALLEL::OUTPUT_SHAPE{}) == BATCH_SIZE);
    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 14); // 8 + 6
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_STATIC_SHAPES_MULTI_LAYER){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 10>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>;

    using LAYER_A1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A1 = rlt::nn::layers::dense::BindConfiguration<LAYER_A1_CONFIG>;
    using LAYER_A2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_A2 = rlt::nn::layers::dense::BindConfiguration<LAYER_A2_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A1, rlt::nn_models::sequential::Module<LAYER_A2>>;

    using LAYER_B1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 12, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B1 = rlt::nn::layers::dense::BindConfiguration<LAYER_B1_CONFIG>;
    using LAYER_B2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_B2 = rlt::nn::layers::dense::BindConfiguration<LAYER_B2_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B1, rlt::nn_models::sequential::Module<LAYER_B2>>;

    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 7); // 4 + 3
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_EVALUATE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, output);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);

    rlt::evaluate(device, model, input_a, input_b, output, buffer, rng);

    // Verify output is not all zeros
    T sum = 0;
    for(TI i = 0; i < BATCH_SIZE; i++){
        for(TI j = 0; j < rlt::get<2>(OUTPUT_SHAPE{}); j++){
            sum += rlt::math::abs(device.math, rlt::get(device, output, 0, i, j));
        }
    }
    ASSERT_GT(sum, 0);

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, output);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_FORWARD){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output_forward;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output_evaluate;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, output_forward);
    rlt::malloc(device, output_evaluate);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);

    rlt::forward(device, model, input_a, input_b, output_forward, buffer, rng);

    // Also run evaluate on a Forward-capability copy to compare
    using PARALLEL_FORWARD = typename PARALLEL::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
    PARALLEL_FORWARD model_forward;
    typename PARALLEL_FORWARD::Buffer<> buffer_forward;
    rlt::malloc(device, model_forward);
    rlt::malloc(device, buffer_forward);
    rlt::copy(device, device, model, model_forward);

    rlt::evaluate(device, model_forward, input_a, input_b, output_evaluate, buffer_forward, rng);

    auto diff = rlt::abs_diff(device, output_forward, output_evaluate);
    std::cout << "Forward vs evaluate abs diff: " << diff << std::endl;
    ASSERT_LT(diff, 1e-10);

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, model_forward);
    rlt::free(device, buffer_forward);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, output_forward);
    rlt::free(device, output_evaluate);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_GRADIENT_CHECK){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 1;
    constexpr T EPSILON = 1e-5;
    constexpr T THRESHOLD = 1e-3;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;

    using LAYER_A1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A1 = rlt::nn::layers::dense::BindConfiguration<LAYER_A1_CONFIG>;
    using LAYER_A2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_A2 = rlt::nn::layers::dense::BindConfiguration<LAYER_A2_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A1, rlt::nn_models::sequential::Module<LAYER_A2>>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model, model_pert;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, model_pert);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    constexpr TI OUTPUT_DIM = rlt::get<2>(OUTPUT_SHAPE{});
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> d_input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> d_input_b;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input_a);
    rlt::malloc(device, d_input_b);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::set_all(device, d_output, (T)0);
    // Set d_output to sum of all outputs (gradient of sum)
    for(TI j = 0; j < OUTPUT_DIM; j++){
        rlt::set(device, d_output, (T)1, 0, 0, j);
    }

    // Forward + backward
    rlt::forward(device, model, input_a, input_b, buffer, rng);
    rlt::zero_gradient(device, model);
    rlt::backward_full(device, model, input_a, input_b, d_output, d_input_a, d_input_b, buffer);

    // Finite-difference check for d_input_a
    auto compute_loss = [&](PARALLEL& m, auto& ia, auto& ib) -> T {
        rlt::forward(device, m, ia, ib, buffer, rng);
        auto out = rlt::output(device, m);
        T loss = 0;
        for(TI j = 0; j < OUTPUT_DIM; j++){
            loss += rlt::get(device, out, 0, 0, j);
        }
        return loss;
    };

    constexpr TI INPUT_DIM_A = rlt::get<2>(INPUT_SHAPE_A{});
    for(TI i = 0; i < INPUT_DIM_A; i++){
        T original = rlt::get(device, input_a, 0, 0, i);
        rlt::copy(device, device, model, model_pert);

        rlt::set(device, input_a, original + EPSILON, 0, 0, i);
        T loss_plus = compute_loss(model_pert, input_a, input_b);

        rlt::copy(device, device, model, model_pert);
        rlt::set(device, input_a, original - EPSILON, 0, 0, i);
        T loss_minus = compute_loss(model_pert, input_a, input_b);

        rlt::set(device, input_a, original, 0, 0, i);

        T numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON);
        T analytical_grad = rlt::get(device, d_input_a, 0, 0, i);

        T abs_error = rlt::math::abs(device.math, numerical_grad - analytical_grad);
        T scale = rlt::math::max(device.math, rlt::math::abs(device.math, numerical_grad), rlt::math::abs(device.math, analytical_grad));
        T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;
        ASSERT_LT(rel_error, THRESHOLD) << "d_input_a gradient mismatch at index " << i
            << ": numerical=" << numerical_grad << " analytical=" << analytical_grad;
    }

    // Finite-difference check for d_input_b
    constexpr TI INPUT_DIM_B = rlt::get<2>(INPUT_SHAPE_B{});
    for(TI i = 0; i < INPUT_DIM_B; i++){
        T original = rlt::get(device, input_b, 0, 0, i);
        rlt::copy(device, device, model, model_pert);

        rlt::set(device, input_b, original + EPSILON, 0, 0, i);
        T loss_plus = compute_loss(model_pert, input_a, input_b);

        rlt::copy(device, device, model, model_pert);
        rlt::set(device, input_b, original - EPSILON, 0, 0, i);
        T loss_minus = compute_loss(model_pert, input_a, input_b);

        rlt::set(device, input_b, original, 0, 0, i);

        T numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON);
        T analytical_grad = rlt::get(device, d_input_b, 0, 0, i);

        T abs_error = rlt::math::abs(device.math, numerical_grad - analytical_grad);
        T scale = rlt::math::max(device.math, rlt::math::abs(device.math, numerical_grad), rlt::math::abs(device.math, analytical_grad));
        T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;
        ASSERT_LT(rel_error, THRESHOLD) << "d_input_b gradient mismatch at index " << i
            << ": numerical=" << numerical_grad << " analytical=" << analytical_grad;
    }

    rlt::free(device, model);
    rlt::free(device, model_pert);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, d_output);
    rlt::free(device, d_input_a);
    rlt::free(device, d_input_b);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_COPY_ABS_DIFF){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, 1, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, 1, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model_a, model_b;
    rlt::malloc(device, model_a);
    rlt::malloc(device, model_b);
    rlt::init_weights(device, model_a, rng);
    rlt::copy(device, device, model_a, model_b);

    auto diff = rlt::abs_diff(device, model_a, model_b);
    ASSERT_EQ(diff, 0);

    rlt::free(device, model_a);
    rlt::free(device, model_b);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_IS_NAN){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, 1, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, 1, 3>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model;
    rlt::malloc(device, model);
    rlt::init_weights(device, model, rng);

    ASSERT_FALSE(rlt::is_nan(device, model));

    rlt::free(device, model);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_TRAINING){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;
    constexpr TI NUM_STEPS = 500;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using LAYER_A2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_A2 = rlt::nn::layers::dense::BindConfiguration<LAYER_A2_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A, rlt::nn_models::sequential::Module<LAYER_A2>>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using LAYER_B2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_B2 = rlt::nn::layers::dense::BindConfiguration<LAYER_B2_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B, rlt::nn_models::sequential::Module<LAYER_B2>>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL model;
    typename PARALLEL::Buffer<> buffer;
    rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>> optimizer;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, optimizer);
    rlt::init(device, optimizer);
    rlt::init_weights(device, model, rng);
    rlt::reset_optimizer_state(device, optimizer, model);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == 2); // 1 + 1

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> target;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, target);
    rlt::malloc(device, d_output);

    T initial_loss = 0;
    T final_loss = 0;

    for(TI step = 0; step < NUM_STEPS; step++){
        rlt::randn(device, input_a, rng);
        rlt::randn(device, input_b, rng);

        // Target: first output = sum(input_a), second output = sum(input_b)
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T sum_a = 0, sum_b = 0;
            for(TI j = 0; j < 2; j++){
                sum_a += rlt::get(device, input_a, 0, bi, j);
                sum_b += rlt::get(device, input_b, 0, bi, j);
            }
            rlt::set(device, target, sum_a, 0, bi, 0);
            rlt::set(device, target, sum_b, 0, bi, 1);
        }

        rlt::forward(device, model, input_a, input_b, buffer, rng);
        auto model_output = rlt::output(device, model);

        // Compute MSE loss and d_output
        T loss = 0;
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI j = 0; j < 2; j++){
                T diff = rlt::get(device, model_output, 0, bi, j) - rlt::get(device, target, 0, bi, j);
                loss += diff * diff;
                rlt::set(device, d_output, 2 * diff / (BATCH_SIZE * 2), 0, bi, j);
            }
        }
        loss /= (BATCH_SIZE * 2);

        if(step == 0) initial_loss = loss;
        if(step == NUM_STEPS - 1) final_loss = loss;

        rlt::zero_gradient(device, model);
        rlt::backward(device, model, input_a, input_b, d_output, buffer);
        rlt::step(device, optimizer, model);
    }

    std::cout << "Training loss: " << initial_loss << " -> " << final_loss << std::endl;
    ASSERT_LT(final_loss, initial_loss * 0.1) << "Training should reduce loss significantly";

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, optimizer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, target);
    rlt::free(device, d_output);
}
