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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B>;

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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 7); // 4 + 3
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_THREE_BRANCHES_SHAPES){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B, BRANCH_C>;

    static_assert(rlt::get<0>(typename PARALLEL::OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(typename PARALLEL::OUTPUT_SHAPE{}) == BATCH_SIZE);
    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 12); // 4 + 6 + 2
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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B>;

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

    auto inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    rlt::evaluate(device, model, inputs, output, buffer, rng);

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

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_THREE_BRANCHES_EVALUATE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 3;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B, BRANCH_C>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 9); // 2 + 3 + 4

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
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, output);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);

    auto inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::evaluate(device, model, inputs, output, buffer, rng);

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
    rlt::free(device, input_c);
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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH_A, BRANCH_B>;

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

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    rlt::forward(device, model, fwd_inputs, output_forward, buffer, rng);

    using PARALLEL_FORWARD = typename PARALLEL::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
    PARALLEL_FORWARD model_forward;
    typename PARALLEL_FORWARD::Buffer<> buffer_forward;
    rlt::malloc(device, model_forward);
    rlt::malloc(device, buffer_forward);
    rlt::copy(device, device, model, model_forward);

    auto eval_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    rlt::evaluate(device, model_forward, eval_inputs, output_evaluate, buffer_forward, rng);

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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH_A, BRANCH_B>;

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
    for(TI j = 0; j < OUTPUT_DIM; j++){
        rlt::set(device, d_output, (T)1, 0, 0, j);
    }

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    rlt::forward(device, model, fwd_inputs, buffer, rng);
    rlt::zero_gradient(device, model);
    auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    auto d_inputs = rlt::nn_models::parallel::pack_inputs(d_input_a, d_input_b);
    rlt::backward_full(device, model, bwd_inputs, d_output, d_inputs, buffer);

    auto compute_loss = [&](PARALLEL& m, auto& ia, auto& ib) -> T {
        auto inputs_pack = rlt::nn_models::parallel::pack_inputs(ia, ib);
        rlt::forward(device, m, inputs_pack, buffer, rng);
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

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_THREE_BRANCHES_GRADIENT_CHECK){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 1;
    constexpr T EPSILON = 1e-5;
    constexpr T THRESHOLD = 1e-3;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH_A, BRANCH_B, BRANCH_C>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    PARALLEL model, model_pert;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, model_pert);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    constexpr TI OUTPUT_DIM = rlt::get<2>(OUTPUT_SHAPE{});
    static_assert(OUTPUT_DIM == 7); // 2 + 3 + 2

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> d_input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> d_input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> d_input_c;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input_a);
    rlt::malloc(device, d_input_b);
    rlt::malloc(device, d_input_c);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);
    rlt::set_all(device, d_output, (T)0);
    for(TI j = 0; j < OUTPUT_DIM; j++){
        rlt::set(device, d_output, (T)1, 0, 0, j);
    }

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::forward(device, model, fwd_inputs, buffer, rng);
    rlt::zero_gradient(device, model);
    auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    auto d_inputs = rlt::nn_models::parallel::pack_inputs(d_input_a, d_input_b, d_input_c);
    rlt::backward_full(device, model, bwd_inputs, d_output, d_inputs, buffer);

    auto compute_loss = [&](PARALLEL& m, auto& ia, auto& ib, auto& ic) -> T {
        auto inputs_pack = rlt::nn_models::parallel::pack_inputs(ia, ib, ic);
        rlt::forward(device, m, inputs_pack, buffer, rng);
        auto out = rlt::output(device, m);
        T loss = 0;
        for(TI j = 0; j < OUTPUT_DIM; j++){
            loss += rlt::get(device, out, 0, 0, j);
        }
        return loss;
    };

    auto check_gradient = [&](auto& input_tensor, auto& d_input_tensor, const char* name){
        constexpr TI INPUT_DIM = rlt::get<2>(typename rlt::utils::typing::remove_reference_t<decltype(input_tensor)>::SPEC::SHAPE{});
        for(TI i = 0; i < INPUT_DIM; i++){
            T original = rlt::get(device, input_tensor, 0, 0, i);
            rlt::copy(device, device, model, model_pert);

            rlt::set(device, input_tensor, original + EPSILON, 0, 0, i);
            T loss_plus = compute_loss(model_pert, input_a, input_b, input_c);

            rlt::copy(device, device, model, model_pert);
            rlt::set(device, input_tensor, original - EPSILON, 0, 0, i);
            T loss_minus = compute_loss(model_pert, input_a, input_b, input_c);

            rlt::set(device, input_tensor, original, 0, 0, i);

            T numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON);
            T analytical_grad = rlt::get(device, d_input_tensor, 0, 0, i);

            T abs_error = rlt::math::abs(device.math, numerical_grad - analytical_grad);
            T scale = rlt::math::max(device.math, rlt::math::abs(device.math, numerical_grad), rlt::math::abs(device.math, analytical_grad));
            T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;
            ASSERT_LT(rel_error, THRESHOLD) << name << " gradient mismatch at index " << i
                << ": numerical=" << numerical_grad << " analytical=" << analytical_grad;
        }
    };

    check_gradient(input_a, d_input_a, "d_input_a");
    check_gradient(input_b, d_input_b, "d_input_b");
    check_gradient(input_c, d_input_c, "d_input_c");

    rlt::free(device, model);
    rlt::free(device, model_pert);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, input_c);
    rlt::free(device, d_output);
    rlt::free(device, d_input_a);
    rlt::free(device, d_input_b);
    rlt::free(device, d_input_c);
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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>, void, BRANCH_A, BRANCH_B>;

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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>, void, BRANCH_A, BRANCH_B>;

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

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH_A, BRANCH_B>;

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

        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T sum_a = 0, sum_b = 0;
            for(TI j = 0; j < 2; j++){
                sum_a += rlt::get(device, input_a, 0, bi, j);
                sum_b += rlt::get(device, input_b, 0, bi, j);
            }
            rlt::set(device, target, sum_a, 0, bi, 0);
            rlt::set(device, target, sum_b, 0, bi, 1);
        }

        auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
        rlt::forward(device, model, fwd_inputs, buffer, rng);
        auto model_output = rlt::output(device, model);

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
        auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
        rlt::backward(device, model, bwd_inputs, d_output, buffer);
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

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_HEAD_SHAPES){
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

    using HEAD_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD = rlt::nn::layers::dense::BindConfiguration<HEAD_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, HEAD_MODULE, BRANCH_A, BRANCH_B>;

    static_assert(rlt::get<0>(typename PARALLEL::OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<1>(typename PARALLEL::OUTPUT_SHAPE{}) == BATCH_SIZE);
    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 2);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_HEAD_EVALUATE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using HEAD_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD = rlt::nn::layers::dense::BindConfiguration<HEAD_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, HEAD_MODULE, BRANCH_A, BRANCH_B>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

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

    auto inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
    rlt::evaluate(device, model, inputs, output, buffer, rng);

    T sum = 0;
    for(TI i = 0; i < BATCH_SIZE; i++){
        sum += rlt::math::abs(device.math, rlt::get(device, output, 0, i, 0));
    }
    ASSERT_GT(sum, 0) << "Output should be non-zero after evaluation with HEAD";

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, output);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_HEAD_TRAINING){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;
    constexpr TI NUM_STEPS = 1000;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using HEAD_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_LAYER = rlt::nn::layers::dense::BindConfiguration<HEAD_LAYER_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD_LAYER>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD_MODULE, BRANCH_A, BRANCH_B>;

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
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == 1);

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

        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T sum_all = 0;
            for(TI j = 0; j < 2; j++){
                sum_all += rlt::get(device, input_a, 0, bi, j);
                sum_all += rlt::get(device, input_b, 0, bi, j);
            }
            rlt::set(device, target, sum_all, 0, bi, 0);
        }

        auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
        rlt::forward(device, model, fwd_inputs, buffer, rng);
        auto model_output = rlt::output(device, model);

        T loss = 0;
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T diff = rlt::get(device, model_output, 0, bi, 0) - rlt::get(device, target, 0, bi, 0);
            loss += diff * diff;
            rlt::set(device, d_output, 2 * diff / BATCH_SIZE, 0, bi, 0);
        }
        loss /= BATCH_SIZE;

        if(step == 0) initial_loss = loss;
        if(step == NUM_STEPS - 1) final_loss = loss;

        rlt::zero_gradient(device, model);
        auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b);
        rlt::backward(device, model, bwd_inputs, d_output, buffer);
        rlt::step(device, optimizer, model);
    }

    std::cout << "HEAD Training loss: " << initial_loss << " -> " << final_loss << std::endl;
    ASSERT_LT(final_loss, initial_loss * 0.2) << "HEAD training should reduce loss significantly";

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, optimizer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, target);
    rlt::free(device, d_output);
}

// ======================== Single branch (N=1) edge case ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_SINGLE_BRANCH_SHAPES){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>;

    using LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER = rlt::nn::layers::dense::BindConfiguration<LAYER_CONFIG>;
    using MODULE = rlt::nn_models::sequential::Module<LAYER>;

    using BRANCH = rlt::nn_models::parallel::Branch<MODULE, INPUT_SHAPE>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 5);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_SINGLE_BRANCH_GRADIENT_CHECK){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 1;
    constexpr T EPSILON = 1e-5;
    constexpr T THRESHOLD = 1e-3;

    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>;
    using LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER = rlt::nn::layers::dense::BindConfiguration<LAYER_CONFIG>;
    using MODULE = rlt::nn_models::sequential::Module<LAYER>;

    using BRANCH = rlt::nn_models::parallel::Branch<MODULE, INPUT_SHAPE>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    PARALLEL model, model_pert;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, model_pert);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    constexpr TI OUTPUT_DIM = rlt::get<2>(OUTPUT_SHAPE{});

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE, true>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE, true>> d_input;

    rlt::malloc(device, input);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input);

    rlt::randn(device, input, rng);
    rlt::set_all(device, d_output, (T)0);
    for(TI j = 0; j < OUTPUT_DIM; j++){
        rlt::set(device, d_output, (T)1, 0, 0, j);
    }

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input);
    rlt::forward(device, model, fwd_inputs, buffer, rng);
    rlt::zero_gradient(device, model);
    auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input);
    auto d_inputs = rlt::nn_models::parallel::pack_inputs(d_input);
    rlt::backward_full(device, model, bwd_inputs, d_output, d_inputs, buffer);

    constexpr TI INPUT_DIM = rlt::get<2>(INPUT_SHAPE{});
    for(TI i = 0; i < INPUT_DIM; i++){
        T original = rlt::get(device, input, 0, 0, i);
        rlt::copy(device, device, model, model_pert);

        rlt::set(device, input, original + EPSILON, 0, 0, i);
        auto fwd_p = rlt::nn_models::parallel::pack_inputs(input);
        rlt::forward(device, model_pert, fwd_p, buffer, rng);
        auto out_p = rlt::output(device, model_pert);
        T loss_plus = 0;
        for(TI j = 0; j < OUTPUT_DIM; j++) loss_plus += rlt::get(device, out_p, 0, 0, j);

        rlt::copy(device, device, model, model_pert);
        rlt::set(device, input, original - EPSILON, 0, 0, i);
        auto fwd_m = rlt::nn_models::parallel::pack_inputs(input);
        rlt::forward(device, model_pert, fwd_m, buffer, rng);
        auto out_m = rlt::output(device, model_pert);
        T loss_minus = 0;
        for(TI j = 0; j < OUTPUT_DIM; j++) loss_minus += rlt::get(device, out_m, 0, 0, j);

        rlt::set(device, input, original, 0, 0, i);

        T numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON);
        T analytical_grad = rlt::get(device, d_input, 0, 0, i);

        T abs_error = rlt::math::abs(device.math, numerical_grad - analytical_grad);
        T scale = rlt::math::max(device.math, rlt::math::abs(device.math, numerical_grad), rlt::math::abs(device.math, analytical_grad));
        T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;
        ASSERT_LT(rel_error, THRESHOLD) << "Single-branch gradient mismatch at index " << i
            << ": numerical=" << numerical_grad << " analytical=" << analytical_grad;
    }

    rlt::free(device, model);
    rlt::free(device, model_pert);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, d_output);
    rlt::free(device, d_input);
}

// ======================== Four branches (N=4) ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_FOUR_BRANCHES_EVALUATE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using LAYER_D_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_D = rlt::nn::layers::dense::BindConfiguration<LAYER_D_CONFIG>;
    using MODULE_D = rlt::nn_models::sequential::Module<LAYER_D>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 6>>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 8>>;
    using BRANCH_D = rlt::nn_models::parallel::Branch<MODULE_D, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B, BRANCH_C, BRANCH_D>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 14); // 3 + 5 + 2 + 4

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
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 6>, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 8>, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>, true>> input_d;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, input_d);
    rlt::malloc(device, output);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);
    rlt::randn(device, input_d, rng);

    auto inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c, input_d);
    rlt::evaluate(device, model, inputs, output, buffer, rng);

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
    rlt::free(device, input_c);
    rlt::free(device, input_d);
    rlt::free(device, output);
}

// ======================== Concatenation correctness ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_CONCATENATION_CORRECTNESS){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 3;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B, BRANCH_C>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 7);

    PARALLEL model;
    typename PARALLEL::Buffer<> buffer;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;
    static_assert(rlt::get<2>(OUTPUT_SHAPE{}) == 9); // 2 + 3 + 4

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output_parallel;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>, true>> output_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>, true>> output_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>, true>> output_c;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, output_parallel);
    rlt::malloc(device, output_a);
    rlt::malloc(device, output_b);
    rlt::malloc(device, output_c);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);

    auto inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::evaluate(device, model, inputs, output_parallel, buffer, rng);

    // Run each pipeline individually
    typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<0>(model.pipelines))>::template Buffer<> buf_a;
    typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<1>(model.pipelines))>::template Buffer<> buf_b;
    typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<2>(model.pipelines))>::template Buffer<> buf_c;
    rlt::malloc(device, buf_a);
    rlt::malloc(device, buf_b);
    rlt::malloc(device, buf_c);
    rlt::evaluate(device, rlt::get<0>(model.pipelines), input_a, output_a, buf_a, rng);
    rlt::evaluate(device, rlt::get<1>(model.pipelines), input_b, output_b, buf_b, rng);
    rlt::evaluate(device, rlt::get<2>(model.pipelines), input_c, output_c, buf_c, rng);

    for(TI bi = 0; bi < BATCH_SIZE; bi++){
        for(TI j = 0; j < 2; j++){
            ASSERT_NEAR(rlt::get(device, output_parallel, 0, bi, j), rlt::get(device, output_a, 0, bi, j), 1e-10)
                << "Branch A mismatch at batch=" << bi << " col=" << j;
        }
        for(TI j = 0; j < 3; j++){
            ASSERT_NEAR(rlt::get(device, output_parallel, 0, bi, 2 + j), rlt::get(device, output_b, 0, bi, j), 1e-10)
                << "Branch B mismatch at batch=" << bi << " col=" << j;
        }
        for(TI j = 0; j < 4; j++){
            ASSERT_NEAR(rlt::get(device, output_parallel, 0, bi, 5 + j), rlt::get(device, output_c, 0, bi, j), 1e-10)
                << "Branch C mismatch at batch=" << bi << " col=" << j;
        }
    }

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, buf_a);
    rlt::free(device, buf_b);
    rlt::free(device, buf_c);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, input_c);
    rlt::free(device, output_parallel);
    rlt::free(device, output_a);
    rlt::free(device, output_b);
    rlt::free(device, output_c);
}

// ======================== 3-branch with HEAD: gradient check ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_THREE_BRANCHES_HEAD_GRADIENT_CHECK){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 1;
    constexpr T EPSILON = 1e-5;
    constexpr T THRESHOLD = 1e-3;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 4>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using HEAD_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD = rlt::nn::layers::dense::BindConfiguration<HEAD_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD_MODULE, BRANCH_A, BRANCH_B, BRANCH_C>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 1);

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 123);

    PARALLEL model, model_pert;
    typename PARALLEL::Buffer<> buffer;

    rlt::malloc(device, model);
    rlt::malloc(device, model_pert);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);

    using OUTPUT_SHAPE = typename PARALLEL::OUTPUT_SHAPE;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> d_input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> d_input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> d_input_c;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, d_output);
    rlt::malloc(device, d_input_a);
    rlt::malloc(device, d_input_b);
    rlt::malloc(device, d_input_c);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);
    rlt::set(device, d_output, (T)1, 0, 0, 0);

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::forward(device, model, fwd_inputs, buffer, rng);
    rlt::zero_gradient(device, model);
    auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    auto d_inputs = rlt::nn_models::parallel::pack_inputs(d_input_a, d_input_b, d_input_c);
    rlt::backward_full(device, model, bwd_inputs, d_output, d_inputs, buffer);

    auto compute_loss = [&](PARALLEL& m, auto& ia, auto& ib, auto& ic) -> T {
        auto pack = rlt::nn_models::parallel::pack_inputs(ia, ib, ic);
        rlt::forward(device, m, pack, buffer, rng);
        auto out = rlt::output(device, m);
        return rlt::get(device, out, 0, 0, 0);
    };

    auto check_gradient = [&](auto& input_tensor, auto& d_input_tensor, const char* name){
        constexpr TI INPUT_DIM = rlt::get<2>(typename rlt::utils::typing::remove_reference_t<decltype(input_tensor)>::SPEC::SHAPE{});
        for(TI i = 0; i < INPUT_DIM; i++){
            T original = rlt::get(device, input_tensor, 0, 0, i);
            rlt::copy(device, device, model, model_pert);

            rlt::set(device, input_tensor, original + EPSILON, 0, 0, i);
            T loss_plus = compute_loss(model_pert, input_a, input_b, input_c);

            rlt::copy(device, device, model, model_pert);
            rlt::set(device, input_tensor, original - EPSILON, 0, 0, i);
            T loss_minus = compute_loss(model_pert, input_a, input_b, input_c);

            rlt::set(device, input_tensor, original, 0, 0, i);

            T numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON);
            T analytical_grad = rlt::get(device, d_input_tensor, 0, 0, i);

            T abs_error = rlt::math::abs(device.math, numerical_grad - analytical_grad);
            T scale = rlt::math::max(device.math, rlt::math::abs(device.math, numerical_grad), rlt::math::abs(device.math, analytical_grad));
            T rel_error = scale > 1e-7 ? abs_error / scale : abs_error;
            ASSERT_LT(rel_error, THRESHOLD) << name << " gradient mismatch at index " << i
                << ": numerical=" << numerical_grad << " analytical=" << analytical_grad;
        }
    };

    check_gradient(input_a, d_input_a, "d_input_a");
    check_gradient(input_b, d_input_b, "d_input_b");
    check_gradient(input_c, d_input_c, "d_input_c");

    rlt::free(device, model);
    rlt::free(device, model_pert);
    rlt::free(device, buffer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, input_c);
    rlt::free(device, d_output);
    rlt::free(device, d_input_a);
    rlt::free(device, d_input_b);
    rlt::free(device, d_input_c);
}

// ======================== 3-branch with HEAD: training ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_THREE_BRANCHES_HEAD_TRAINING){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;
    constexpr TI NUM_STEPS = 1000;

    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;
    using INPUT_SHAPE_C = rlt::tensor::Shape<TI, 1, BATCH_SIZE, 2>;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using HEAD_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_LAYER = rlt::nn::layers::dense::BindConfiguration<HEAD_LAYER_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<HEAD_LAYER>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, INPUT_SHAPE_C>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD_MODULE, BRANCH_A, BRANCH_B, BRANCH_C>;

    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 1);

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

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_A, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_B, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_C, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> target;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> d_output;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, target);
    rlt::malloc(device, d_output);

    T initial_loss = 0;
    T final_loss = 0;

    for(TI step = 0; step < NUM_STEPS; step++){
        rlt::randn(device, input_a, rng);
        rlt::randn(device, input_b, rng);
        rlt::randn(device, input_c, rng);

        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T sum_all = 0;
            for(TI j = 0; j < 2; j++){
                sum_all += rlt::get(device, input_a, 0, bi, j);
                sum_all += rlt::get(device, input_b, 0, bi, j);
                sum_all += rlt::get(device, input_c, 0, bi, j);
            }
            rlt::set(device, target, sum_all, 0, bi, 0);
        }

        auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
        rlt::forward(device, model, fwd_inputs, buffer, rng);
        auto model_output = rlt::output(device, model);

        T loss = 0;
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            T diff = rlt::get(device, model_output, 0, bi, 0) - rlt::get(device, target, 0, bi, 0);
            loss += diff * diff;
            rlt::set(device, d_output, 2 * diff / BATCH_SIZE, 0, bi, 0);
        }
        loss /= BATCH_SIZE;

        if(step == 0) initial_loss = loss;
        if(step == NUM_STEPS - 1) final_loss = loss;

        rlt::zero_gradient(device, model);
        auto bwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
        rlt::backward(device, model, bwd_inputs, d_output, buffer);
        rlt::step(device, optimizer, model);
    }

    std::cout << "3-branch HEAD Training loss: " << initial_loss << " -> " << final_loss << std::endl;
    ASSERT_LT(final_loss, initial_loss * 0.2) << "3-branch HEAD training should reduce loss significantly";

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, optimizer);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, input_c);
    rlt::free(device, target);
    rlt::free(device, d_output);
}

// ======================== CHANGE_BATCH_SIZE ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_CHANGE_BATCH_SIZE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 4;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>>;
    using PARALLEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, void, BRANCH_A, BRANCH_B, BRANCH_C>;

    static_assert(rlt::get<1>(typename PARALLEL::OUTPUT_SHAPE{}) == BATCH_SIZE);
    static_assert(rlt::get<2>(typename PARALLEL::OUTPUT_SHAPE{}) == 12);

    using PARALLEL_BS8 = typename PARALLEL::template CHANGE_BATCH_SIZE<TI, 8>;
    static_assert(rlt::get<1>(typename PARALLEL_BS8::OUTPUT_SHAPE{}) == 8);
    static_assert(rlt::get<2>(typename PARALLEL_BS8::OUTPUT_SHAPE{}) == 12);

    using PARALLEL_BS1 = typename PARALLEL::template CHANGE_BATCH_SIZE<TI, 1>;
    static_assert(rlt::get<1>(typename PARALLEL_BS1::OUTPUT_SHAPE{}) == 1);
    static_assert(rlt::get<2>(typename PARALLEL_BS1::OUTPUT_SHAPE{}) == 12);
}

// ======================== CHANGE_CAPABILITY 3-branch ========================

TEST(RL_TOOLS_NN_MODELS_PARALLEL, TEST_CHANGE_CAPABILITY_THREE_BRANCHES){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TI = typename DEVICE::index_t;

    constexpr TI BATCH_SIZE = 2;

    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 3, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;

    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;

    using LAYER_C_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_C = rlt::nn::layers::dense::BindConfiguration<LAYER_C_CONFIG>;
    using MODULE_C = rlt::nn_models::sequential::Module<LAYER_C>;

    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>>;
    using BRANCH_C = rlt::nn_models::parallel::Branch<MODULE_C, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>>;

    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using PARALLEL_GRAD = rlt::nn_models::parallel::Build<CAPABILITY, void, BRANCH_A, BRANCH_B, BRANCH_C>;
    using PARALLEL_FWD = typename PARALLEL_GRAD::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    PARALLEL_GRAD model_grad;
    PARALLEL_FWD model_fwd;
    typename PARALLEL_GRAD::Buffer<> buffer_grad;
    typename PARALLEL_FWD::Buffer<> buffer_fwd;

    rlt::malloc(device, model_grad);
    rlt::malloc(device, model_fwd);
    rlt::malloc(device, buffer_grad);
    rlt::malloc(device, buffer_fwd);
    rlt::init_weights(device, model_grad, rng);
    rlt::copy(device, device, model_grad, model_fwd);

    using OUTPUT_SHAPE = typename PARALLEL_GRAD::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 5>, true>> input_a;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 3>, true>> input_b;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, 7>, true>> input_c;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output_grad;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE, true>> output_fwd;

    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, input_c);
    rlt::malloc(device, output_grad);
    rlt::malloc(device, output_fwd);

    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);
    rlt::randn(device, input_c, rng);

    auto fwd_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::forward(device, model_grad, fwd_inputs, output_grad, buffer_grad, rng);

    auto eval_inputs = rlt::nn_models::parallel::pack_inputs(input_a, input_b, input_c);
    rlt::evaluate(device, model_fwd, eval_inputs, output_fwd, buffer_fwd, rng);

    T diff = rlt::abs_diff(device, output_grad, output_fwd);
    ASSERT_LT(diff, 1e-10) << "Forward (grad) vs evaluate (fwd) should match for 3 branches";

    rlt::free(device, model_grad);
    rlt::free(device, model_fwd);
    rlt::free(device, buffer_grad);
    rlt::free(device, buffer_fwd);
    rlt::free(device, input_a);
    rlt::free(device, input_b);
    rlt::free(device, input_c);
    rlt::free(device, output_grad);
    rlt::free(device, output_fwd);
}
