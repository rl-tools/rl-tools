#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>


//template <typename T_CONTENT>
//struct OutputModule{
//    using CONTENT = T_CONTENT;
//    static constexpr auto MAX_HIDDEN_DIM = CONTENT::INPUT_DIM;
//    CONTENT content;
//};
//
//template <typename T_CONTENT, typename T_NEXT_MODULE>
//struct Specification{
//    using CONTENT = T_CONTENT;
//    using NEXT_MODULE = T_NEXT_MODULE;
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//    static_assert(NEXT_MODULE_INPUT_DIM == CONTENT::OUTPUT_DIM);
//    static constexpr auto NEXT_MODULE_INPUT_DIM = NEXT_MODULE::CONTENT::INPUT_DIM;
//};
//



TEST(RL_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_SEQUENTIAL_STATIC){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    {
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 20, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;

        using namespace rlt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1>;

        static_assert(rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 0);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 0);
        static_assert(rlt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 10);
    }
    {
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 1, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 1, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace rlt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(rlt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 1);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 1);
    }
    {
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 100, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 100, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace rlt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(rlt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 100);
    }
    {
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 20, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 11, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 11, 11, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
        using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = rlt::nn::layers::dense::Specification<T, TI, 11, 20, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
        using LAYER_4 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;

        using namespace rlt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4>>>>;

        static_assert(rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 11);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 11);
        static_assert(rlt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }
    {
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 20, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 11, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 11, 11, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
        using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = rlt::nn::layers::dense::Specification<T, TI, 11, 100, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
        using LAYER_4 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;
        using LAYER_5_SPEC = rlt::nn::layers::dense::Specification<T, TI, 100, 20, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
        using LAYER_5 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_5_SPEC>;

        using namespace rlt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4, Module<LAYER_5>>>>>;

        static_assert(rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 100);
        static_assert(rlt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }

}

TEST(RL_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_FORWARD){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using MLP_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = rlt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = rlt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 5, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
    using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace rlt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;

    rlt::malloc(device, mlp);
    rlt::malloc(device, layer_1);
    rlt::malloc(device, layer_2);
    rlt::malloc(device, layer_3);

    rlt::malloc(device, sequential);

    rlt::init_weights(device, mlp, rng);
    rlt::copy(device, device, mlp.input_layer, layer_1);
    rlt::copy(device, device, mlp.hidden_layers[0], layer_2);
    rlt::copy(device, device, mlp.output_layer, layer_3);

    rlt::copy(device, device, mlp.input_layer, sequential.content);
    rlt::copy(device, device, mlp.hidden_layers[0], sequential.next_module.content);
    rlt::copy(device, device, mlp.output_layer, sequential.next_module.next_module.content);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 5>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_chain;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    rlt::malloc(device, input);
    rlt::malloc(device, hidden_tick);
    rlt::malloc(device, hidden_tock);
    rlt::malloc(device, output_mlp);
    rlt::malloc(device, output_chain);
    rlt::malloc(device, output_sequential);

    rlt::randn(device, input, rng);
    rlt::print(device, input);

    for(TI i = 0; i < 2; i++){
        rlt::forward(device, mlp, input, output_mlp);
        rlt::print(device, output_mlp);

        rlt::forward(device, layer_1, input, hidden_tick);
        rlt::forward(device, layer_2, hidden_tick, hidden_tock);
        rlt::forward(device, layer_3, hidden_tock, output_chain);
        rlt::print(device, output_chain);

        rlt::forward(device, sequential.content                        , input, hidden_tick);
        rlt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
        rlt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);
        rlt::print(device, output_sequential);

        rlt::forward(device, sequential, input, output_sequential);
        rlt::print(device, output_sequential);

        auto abs_diff_sequential = rlt::abs_diff(device, output_mlp, output_sequential);
        auto abs_diff_chain = rlt::abs_diff(device, output_mlp, output_sequential);

        std::cout << "Abs diff sequential: " << abs_diff_sequential << std::endl;
        std::cout << "Abs diff chain: " << abs_diff_chain << std::endl;

        ASSERT_LT(abs_diff_sequential, 1e-8);
        ASSERT_LT(abs_diff_chain, 1e-8);

        rlt::init_weights(device, sequential, rng);
        rlt::copy(device, device, sequential.content, mlp.input_layer);
        rlt::copy(device, device, sequential.next_module.content, mlp.hidden_layers[0]);
        rlt::copy(device, device, sequential.next_module.next_module.content, mlp.output_layer);
    }
}

TEST(RL_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_INCOMPATIBLE_DEFINITION){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 1, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam, 10>;
    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 1, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam, 10>;
    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

    using namespace rlt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

    DEVICE device;
    SEQUENTIAL model;
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    rlt::malloc(device, model);
    rlt::init_weights(device, model, rng);

    static_assert(rlt::nn_models::sequential::check_batch_size_consistency<SEQUENTIAL>);
}

TEST(RL_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_EVALUATE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;

    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 5, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
    using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace rlt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    typename SEQUENTIAL::Buffer<1> buffer;

    rlt::malloc(device, sequential);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, sequential, rng);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 5>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_sequential_evaluate;
    rlt::malloc(device, input);
    rlt::malloc(device, output_sequential);
    rlt::malloc(device, output_sequential_evaluate);

    rlt::randn(device, input, rng);
    rlt::print(device, input);

    rlt::forward(device, sequential, input, output_sequential);
    rlt::print(device, output_sequential);
    rlt::evaluate(device, sequential, input, output_sequential_evaluate, buffer);
    rlt::print(device, output_sequential_evaluate);

    auto abs_diff = rlt::abs_diff(device, output_sequential_evaluate, output_sequential);

    std::cout << "Abs diff evaluate: " << abs_diff << std::endl;

    ASSERT_LT(abs_diff, 1e-8);

}

TEST(RL_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_BACKWARD){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    constexpr T THRESHOLD = 1e-8;

    using MLP_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = rlt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = rlt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 5, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 10, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam>;
    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 10, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam>;
    using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace rlt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << rlt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    typename MLP::Buffer<1> mlp_buffers;
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    SEQUENTIAL::Buffer<1> buffer_sequential;

    rlt::malloc(device, mlp);
    rlt::malloc(device, layer_1);
    rlt::malloc(device, layer_2);
    rlt::malloc(device, layer_3);
    rlt::malloc(device, mlp_buffers);

    rlt::malloc(device, sequential);
    rlt::malloc(device, buffer_sequential);

    rlt::init_weights(device, mlp, rng);
    rlt::copy(device, device, mlp.input_layer, layer_1);
    rlt::copy(device, device, mlp.hidden_layers[0], layer_2);
    rlt::copy(device, device, mlp.output_layer, layer_3);

    rlt::copy(device, device, mlp.input_layer, sequential.content);
    rlt::copy(device, device, mlp.hidden_layers[0], sequential.next_module.content);
    rlt::copy(device, device, mlp.output_layer, sequential.next_module.next_module.content);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 5>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 5>> d_input_mlp, d_input_chain, d_input_sequential;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> d_hidden_tick;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 10>> d_hidden_tock;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_chain;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 2>> d_output;
    rlt::malloc(device, input);
    rlt::malloc(device, d_input_mlp);
    rlt::malloc(device, d_input_chain);
    rlt::malloc(device, d_input_sequential);
    rlt::malloc(device, hidden_tick);
    rlt::malloc(device, hidden_tock);
    rlt::malloc(device, d_hidden_tick);
    rlt::malloc(device, d_hidden_tock);
    rlt::malloc(device, output_mlp);
    rlt::malloc(device, output_chain);
    rlt::malloc(device, output_sequential);
    rlt::malloc(device, d_output);

    rlt::randn(device, input, rng);
    rlt::randn(device, d_output, rng);
    rlt::print(device, input);

    rlt::forward(device, mlp, input, output_mlp);
    rlt::zero_gradient(device, mlp);
    rlt::backward_full(device, mlp, input, d_output, d_input_mlp, mlp_buffers);

    rlt::print(device, d_input_mlp);

    rlt::zero_gradient(device, layer_1);
    rlt::zero_gradient(device, layer_2);
    rlt::zero_gradient(device, layer_3);
    rlt::forward(device, layer_1, input, hidden_tick);
    rlt::forward(device, layer_2, hidden_tick, hidden_tock);
    rlt::forward(device, layer_3, hidden_tock, output_chain);
    rlt::backward(device, layer_3, hidden_tock, d_output, d_hidden_tick);
    rlt::backward(device, layer_2, hidden_tick, d_hidden_tick, d_hidden_tock);
    rlt::backward(device, layer_1, input, d_hidden_tock, d_input_chain);

    rlt::print(device, d_input_chain);

    {
        auto abs_diff_d_input = rlt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = rlt::abs_diff(device, mlp.input_layer.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = rlt::abs_diff(device, mlp.input_layer.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = rlt::abs_diff(device, mlp.hidden_layers[0].weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = rlt::abs_diff(device, mlp.hidden_layers[0].biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = rlt::abs_diff(device, mlp.output_layer.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = rlt::abs_diff(device, mlp.output_layer.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }

    rlt::forward(device, sequential.content                        , input, hidden_tick);
    rlt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
    rlt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);

    rlt::set(sequential.content.weights.gradient, 0, 0, 10);
    rlt::set(sequential.next_module.content.weights.gradient, 0, 0, 10);
    rlt::set(sequential.next_module.next_module.content.weights.gradient, 0, 0, 10);
    rlt::forward(device, sequential, input, output_sequential);
    rlt::zero_gradient(device, sequential);
    rlt::backward_full(device, sequential, input, d_output, d_input_sequential, buffer_sequential);

    rlt::print(device, d_input_sequential);

    {
        auto abs_diff_d_input = rlt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = rlt::abs_diff(device, sequential.content.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = rlt::abs_diff(device, sequential.content.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = rlt::abs_diff(device, sequential.next_module.content.weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = rlt::abs_diff(device, sequential.next_module.content.biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = rlt::abs_diff(device, sequential.next_module.next_module.content.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = rlt::abs_diff(device, sequential.next_module.next_module.content.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }
}
