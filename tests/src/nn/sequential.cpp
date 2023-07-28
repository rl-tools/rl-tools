#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/optimizers/adam/operations_generic.h>
#include <backprop_tools/nn/operations_cpu.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>

namespace bpt = backprop_tools;

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



TEST(BACKPROP_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_SEQUENTIAL_STATIC){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 0);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 0);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 10);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 1, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 1, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 1);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 1);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 100, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 100, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 10);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 10);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 100);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 11, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 11, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 20, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_4 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4>>>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 11);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 11);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }
    {
        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 20, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 11, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 11, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
        using LAYER_4_SPEC = bpt::nn::layers::dense::Specification<T, TI, 11, 100, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_4 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_4_SPEC>;
        using LAYER_5_SPEC = bpt::nn::layers::dense::Specification<T, TI, 100, 20, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
        using LAYER_5 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_5_SPEC>;

        using namespace bpt::nn_models::sequential::interface;
        using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3, Module<LAYER_4, Module<LAYER_5>>>>>;

        static_assert(bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() == 100);
        static_assert(SEQUENTIAL::SPEC::MAX_HIDDEN_DIM == 100);
        static_assert(bpt::nn_models::sequential::find_output_dim<typename SEQUENTIAL::SPEC>() == 20);
        static_assert(SEQUENTIAL::SPEC::OUTPUT_DIM == 20);
    }

}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_FORWARD){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = bpt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = bpt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;

    bpt::malloc(device, mlp);
    bpt::malloc(device, layer_1);
    bpt::malloc(device, layer_2);
    bpt::malloc(device, layer_3);

    bpt::malloc(device, sequential);

    bpt::init_weights(device, mlp, rng);
    bpt::copy(device, device, layer_1, mlp.input_layer);
    bpt::copy(device, device, layer_2, mlp.hidden_layers[0]);
    bpt::copy(device, device, layer_3, mlp.output_layer);

    bpt::copy(device, device, sequential.content, mlp.input_layer);
    bpt::copy(device, device, sequential.next_module.content, mlp.hidden_layers[0]);
    bpt::copy(device, device, sequential.next_module.next_module.content, mlp.output_layer);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_chain;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::malloc(device, input);
    bpt::malloc(device, hidden_tick);
    bpt::malloc(device, hidden_tock);
    bpt::malloc(device, output_mlp);
    bpt::malloc(device, output_chain);
    bpt::malloc(device, output_sequential);

    bpt::randn(device, input, rng);
    bpt::print(device, input);

    for(TI i = 0; i < 2; i++){
        bpt::forward(device, mlp, input, output_mlp);
        bpt::print(device, output_mlp);

        bpt::forward(device, layer_1, input, hidden_tick);
        bpt::forward(device, layer_2, hidden_tick, hidden_tock);
        bpt::forward(device, layer_3, hidden_tock, output_chain);
        bpt::print(device, output_chain);

        bpt::forward(device, sequential.content                        , input, hidden_tick);
        bpt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
        bpt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);
        bpt::print(device, output_sequential);

        bpt::forward(device, sequential, input, output_sequential);
        bpt::print(device, output_sequential);

        auto abs_diff_sequential = bpt::abs_diff(device, output_mlp, output_sequential);
        auto abs_diff_chain = bpt::abs_diff(device, output_mlp, output_sequential);

        std::cout << "Abs diff sequential: " << abs_diff_sequential << std::endl;
        std::cout << "Abs diff chain: " << abs_diff_chain << std::endl;

        ASSERT_LT(abs_diff_sequential, 1e-8);
        ASSERT_LT(abs_diff_chain, 1e-8);

        bpt::init_weights(device, sequential, rng);
        bpt::copy(device, device, mlp.input_layer, sequential.content);
        bpt::copy(device, device, mlp.hidden_layers[0], sequential.next_module.content);
        bpt::copy(device, device, mlp.output_layer, sequential.next_module.next_module.content);
    }
}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_INCOMPATIBLE_DEFINITION){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 1, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, 10>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 1, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, 10>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2>>;

    DEVICE device;
    SEQUENTIAL model;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    bpt::malloc(device, model);
    bpt::init_weights(device, model, rng);

    static_assert(bpt::nn_models::sequential::check_batch_size_consistency<SEQUENTIAL>);
}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_EVALUATE){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    typename SEQUENTIAL::DoubleBuffer<1> buffer;

    bpt::malloc(device, sequential);
    bpt::malloc(device, buffer);
    bpt::init_weights(device, sequential, rng);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential_evaluate;
    bpt::malloc(device, input);
    bpt::malloc(device, output_sequential);
    bpt::malloc(device, output_sequential_evaluate);

    bpt::randn(device, input, rng);
    bpt::print(device, input);

    bpt::forward(device, sequential, input, output_sequential);
    bpt::print(device, output_sequential);
    bpt::evaluate(device, sequential, input, output_sequential_evaluate, buffer);
    bpt::print(device, output_sequential_evaluate);

    auto abs_diff = bpt::abs_diff(device, output_sequential_evaluate, output_sequential);

    std::cout << "Abs diff evaluate: " << abs_diff << std::endl;

    ASSERT_LT(abs_diff, 1e-8);

}

TEST(BACKPROP_TOOLS_NN_MODELS_MLP_SEQUENTIAL, TEST_BACKWARD){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    constexpr T THRESHOLD = 1e-8;

    using MLP_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, 5, 2, 3, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_SPEC = bpt::nn_models::mlp::AdamSpecification<MLP_STRUCTURE_SPEC>;
    using MLP = bpt::nn_models::mlp::NeuralNetworkAdam<MLP_SPEC>;

    using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, 5, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 10, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam>;
    using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
    using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, 10, 2, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam>;
    using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

    using namespace bpt::nn_models::sequential::interface;
    using SEQUENTIAL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;

    std::cout << "Max hidden dim: " << bpt::nn_models::sequential::find_max_hiddend_dim<TI, typename SEQUENTIAL::SPEC>() << std::endl;

    DEVICE device;
    MLP mlp;
    typename MLP::Buffers<1> mlp_buffers;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 1);

    LAYER_1 layer_1;
    LAYER_2 layer_2;
    LAYER_3 layer_3;

    SEQUENTIAL sequential;
    SEQUENTIAL::DoubleBuffer<1> buffer_sequential;

    bpt::malloc(device, mlp);
    bpt::malloc(device, layer_1);
    bpt::malloc(device, layer_2);
    bpt::malloc(device, layer_3);
    bpt::malloc(device, mlp_buffers);

    bpt::malloc(device, sequential);
    bpt::malloc(device, buffer_sequential);

    bpt::init_weights(device, mlp, rng);
    bpt::copy(device, device, layer_1, mlp.input_layer);
    bpt::copy(device, device, layer_2, mlp.hidden_layers[0]);
    bpt::copy(device, device, layer_3, mlp.output_layer);

    bpt::copy(device, device, sequential.content, mlp.input_layer);
    bpt::copy(device, device, sequential.next_module.content, mlp.hidden_layers[0]);
    bpt::copy(device, device, sequential.next_module.next_module.content, mlp.output_layer);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 5>> d_input_mlp, d_input_chain, d_input_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> d_hidden_tick;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 10>> d_hidden_tock;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_mlp;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_chain;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> output_sequential;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 2>> d_output;
    bpt::malloc(device, input);
    bpt::malloc(device, d_input_mlp);
    bpt::malloc(device, d_input_chain);
    bpt::malloc(device, d_input_sequential);
    bpt::malloc(device, hidden_tick);
    bpt::malloc(device, hidden_tock);
    bpt::malloc(device, d_hidden_tick);
    bpt::malloc(device, d_hidden_tock);
    bpt::malloc(device, output_mlp);
    bpt::malloc(device, output_chain);
    bpt::malloc(device, output_sequential);
    bpt::malloc(device, d_output);

    bpt::randn(device, input, rng);
    bpt::randn(device, d_output, rng);
    bpt::print(device, input);

    bpt::forward(device, mlp, input, output_mlp);
    bpt::zero_gradient(device, mlp);
    bpt::backward_full(device, mlp, input, d_output, d_input_mlp, mlp_buffers);

    bpt::print(device, d_input_mlp);

    bpt::zero_gradient(device, layer_1);
    bpt::zero_gradient(device, layer_2);
    bpt::zero_gradient(device, layer_3);
    bpt::forward(device, layer_1, input, hidden_tick);
    bpt::forward(device, layer_2, hidden_tick, hidden_tock);
    bpt::forward(device, layer_3, hidden_tock, output_chain);
    bpt::backward(device, layer_3, hidden_tock, d_output, d_hidden_tick);
    bpt::backward(device, layer_2, hidden_tick, d_hidden_tick, d_hidden_tock);
    bpt::backward(device, layer_1, input, d_hidden_tock, d_input_chain);

    bpt::print(device, d_input_chain);

    {
        auto abs_diff_d_input = bpt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = bpt::abs_diff(device, mlp.input_layer.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = bpt::abs_diff(device, mlp.input_layer.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = bpt::abs_diff(device, mlp.hidden_layers[0].weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = bpt::abs_diff(device, mlp.hidden_layers[0].biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = bpt::abs_diff(device, mlp.output_layer.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = bpt::abs_diff(device, mlp.output_layer.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }

    bpt::forward(device, sequential.content                        , input, hidden_tick);
    bpt::forward(device, sequential.next_module.content            , hidden_tick, hidden_tock);
    bpt::forward(device, sequential.next_module.next_module.content, hidden_tock, output_sequential);

    bpt::set(sequential.content.weights.gradient, 0, 0, 10);
    bpt::set(sequential.next_module.content.weights.gradient, 0, 0, 10);
    bpt::set(sequential.next_module.next_module.content.weights.gradient, 0, 0, 10);
    bpt::forward(device, sequential, input, output_sequential);
    bpt::zero_gradient(device, sequential);
    bpt::backward_full(device, sequential, input, d_output, d_input_sequential, buffer_sequential);

    bpt::print(device, d_input_sequential);

    {
        auto abs_diff_d_input = bpt::abs_diff(device, d_input_mlp, d_input_chain);
        auto abs_diff_grad_W_1 = bpt::abs_diff(device, sequential.content.weights.gradient, layer_1.weights.gradient);
        auto abs_diff_grad_b_1 = bpt::abs_diff(device, sequential.content.biases.gradient, layer_1.biases.gradient);
        auto abs_diff_grad_W_2 = bpt::abs_diff(device, sequential.next_module.content.weights.gradient, layer_2.weights.gradient);
        auto abs_diff_grad_b_2 = bpt::abs_diff(device, sequential.next_module.content.biases.gradient, layer_2.biases.gradient);
        auto abs_diff_grad_W_3 = bpt::abs_diff(device, sequential.next_module.next_module.content.weights.gradient, layer_3.weights.gradient);
        auto abs_diff_grad_b_3 = bpt::abs_diff(device, sequential.next_module.next_module.content.biases.gradient, layer_3.biases.gradient);

        ASSERT_LT(abs_diff_d_input, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_1, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_2, THRESHOLD);
        ASSERT_LT(abs_diff_grad_W_3, THRESHOLD);
        ASSERT_LT(abs_diff_grad_b_3, THRESHOLD);
    }
}
