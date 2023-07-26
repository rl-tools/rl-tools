#define BACKPROP_TOOLS_DISABLE_TENSORBOARD
#define BACKPROP_TOOLS_BACKEND_ENABLE_BLAS
//#define BACKPROP_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/operations_cpu.h>
//#define BACKPROP_TOOLS_BACKEND_DISABLE_BLAS
#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn/layers/concat_constant/operations_generic.h>

#include <backprop_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <gtest/gtest.h>
#include <thread>

namespace bpt = backprop_tools;

namespace config{
    using namespace bpt::nn_models::sequential::interface;
    template <typename T_T, typename T_TI>
    struct CONFIG{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI INPUT_DIM = 4;
        static constexpr TI HIDDEN_DIM = 64;
        static constexpr TI OUTPUT_DIM = 1;
        static constexpr TI BATCH_SIZE = 64;
        static constexpr T THRESHOLD = 1e-5;

        using STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, BATCH_SIZE>;
        using SPEC = bpt::nn_models::mlp::AdamSpecification<STRUCTURE_SPEC>;
        using MODEL = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>;

        using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, INPUT_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, BATCH_SIZE>;
        using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, BATCH_SIZE>;
        using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam, BATCH_SIZE>;
        using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

        using OPTIMIZER = bpt::nn::optimizers::Adam<bpt::nn::optimizers::adam::DefaultParametersTF<T, TI>>;
        using SEQUENTIAL_OPTIMIZER = bpt::nn::optimizers::Adam<bpt::nn::optimizers::adam::DefaultParametersTF<T, TI>>;

        using SEQUENTIAL_MODEL = Module<CONFIG::LAYER_1, Module<CONFIG::LAYER_2, Module<CONFIG::LAYER_3>>>;
    };
}




template <typename DEVICE, typename SEQUENTIAL_DEVICE, typename CONFIG>
void test_correctness(){
    typename CONFIG::MODEL model, model_temp;
    typename CONFIG::MODEL::template Buffers<CONFIG::BATCH_SIZE> buffer;
    DEVICE device;
    SEQUENTIAL_DEVICE sdevice;
    typename CONFIG::SEQUENTIAL_MODEL sequential_model, sequential_model_temp;
    typename CONFIG::SEQUENTIAL_MODEL::template DoubleBuffer<CONFIG::BATCH_SIZE> sequential_buffer;
    typename CONFIG::OPTIMIZER optimizer;
    typename CONFIG::SEQUENTIAL_OPTIMIZER sequential_optimizer;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;

    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 0);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::MODEL::INPUT_DIM>> input, d_input, d_input_sequential, d_input_only, d_input_sequential_only;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::MODEL::OUTPUT_DIM>> output, output_eval, d_output, output_sequential, output_sequential_eval;

    bpt::malloc(device, input);
    bpt::malloc(device, d_input);
    bpt::malloc(device, d_input_only);
    bpt::malloc(sdevice, d_input_sequential);
    bpt::malloc(sdevice, d_input_sequential_only);
    bpt::malloc(device, output);
    bpt::malloc(device, output_eval);
    bpt::malloc(sdevice, output_sequential);
    bpt::malloc(sdevice, output_sequential_eval);
    bpt::malloc(device, d_output);
    bpt::malloc(device, model);
    bpt::malloc(device, model_temp);
    bpt::malloc(device, buffer);
    bpt::malloc(sdevice, sequential_model);
    bpt::malloc(sdevice, sequential_model_temp);
    bpt::malloc(sdevice, sequential_buffer);

    bpt::init_weights(device, model, rng);
    bpt::randn(device, input, rng);
    bpt::randn(device, d_output, rng);

    bpt::copy(sdevice, device, sequential_model.content, model.input_layer);
    bpt::copy(sdevice, device, sequential_model.next_module.content, model.hidden_layers[0]);
    bpt::copy(sdevice, device, sequential_model.next_module.next_module.content, model.output_layer);

    bpt::evaluate(device, model, input, output_eval, buffer);
    bpt::evaluate(sdevice, sequential_model, input, output_sequential_eval, sequential_buffer);

    {
        auto abs_diff = bpt::abs_diff(device, output_eval, output_sequential_eval);
        std::cout << "abs_diff evaluate: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    bpt::forward(device, model, input, output);
    bpt::forward(sdevice, sequential_model, input, output_sequential);

    {
        auto abs_diff = bpt::abs_diff(device, output, output_sequential);
        std::cout << "abs_diff forward: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    bpt::reset_optimizer_state(device, optimizer, model);
    bpt::reset_optimizer_state(sdevice, sequential_optimizer, sequential_model);
    bpt::zero_gradient(device, model);
    bpt::zero_gradient(sdevice, sequential_model);
    bpt::backward(device, model, input, d_output, buffer);
    bpt::backward(sdevice, sequential_model, input, d_output, sequential_buffer);

    {
        auto abs_diff = bpt::abs_diff(device, model.input_layer, sequential_model.content);
        abs_diff += bpt::abs_diff(device, model.hidden_layers[0], sequential_model.next_module.content);
        abs_diff += bpt::abs_diff(device, model.output_layer, sequential_model.next_module.next_module.content);
        std::cout << "abs_diff gradient: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    bpt::copy(device, device, model_temp, model);
    bpt::copy(sdevice, sdevice, sequential_model_temp, sequential_model);

    bpt::step(device, optimizer, model);
    bpt::step(sdevice, sequential_optimizer, sequential_model);

    {
        auto abs_diff = bpt::abs_diff(device, model.input_layer, sequential_model.content);
        abs_diff += bpt::abs_diff(device, model.hidden_layers[0], sequential_model.next_module.content);
        abs_diff += bpt::abs_diff(device, model.output_layer, sequential_model.next_module.next_module.content);
        std::cout << "abs_diff adam step: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    {
        auto abs_diff = bpt::abs_diff(device, model, model_temp);
        abs_diff += bpt::abs_diff(device, sequential_model, sequential_model_temp);
        std::cout << "abs_diff pre-post: " << abs_diff << std::endl;
        ASSERT_GT(abs_diff, 0);
    }

    bpt::reset_forward_state(device, model);
    bpt::reset_forward_state(sdevice, sequential_model);

    {
        auto abs_diff = bpt::abs_diff(device, model.input_layer, sequential_model.content);
        abs_diff += bpt::abs_diff(device, model.hidden_layers[0], sequential_model.next_module.content);
        abs_diff += bpt::abs_diff(device, model.output_layer, sequential_model.next_module.next_module.content);
        std::cout << "abs diff reset forward state: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    bpt::reset_optimizer_state(device, optimizer, model);
    bpt::reset_optimizer_state(sdevice, sequential_optimizer, sequential_model);
    bpt::zero_gradient(device, model);
    bpt::zero_gradient(sdevice, sequential_model);
    bpt::forward(device, model, input, output);
    bpt::forward(sdevice, sequential_model, input, output_sequential);
    bpt::backward_full(device, model, input, d_output, d_input, buffer);
    bpt::backward_full(sdevice, sequential_model, input, d_output, d_input_sequential, sequential_buffer);

    {
        auto abs_diff = bpt::abs_diff(device, model.input_layer, sequential_model.content);
        abs_diff += bpt::abs_diff(device, model.hidden_layers[0], sequential_model.next_module.content);
        abs_diff += bpt::abs_diff(device, model.output_layer, sequential_model.next_module.next_module.content);
        std::cout << "abs_diff gradient full: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }
    {
        auto abs_diff = bpt::abs_diff(device, d_input, d_input_sequential);
        std::cout << "abs_diff d_input: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    bpt::copy(device, device, model_temp, model);
    bpt::copy(sdevice, sdevice, sequential_model_temp, sequential_model);

    bpt::backward_input(device, model, d_output, d_input_only, buffer);
    bpt::backward_input(sdevice, sequential_model, d_output, d_input_sequential_only, sequential_buffer);

    {
        auto abs_diff = bpt::abs_diff(device, d_input_only, d_input_sequential_only);
        std::cout << "abs_diff d_input only: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }
    {
        auto abs_diff = bpt::abs_diff(device, d_input_only, d_input);
        std::cout << "abs_diff d_input pre-post: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, CONFIG::THRESHOLD);
    }

    {
        auto abs_diff = bpt::abs_diff(device, model, model_temp);
        abs_diff += bpt::abs_diff(device, sequential_model, sequential_model_temp);
        std::cout << "abs_diff pre-post: " << abs_diff << std::endl;
        ASSERT_EQ(abs_diff, 0);
    }

}
TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, CORRECTNESS_BACKWARD_PARAMS_BLAS){
    using T = float;
//using DEVICE = bpt::devices::DefaultCPU;
    using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
    using TI = typename DEVICE::index_t;

    test_correctness<DEVICE, DEVICE, config::CONFIG<T, TI>>();
}

TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, CORRECTNESS_BACKWARD_PARAMS_BLAS_CPU){
    using T = double;
//using DEVICE = bpt::devices::DefaultCPU;
    using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
    using SEQUENTIAL_DEVICE = bpt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;

    test_correctness<DEVICE, SEQUENTIAL_DEVICE, config::CONFIG<T, TI>>();
}

TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, CORRECTNESS_BACKWARD_PARAMS_CPU_BLAS){
    using T = double;
//using DEVICE = bpt::devices::DefaultCPU;
    using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
    using SEQUENTIAL_DEVICE = bpt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;

    test_correctness<SEQUENTIAL_DEVICE, DEVICE, config::CONFIG<T, TI>>();
}

//TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, BENCHMARK){
template <typename DEVICE, typename CONFIG>
void test_benchmark(){
    typename CONFIG::MODEL model;
    DEVICE device;
    typename CONFIG::SEQUENTIAL_MODEL sequential_model;
    typename CONFIG::SEQUENTIAL_MODEL::template DoubleBuffer<CONFIG::BATCH_SIZE> sequential_buffer;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;

    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 0);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::MODEL::INPUT_DIM>> input, d_input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::MODEL::OUTPUT_DIM>> output, d_output, output_sequential;

    bpt::malloc(device, input);
    bpt::malloc(device, d_input);
    bpt::malloc(device, output);
    bpt::malloc(device, output_sequential);
    bpt::malloc(device, d_output);
    bpt::malloc(device, model);
    bpt::malloc(device, sequential_model);
    bpt::malloc(device, sequential_buffer);

    bpt::init_weights(device, model, rng);
    bpt::randn(device, input, rng);
    bpt::randn(device, d_output, rng);

    bpt::copy(device, device, sequential_model.content, model.input_layer);
    bpt::copy(device, device, sequential_model.next_module.content, model.hidden_layers[0]);
    bpt::copy(device, device, sequential_model.next_module.next_module.content, model.output_layer);

    bpt::forward(device, model, input, output);
    bpt::evaluate(device, sequential_model, input, output_sequential, sequential_buffer);

    bpt::print(device, output);
    bpt::print(device, output_sequential);
    auto abs_diff = bpt::abs_diff(device, output, output_sequential);
    std::cout << "abs_diff: " << abs_diff << std::endl;

    T mean_factor = 0;
    T std_factor = 0;

    for(TI it=0; it < 100; it++){
        T time_d_input, time;
        std::this_thread::sleep_for(std::chrono::milliseconds (100));
        {
            constexpr TI NUM_ITERATIONS = 1000;
            auto start = std::chrono::high_resolution_clock::now();
            for(TI i = 0; i < NUM_ITERATIONS; i++){
                bpt::backward(device, sequential_model, input, d_output, sequential_buffer);
            }
            auto end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
//            std::cout << "No d_input: Iterations per second: " << NUM_ITERATIONS / std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds (100));
        {
            constexpr TI NUM_ITERATIONS = 1000;
            auto start = std::chrono::high_resolution_clock::now();
            for(TI i = 0; i < NUM_ITERATIONS; i++){
                bpt::backward_full(device, sequential_model, input, d_output, d_input, sequential_buffer);
            }
            auto end = std::chrono::high_resolution_clock::now();
            time_d_input = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
//            std::cout << "d_input: Iterations per second: " << NUM_ITERATIONS / std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
        }
        std::cout << "w/o d_input " << time_d_input/time << "x faster" << std::endl;
        mean_factor += time_d_input/time;
        std_factor += (time_d_input/time)*(time_d_input/time);
    }
    mean_factor /= 100;
    std_factor /= 100;
    std_factor = std::sqrt(std_factor - mean_factor*mean_factor);

    std::cout << "mean_factor: " << mean_factor << std::endl;
    std::cout << "std_factor: " << std_factor << std::endl;

    ASSERT_GT(mean_factor, 1.0);

}

TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, BENCHMARK){
    using T = double;
    using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
    using TI = typename DEVICE::index_t;

    test_benchmark<DEVICE, config::CONFIG<T, TI>>();
}
