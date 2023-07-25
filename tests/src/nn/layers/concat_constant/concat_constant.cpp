#define BACKPROP_TOOLS_DISABLE_TENSORBOARD
#define BACKPROP_TOOLS_BACKEND_ENABLE_BLAS
#define BACKPROP_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
//#define BACKPROP_TOOLS_BACKEND_DISABLE_BLAS
#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn/layers/concat_constant/operations_generic.h>

#include <backprop_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <gtest/gtest.h>
#include <thread>

namespace bpt = backprop_tools;

using T = float;
//using DEVICE = bpt::devices::DefaultCPU;
using DEVICE = bpt::DEVICE_FACTORY<bpt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

constexpr TI INPUT_DIM = 512;
constexpr TI HIDDEN_DIM = 64;
constexpr TI OUTPUT_DIM = 4;
constexpr TI BATCH_SIZE = 64;

using STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::ActivationFunction::IDENTITY, BATCH_SIZE>;
using SPEC = bpt::nn_models::mlp::BackwardGradientSpecification<STRUCTURE_SPEC>;
using MODEL = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>;

using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, INPUT_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, bpt::nn::activation_functions::ActivationFunction::IDENTITY, bpt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

namespace sequential_model_factory{
    using namespace bpt::nn_models::sequential::interface;
    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}
using SEQUENTIAL_MODEL = sequential_model_factory::MODEL;

TEST(BACKPROP_TOOLS_NN_LAYERS_CONCAT_CONSTANT, TEST){
    MODEL model;
    DEVICE device;
    SEQUENTIAL_MODEL sequential_model;
    SEQUENTIAL_MODEL::DoubleBuffer<BATCH_SIZE> sequential_buffer;

    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM{}, 0);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, BATCH_SIZE, MODEL::INPUT_DIM>> input, d_input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, BATCH_SIZE, MODEL::OUTPUT_DIM>> output, d_output, output_sequential;

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



}