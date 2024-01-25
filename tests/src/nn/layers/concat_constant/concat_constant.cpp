#define RL_TOOLS_DISABLE_TENSORBOARD
#define RL_TOOLS_BACKEND_ENABLE_BLAS
#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
//#define RL_TOOLS_BACKEND_DISABLE_BLAS
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/concat_constant/operations_generic.h>

#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <gtest/gtest.h>
#include <thread>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using T = float;
//using DEVICE = rlt::devices::DefaultCPU;
using DEVICE = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

constexpr TI INPUT_DIM = 4;
constexpr TI HIDDEN_DIM = 64;
constexpr TI OUTPUT_DIM = 1;
constexpr TI BATCH_SIZE = 64;

using STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY, BATCH_SIZE>;
using SPEC = rlt::nn_models::mlp::BackwardGradientSpecification<STRUCTURE_SPEC>;
using MODEL = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>;

using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, INPUT_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::parameters::Adam, BATCH_SIZE>;
using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

namespace sequential_model_factory{
    using namespace rlt::nn_models::sequential::interface;
    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
}
using SEQUENTIAL_MODEL = sequential_model_factory::MODEL;

TEST(RL_TOOLS_NN_LAYERS_CONCAT_CONSTANT, TEST){
    MODEL model;
    DEVICE device;
    SEQUENTIAL_MODEL sequential_model;
    SEQUENTIAL_MODEL::Buffer<BATCH_SIZE> sequential_buffer;

    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{}, 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, MODEL::INPUT_DIM>> input, d_input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, MODEL::OUTPUT_DIM>> output, d_output, output_sequential;

    rlt::malloc(device, input);
    rlt::malloc(device, d_input);
    rlt::malloc(device, output);
    rlt::malloc(device, output_sequential);
    rlt::malloc(device, d_output);
    rlt::malloc(device, model);
    rlt::malloc(device, sequential_model);
    rlt::malloc(device, sequential_buffer);

    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::randn(device, d_output, rng);

    rlt::copy(device, device, model.input_layer, sequential_model.content);
    rlt::copy(device, device, model.hidden_layers[0], sequential_model.next_module.content);
    rlt::copy(device, device, model.output_layer, sequential_model.next_module.next_module.content);

    rlt::forward(device, model, input, output);
    rlt::evaluate(device, sequential_model, input, output_sequential, sequential_buffer);

    rlt::print(device, output);
    rlt::print(device, output_sequential);
    auto abs_diff = rlt::abs_diff(device, output, output_sequential);
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
                rlt::backward(device, sequential_model, input, d_output, sequential_buffer);
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
                rlt::backward_full(device, sequential_model, input, d_output, d_input, sequential_buffer);
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