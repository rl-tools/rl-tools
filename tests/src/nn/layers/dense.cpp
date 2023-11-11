
#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
DEVICE device;
TI seed = 1;
auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);

constexpr TI INPUT_DIM = 5;
constexpr TI OUTPUT_DIM = 5;
constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
using PARAMETER_TYPE = rlt::nn::parameters::Plain;

using LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE>;



#include <gtest/gtest.h>
#include <cstring>



TEST(RL_TOOLS_NN_LAYERS_DENSE, COPY_REGRESSION) {

    rlt::nn::layers::dense::Layer<LAYER_SPEC> layer;
    rlt::malloc(device, layer);
    rlt::init_kaiming(device, layer, rng);
    constexpr TI BATCH_SIZE = 1;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> output;
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::randn(device, input, rng);
    rlt::print(device, input);
    rlt::evaluate(device, layer, input, output);
    using PARAMETER_TYPE_2 = rlt::nn::parameters::Gradient;
    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE_2>;
    rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC> layer_2;
    rlt::malloc(device, layer_2);
    rlt::copy(device, device, layer, layer_2);
    rlt::zero_gradient(device, layer_2);
    auto abs_diff = rlt::abs_diff(device, layer, layer_2);
    EXPECT_EQ(abs_diff, 0);
}

TEST(RL_TOOLS_NN_LAYERS_DENSE, COPY_TIMING) {
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 100, 100>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 100, 100>> output;
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    constexpr TI ITERATIONS = 1000;
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            std::memcpy(output._data, input._data, decltype(input)::SPEC::SIZE_BYTES);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "memcpy: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    }
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            for(TI i = 0; i < decltype(input)::SPEC::SIZE; i++){
                output._data[i] = input._data[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "memcpy: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    }
    rlt::free(device, input);
    rlt::free(device, output);
}

