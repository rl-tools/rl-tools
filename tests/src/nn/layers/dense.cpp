
#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/nn/layers/dense/operations_cpu.h>
namespace bpt = backprop_tools;
using DEVICE = bpt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
DEVICE device;
TI seed = 1;
auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);

constexpr TI INPUT_DIM = 5;
constexpr TI OUTPUT_DIM = 5;
constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::RELU;
using PARAMETER_TYPE = bpt::nn::parameters::Plain;

using LAYER_SPEC = bpt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE>;



#include <gtest/gtest.h>
#include <cstring>



TEST(BACKPROP_TOOLS_NN_LAYERS_DENSE, COPY_REGRESSION) {

    bpt::nn::layers::dense::Layer<LAYER_SPEC> layer;
    bpt::malloc(device, layer);
    bpt::init_kaiming(device, layer, rng);
    constexpr TI BATCH_SIZE = 1;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> output;
    bpt::malloc(device, input);
    bpt::malloc(device, output);
    bpt::randn(device, input, rng);
    bpt::print(device, input);
    bpt::evaluate(device, layer, input, output);
    using PARAMETER_TYPE_2 = bpt::nn::parameters::Gradient;
    using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE_2>;
    bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC> layer_2;
    bpt::malloc(device, layer_2);
    bpt::copy(device, device, layer_2, layer);
    bpt::zero_gradient(device, layer_2);
    auto abs_diff = bpt::abs_diff(device, layer, layer_2);
    EXPECT_EQ(abs_diff, 0);
}

TEST(BACKPROP_TOOLS_NN_LAYERS_DENSE, COPY_TIMING) {
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> input;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> output;
    bpt::malloc(device, input);
    bpt::malloc(device, output);
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
    bpt::free(device, input);
    bpt::free(device, output);
}

