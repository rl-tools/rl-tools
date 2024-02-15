#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_cpu_mkl.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE_MKL = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GENERIC = rlt::devices::DefaultCPU;



#include <gtest/gtest.h>
#include <cstring>


template <typename T, typename TI, TI INPUT_DIM, TI OUTPUT_DIM, TI BATCH_SIZE>
void test(){
    DEVICE_MKL device_mkl;
    DEVICE_GENERIC device_generic;
    TI seed = 1;
    auto rng = rlt::random::default_engine(DEVICE_MKL::SPEC::RANDOM(), seed);

//    constexpr TI INPUT_DIM = 5;
//    constexpr TI OUTPUT_DIM = 5;
    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
    using PARAMETER_TYPE = rlt::nn::parameters::Plain;

    using LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE>;

    rlt::nn::layers::dense::Layer<LAYER_SPEC> layer;
    rlt::malloc(device_generic, layer);
    rlt::init_kaiming(device_generic, layer, rng);
//    constexpr TI BATCH_SIZE = 1;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> input;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM>> output_generic, output_mkl;
    rlt::malloc(device_generic, input);
    rlt::malloc(device_generic, output_generic);
    rlt::malloc(device_generic, output_mkl);
    rlt::randn(device_generic, input, rng);
    rlt::print(device_generic, input);
    rlt::evaluate(device_generic, layer, input, output_generic);
    rlt::evaluate(device_mkl, layer, input, output_mkl);
    auto diff = rlt::abs_diff(device_generic, output_generic, output_mkl);
    T diff_per_element = diff / (BATCH_SIZE * OUTPUT_DIM);
    std::cout << "Matrix mul diff: " << diff << " per element: " << diff_per_element << std::endl;
    if(rlt::utils::typing::is_same_v<T, float>){
        ASSERT_TRUE(diff_per_element < 1e-5);
    }else{
        ASSERT_TRUE(diff_per_element < 1e-10);
    }
}

TEST(RL_TOOLS_NN_LAYERS_DENSE, COPY_REGRESSION) {
    using TI = typename DEVICE_MKL::index_t;
    test<float, TI, 5, 5, 1>();
    test<float, TI, 5, 5, 2>();
    test<float, TI, 2, 5, 10>();
    test<float, TI, 3, 5, 100>();
    test<float, TI, 15, 16, 80>();
    test<float, TI, 15, 16, 81>();
}
