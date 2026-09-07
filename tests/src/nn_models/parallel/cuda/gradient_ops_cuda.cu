// add_gradient / copy_gradient on CUDA: the all-reduce building blocks used by multi-GPU targets.
// The shadow gradient is copied across two devices::CUDA instances (two GPUs when available) and
// accumulated on the first one; the result must match the CPU computation bit for bit
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

namespace {
    using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
    using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
    using T = float;
    using TI = typename DEVICE_CPU::index_t;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    constexpr TI SEQUENCE_LENGTH = 3;
    constexpr TI BATCH_SIZE = 4;
    using INPUT_SHAPE_A = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 5>;
    using INPUT_SHAPE_B = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 3>;
    using LAYER_A_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_A = rlt::nn::layers::dense::BindConfiguration<LAYER_A_CONFIG>;
    using MODULE_A = rlt::nn_models::sequential::Module<LAYER_A>;
    using LAYER_B_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 6, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using LAYER_B = rlt::nn::layers::dense::BindConfiguration<LAYER_B_CONFIG>;
    using MODULE_B = rlt::nn_models::sequential::Module<LAYER_B>;
    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 4>;
    using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
    using HEAD_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_CONFIG>;
    using HEAD_MODULE = rlt::nn_models::sequential::Module<GRU, HEAD_DENSE>;
    using BRANCH_A = rlt::nn_models::parallel::Branch<MODULE_A, INPUT_SHAPE_A>;
    using BRANCH_B = rlt::nn_models::parallel::Branch<MODULE_B, INPUT_SHAPE_B>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, HEAD_MODULE, BRANCH_A, BRANCH_B>;

    template<typename F, typename... MODELS>
    void for_each_gradient(F&& f, MODELS&... models){
        auto apply = [&](auto&& accessor){ f(accessor(models)...); };
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<0>(m.pipelines)).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<0>(m.pipelines)).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<1>(m.pipelines)).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(rlt::get<1>(m.pipelines)).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).weights_input.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).biases_input.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).weights_hidden.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).biases_hidden.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m.head).initial_hidden_state.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<1>(m.head).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<1>(m.head).biases.gradient; });
    }
}

namespace {
    constexpr TI CONV_BATCH_SIZE = 2;
    using CONV_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, CONV_BATCH_SIZE, 6, 6, 3>;
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
    using CONV_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>>;
    using CONV_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CONV_DENSE = rlt::nn::layers::dense::BindConfiguration<CONV_DENSE_CONFIG>;
    using CONV_MODULE = rlt::nn_models::sequential::Module<CONV, CONV_FLATTEN, CONV_DENSE>;
    using CONV_MODEL = rlt::nn_models::sequential::Build<CAPABILITY, CONV_MODULE, CONV_INPUT_SHAPE>;

    template<typename F, typename... MODELS>
    void for_each_conv_gradient(F&& f, MODELS&... models){
        auto apply = [&](auto&& accessor){ f(accessor(models)...); };
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<0>(m).biases.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<2>(m).weights.gradient; });
        apply([](auto& m) -> auto& { return rlt::nn_models::sequential::layer<2>(m).biases.gradient; });
    }
}

// conv2d gradients are 4-D tensors: covers the flattened dense fast path of the CUDA elementwise ops
TEST(RL_TOOLS_NN_MODELS_PARALLEL_CUDA, ADD_GRADIENT_CONV2D){
    int device_count = 0;
    if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE_CPU device_cpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device_cpu, rng);
    rlt::init(device_cpu, rng, 5);

    CONV_MODEL a_cpu, b_cpu, expected_cpu, result_cpu;
    rlt::malloc(device_cpu, a_cpu);
    rlt::malloc(device_cpu, b_cpu);
    rlt::malloc(device_cpu, expected_cpu);
    rlt::malloc(device_cpu, result_cpu);
    rlt::init_weights(device_cpu, a_cpu, rng);
    rlt::init_weights(device_cpu, b_cpu, rng);
    for_each_conv_gradient([&](auto& ta, auto& tb){ rlt::randn(device_cpu, ta, rng); rlt::randn(device_cpu, tb, rng); }, a_cpu, b_cpu);
    rlt::copy(device_cpu, device_cpu, a_cpu, expected_cpu);
    rlt::add_gradient(device_cpu, b_cpu, expected_cpu);

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    DEVICE_CUDA device_gpu;
    rlt::init(device_gpu);
    CONV_MODEL a_gpu, b_gpu;
    rlt::malloc(device_gpu, a_gpu);
    rlt::malloc(device_gpu, b_gpu);
    rlt::copy(device_cpu, device_gpu, a_cpu, a_gpu);
    rlt::copy(device_cpu, device_gpu, b_cpu, b_gpu);
    rlt::add_gradient(device_gpu, b_gpu, a_gpu);
    ASSERT_EQ(cudaStreamSynchronize(device_gpu.stream), cudaSuccess);
    rlt::copy(device_gpu, device_cpu, a_gpu, result_cpu);

    for_each_conv_gradient([&](auto& tr, auto& te){ EXPECT_EQ(rlt::abs_diff(device_cpu, tr, te), (T)0); }, result_cpu, expected_cpu);

    rlt::free(device_gpu, a_gpu);
    rlt::free(device_gpu, b_gpu);
    rlt::free(device_cpu, a_cpu);
    rlt::free(device_cpu, b_cpu);
    rlt::free(device_cpu, expected_cpu);
    rlt::free(device_cpu, result_cpu);
    rlt::free(device_cpu, rng);
}

TEST(RL_TOOLS_NN_MODELS_PARALLEL_CUDA, ALL_REDUCE_GRADIENT_ACROSS_DEVICES){
    int device_count = 0;
    if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    const int peer_ordinal = device_count > 1 ? 1 : 0;

    DEVICE_CPU device_cpu;
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device_cpu, rng);
    rlt::init(device_cpu, rng, 3);

    MODEL a_cpu, b_cpu, expected_cpu, result_cpu;
    rlt::malloc(device_cpu, a_cpu);
    rlt::malloc(device_cpu, b_cpu);
    rlt::malloc(device_cpu, expected_cpu);
    rlt::malloc(device_cpu, result_cpu);
    rlt::init_weights(device_cpu, a_cpu, rng);
    rlt::init_weights(device_cpu, b_cpu, rng);
    for_each_gradient([&](auto& ta, auto& tb){ rlt::randn(device_cpu, ta, rng); rlt::randn(device_cpu, tb, rng); }, a_cpu, b_cpu);
    rlt::copy(device_cpu, device_cpu, a_cpu, expected_cpu);
    rlt::add_gradient(device_cpu, b_cpu, expected_cpu);

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    DEVICE_CUDA device_gpu_0;
    rlt::init(device_gpu_0);
    MODEL a_gpu, shadow_gpu_0;
    rlt::malloc(device_gpu_0, a_gpu);
    rlt::malloc(device_gpu_0, shadow_gpu_0);
    rlt::copy(device_cpu, device_gpu_0, a_cpu, a_gpu);

    ASSERT_EQ(cudaSetDevice(peer_ordinal), cudaSuccess);
    DEVICE_CUDA device_gpu_1;
    rlt::init(device_gpu_1);
    MODEL b_gpu;
    rlt::malloc(device_gpu_1, b_gpu);
    rlt::copy(device_cpu, device_gpu_1, b_cpu, b_gpu);

    // push protocol: the transfer is enqueued on the source stream, the consumer waits for it
    rlt::copy_gradient(device_gpu_1, device_gpu_0, b_gpu, shadow_gpu_0);
    ASSERT_EQ(cudaStreamSynchronize(device_gpu_1.stream), cudaSuccess);

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    rlt::add_gradient(device_gpu_0, shadow_gpu_0, a_gpu);
    ASSERT_EQ(cudaStreamSynchronize(device_gpu_0.stream), cudaSuccess);
    rlt::copy(device_gpu_0, device_cpu, a_gpu, result_cpu);

    for_each_gradient([&](auto& tr, auto& te){ EXPECT_EQ(rlt::abs_diff(device_cpu, tr, te), (T)0); }, result_cpu, expected_cpu);
    EXPECT_EQ(rlt::abs_diff(device_cpu, result_cpu, expected_cpu), (T)0);

    rlt::free(device_gpu_0, a_gpu);
    rlt::free(device_gpu_0, shadow_gpu_0);
    ASSERT_EQ(cudaSetDevice(peer_ordinal), cudaSuccess);
    rlt::free(device_gpu_1, b_gpu);
    rlt::free(device_cpu, a_cpu);
    rlt::free(device_cpu, b_cpu);
    rlt::free(device_cpu, expected_cpu);
    rlt::free(device_cpu, result_cpu);
    rlt::free(device_cpu, rng);
}
