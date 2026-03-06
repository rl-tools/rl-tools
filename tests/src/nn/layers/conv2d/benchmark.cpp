#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
#include <rl_tools/nn/layers/conv2d/operations_cpu_mkl.h>
#endif

#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using DEVICE_GENERIC = rlt::devices::DefaultCPU;
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
using DEVICE_MKL = rlt::devices::DEVICE_FACTORY<>;
#endif
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE_GENERIC::index_t;

struct BenchmarkResult{
    std::string label;
    double generic_us;
    double dnnl_us;
    double speedup;
};

template<typename CONV_CONFIG, TI BATCH_SIZE, TI HEIGHT, TI WIDTH, TI INPUT_CHANNELS, TI ITERATIONS>
BenchmarkResult benchmark_case(const std::string& label){
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HEIGHT, WIDTH, INPUT_CHANNELS>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
    using LAYER_TYPE = rlt::nn::layers::conv2d::Layer<CONV_CONFIG, CAPABILITY, INPUT_SHAPE>;

    constexpr TI OUTPUT_HEIGHT = LAYER_TYPE::OUTPUT_HEIGHT;
    constexpr TI OUTPUT_WIDTH = LAYER_TYPE::OUTPUT_WIDTH;
    constexpr TI OUTPUT_CHANNELS = CONV_CONFIG::OUTPUT_CHANNELS;

    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNELS>;
    using D_INPUT_SHAPE = INPUT_SHAPE;

    // ======================== Generic (CPU) ========================
    double generic_fwd_us, generic_bwd_us, generic_eval_us;
    {
        DEVICE_GENERIC device;
        DEVICE_GENERIC::SPEC::RANDOM::ENGINE<> rng;
        rlt::malloc(device, rng);
        rlt::init(device, rng, 42);

        LAYER_TYPE layer;
        typename LAYER_TYPE::template Buffer<true> buffer;
        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
        rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output, d_output;
        rlt::Tensor<rlt::tensor::Specification<T, TI, D_INPUT_SHAPE>> d_input;

        rlt::malloc(device, layer);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input);
        rlt::malloc(device, output);
        rlt::malloc(device, d_output);
        rlt::malloc(device, d_input);
        rlt::init_weights(device, layer, rng);
        rlt::randn(device, input, rng);
        rlt::randn(device, d_output, rng);

        // Warmup
        rlt::forward(device, layer, input, buffer, rng);
        rlt::evaluate(device, layer, input, output, buffer, rng);

        // Evaluate benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::evaluate(device, layer, input, output, buffer, rng);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        generic_eval_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        // Forward benchmark
        t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::forward(device, layer, input, buffer, rng);
        }
        t1 = std::chrono::high_resolution_clock::now();
        generic_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        // Backward benchmark
        rlt::zero_gradient(device, layer);
        rlt::backward_full(device, layer, input, d_output, d_input, buffer);
        t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::zero_gradient(device, layer);
            rlt::backward_full(device, layer, input, d_output, d_input, buffer);
        }
        t1 = std::chrono::high_resolution_clock::now();
        generic_bwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        rlt::free(device, layer);
        rlt::free(device, buffer);
        rlt::free(device, input);
        rlt::free(device, output);
        rlt::free(device, d_output);
        rlt::free(device, d_input);
    }

    // ======================== oneDNN (CPU_MKL) ========================
    double dnnl_eval_us = 0, dnnl_fwd_us = 0, dnnl_bwd_us = 0;
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
    {
        DEVICE_MKL device;
        DEVICE_MKL::SPEC::RANDOM::ENGINE<> rng;
        rlt::malloc(device, rng);
        rlt::init(device, rng, 42);

        LAYER_TYPE layer;
        typename LAYER_TYPE::template Buffer<true> buffer;
        rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
        rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output, d_output;
        rlt::Tensor<rlt::tensor::Specification<T, TI, D_INPUT_SHAPE>> d_input;

        rlt::malloc(device, layer);
        rlt::malloc(device, buffer);
        rlt::malloc(device, input);
        rlt::malloc(device, output);
        rlt::malloc(device, d_output);
        rlt::malloc(device, d_input);
        rlt::init_weights(device, layer, rng);
        rlt::randn(device, input, rng);
        rlt::randn(device, d_output, rng);

        // Warmup
        rlt::forward(device, layer, input, buffer, rng);
        rlt::evaluate(device, layer, input, output, buffer, rng);

        // Evaluate benchmark (pure DNNL, no copy/pre_activations)
        auto t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::evaluate(device, layer, input, output, buffer, rng);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        dnnl_eval_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        // Forward benchmark
        t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::forward(device, layer, input, buffer, rng);
        }
        t1 = std::chrono::high_resolution_clock::now();
        dnnl_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        // Backward benchmark
        rlt::zero_gradient(device, layer);
        rlt::backward_full(device, layer, input, d_output, d_input, buffer);
        t0 = std::chrono::high_resolution_clock::now();
        for(TI i = 0; i < ITERATIONS; i++){
            rlt::zero_gradient(device, layer);
            rlt::backward_full(device, layer, input, d_output, d_input, buffer);
        }
        t1 = std::chrono::high_resolution_clock::now();
        dnnl_bwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERATIONS;

        rlt::free(device, layer);
        rlt::free(device, buffer);
        rlt::free(device, input);
        rlt::free(device, output);
        rlt::free(device, d_output);
        rlt::free(device, d_input);
    }
#endif

    std::cout << std::left << std::setw(30) << label;
    std::cout << std::right << std::fixed << std::setprecision(0);
    std::cout << std::setw(10) << generic_eval_us;
    std::cout << std::setw(10) << generic_fwd_us;
    std::cout << std::setw(10) << generic_bwd_us;
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
    std::cout << std::setw(10) << dnnl_eval_us;
    std::cout << std::setw(10) << dnnl_fwd_us;
    std::cout << std::setw(10) << dnnl_bwd_us;
    double total_generic = generic_eval_us + generic_fwd_us + generic_bwd_us;
    double total_dnnl = dnnl_eval_us + dnnl_fwd_us + dnnl_bwd_us;
    std::cout << std::setw(8) << std::setprecision(1) << (total_dnnl > 0 ? total_generic / total_dnnl : 0) << "x";
#endif
    std::cout << std::endl;

    return {label, 0, 0, 0};
}

int main(){
    std::cout << std::endl;
    std::cout << "Conv2d Performance Benchmark: Generic vs oneDNN (us per call)" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Config";
    std::cout << std::right;
    std::cout << std::setw(10) << "G Eval";
    std::cout << std::setw(10) << "G Fwd";
    std::cout << std::setw(10) << "G Bwd";
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
    std::cout << std::setw(10) << "D Eval";
    std::cout << std::setw(10) << "D Fwd";
    std::cout << std::setw(10) << "D Bwd";
    std::cout << std::setw(10) << "Speedup";
#endif
    std::cout << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    // Small: typical early conv layer
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 32, 32, 32, 3, 100>("B32 32x32 3->16 k3 RELU");
    }
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 32, 32, 32, 3, 100>("B32 32x32 3->16 k3s2 RELU");
    }

    // Medium: mid-network
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 32, 16, 16, 16, 100>("B32 16x16 16->32 k3 RELU");
    }
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 32, 16, 16, 16, 100>("B32 16x16 16->32 k3s2 RELU");
    }

    // Large batch: PPO-style
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 256, 32, 32, 3, 20>("B256 32x32 3->16 k3s2 RELU");
    }
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 256, 16, 16, 16, 20>("B256 16x16 16->32 k3s2 RELU");
    }

    // Pointwise 1x1
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 1, 1, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
        benchmark_case<CONFIG, 32, 8, 8, 16, 200>("B32 8x8 16->32 k1 IDENTITY");
    }

    // 5x5 kernel
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 5, 5, 2, 2, 2, 2, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
        benchmark_case<CONFIG, 32, 16, 16, 8, 100>("B32 16x16 8->16 k5s2 IDENTITY");
    }

    // Large channels
    {
        using CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
        benchmark_case<CONFIG, 16, 8, 8, 64, 100>("B16 8x8 64->64 k3 RELU");
    }

    std::cout << std::string(80 + 34, '=') << std::endl;
    return 0;
}
