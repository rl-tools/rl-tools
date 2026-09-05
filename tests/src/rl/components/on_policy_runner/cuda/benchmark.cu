#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/random/operations_generic_array.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <metra/metra.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#ifndef RL_TOOLS_BENCHMARK_ENVIRONMENTS
#define RL_TOOLS_BENCHMARK_ENVIRONMENTS 64
#endif
#ifndef RL_TOOLS_BENCHMARK_HIDDEN_DIM
#define RL_TOOLS_BENCHMARK_HIDDEN_DIM 32
#endif

namespace {
    using namespace rl_tools;
    using CPU = devices::DefaultCPU;
    using GPU = devices::DefaultCUDA;
    using TI = CPU::index_t;
    using TYPE_POLICY = numeric_types::Policy<float>;
    constexpr TI N = RL_TOOLS_BENCHMARK_ENVIRONMENTS;
    constexpr TI STEPS = 64;
    constexpr TI ROLLOUTS = 10;
    using ENVIRONMENT = rl::environments::Pendulum<rl::environments::pendulum::Specification<float, TI>>;
    using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<ENVIRONMENT, N>>;
    using RNG = devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, 1024>>;
    using ACTOR_CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, 3, RL_TOOLS_BENCHMARK_HIDDEN_DIM, nn::activation_functions::TANH, nn::activation_functions::IDENTITY>;
    using ACTOR = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<nn_models::mlp_unconditional_stddev::BindConfiguration<ACTOR_CONFIG>>, tensor::Shape<TI, 1, N, 3>>;
    using RUNNER_SPEC = rl::components::on_policy_runner::Specification<TYPE_POLICY, BATCH, ACTOR::State<>, ENVIRONMENT::Observation, ENVIRONMENT::ObservationPrivileged, float, float, 0>;
    using DATASET_SPEC = rl::components::on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS>;

    void check(cudaError_t status) {
        if(status != cudaSuccess) {
            std::cerr << cudaGetErrorString(status) << '\n';
            std::exit(1);
        }
    }
}

int main(int argc, char** argv) {
    using namespace rl_tools;
    const bool graph_mode = argc == 2 && std::string(argv[1]) == "--graph";
    if(argc > 1 && !graph_mode) {
        std::cerr << "Usage: " << argv[0] << " [--graph]\n";
        return 1;
    }
    CPU cpu;
    GPU gpu;
    init(gpu);
    RNG rng, initial_rng;
    ACTOR actor, host_actor;
    ACTOR::Buffer<> actor_buffer;
    rl::components::OnPolicyRunner<RUNNER_SPEC> runner;
    rl::components::on_policy_runner::Buffer<RUNNER_SPEC> buffer;
    rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, host_dataset;
    BATCH environment;
    malloc(cpu, initial_rng); init(cpu, initial_rng, 42);
    malloc(cpu, host_actor); init_weights(cpu, host_actor, initial_rng);
    malloc(gpu, rng); init(cpu, initial_rng, 7); copy(cpu, gpu, initial_rng, rng);
    malloc(gpu, actor); copy(cpu, gpu, host_actor, actor);
    malloc(gpu, actor_buffer); malloc(gpu, runner); malloc(gpu, buffer);
    malloc(gpu, dataset); malloc(cpu, host_dataset);
    malloc(gpu, environment); init(gpu, environment); init(gpu, runner, environment, rng);
    auto collect_rollout = [&]() { collect(gpu, dataset, runner, buffer, environment, actor, actor_buffer, rng); };
    collect_rollout(); check(cudaDeviceSynchronize());
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if(graph_mode) {
        gpu.graph_capture_active = true;
        check(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeGlobal));
        collect_rollout();
        check(cudaStreamEndCapture(gpu.stream, &graph));
        gpu.graph_capture_active = false;
        check(cudaGraphInstantiate(&executable, graph, 0));
    }
    check(cudaProfilerStart());
    double samples[5];
    for(auto& seconds: samples) {
        const auto start = std::chrono::steady_clock::now();
        for(TI i = 0; i < ROLLOUTS; i++) {
            if(graph_mode) {
                check(cudaGraphLaunch(executable, gpu.stream));
            }
            else collect_rollout();
        }
        check(cudaDeviceSynchronize());
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    check(cudaProfilerStop());
    std::sort(samples, samples + 5);
    copy(gpu, cpu, dataset.scalar_data, host_dataset.scalar_data);
    copy(gpu, cpu, dataset.all_observations, host_dataset.all_observations);
    std::uint64_t hash = 14695981039346656037ULL;
    auto hash_float = [&](float value) {
        unsigned char bytes[sizeof(float)];
        std::memcpy(bytes, &value, sizeof(float));
        for(auto byte: bytes) { hash ^= byte; hash *= 1099511628211ULL; }
    };
    double reward_sum = 0;
    for(TI i = 0; i < N * STEPS; i++) {
        const float reward = get(host_dataset.rewards, i, 0);
        reward_sum += reward;
        hash_float(reward); hash_float(get(host_dataset.actions, i, 0));
    }
    for(TI i = 0; i < N * (STEPS + 1); i++) for(TI j = 0; j < 3; j++) hash_float(get(cpu, host_dataset.all_observations, i, j));
    const double throughput = N * STEPS * ROLLOUTS / samples[2];
    std::cout << std::setprecision(12) << "environments=" << N << " graph=" << graph_mode << " median_seconds=" << samples[2]
              << " transitions_per_second=" << throughput << " min=" << samples[0] << " max=" << samples[4]
              << " reward_sum=" << reward_sum << " trajectory_hash=" << hash << '\n';
    const std::string metric = "on_policy_runner/collect/" + std::to_string(N) + (graph_mode ? "/graph" : "/eager");
    metra::log(metric + "/transitions_per_second", throughput);
    metra::log(metric + "/seconds", samples[2]);
    if(graph_mode) { check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); }
    free(cpu, host_dataset); free(gpu, dataset); free(gpu, buffer); free(gpu, runner);
    free(gpu, environment); free(gpu, actor_buffer); free(gpu, actor); free(gpu, rng);
    free(cpu, host_actor); free(cpu, initial_rng);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    if(cudnnDestroy(gpu.cudnn_handle) != CUDNN_STATUS_SUCCESS) return 1;
#endif
    if(cublasDestroy(gpu.handle) != CUBLAS_STATUS_SUCCESS) return 1;
    check(cudaStreamDestroy(gpu.stream));
}
