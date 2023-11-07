#include <backprop_tools/operations/cuda.h>


#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include "parameters.h"
namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#include <iostream>
#include <chrono>
#include <cassert>

using T = float;
constexpr T DT = 0.01;
constexpr size_t N_BLOCKS = 64;
constexpr size_t N_BLOCKS_CPU = 1;
constexpr size_t N_THREADS = 128;
constexpr size_t N_THREADS_CPU = 1;
constexpr size_t N_ITERATIONS = 1000000;

using DEVICE_GPU = bpt::devices::CUDA<bpt::devices::DefaultCUDASpecification>;
using DEVICE_CPU = bpt::devices::CPU<bpt::devices::DefaultCPUSpecification>;

using TI_GPU = DEVICE_CPU::index_t;
using TI_CPU = DEVICE_CPU::index_t;

using penv = parameters_fast_learning::environment<T, TI_GPU>;
using ENVIRONMENT = typename penv::ENVIRONMENT;
using STATE = typename ENVIRONMENT::State;


template <TI_GPU T_N_BLOCKS, TI_GPU T_BLOCK_DIM, TI_GPU T_N_ITERATIONS>
struct SimulateParallelSpec{
    static constexpr TI_GPU N_BLOCKS = T_N_BLOCKS;
    static constexpr TI_GPU BLOCK_DIM = T_BLOCK_DIM;
    static constexpr TI_GPU N_ITERATIONS = T_N_ITERATIONS;
};

template <typename DEVICE, typename SPEC, typename SPEC_SIMULATE>
void simulate_sequential(DEVICE& device, const bpt::rl::environments::Multirotor<SPEC>* envs, const typename bpt::rl::environments::Multirotor<SPEC>::State* states_input, typename bpt::rl::environments::Multirotor<SPEC>::State* next_states_output, const SPEC_SIMULATE) {
    using ENVIRONMENT = bpt::rl::environments::Multirotor<SPEC>;
    using STATE = typename ENVIRONMENT::State;
    using TI = typename DEVICE::index_t;
    for(TI block_i = 0; block_i < SPEC_SIMULATE::N_BLOCKS; block_i++){
        for(TI thread_i = 0; thread_i < SPEC_SIMULATE::BLOCK_DIM; thread_i++){
            const TI full_id = block_i * SPEC_SIMULATE::BLOCK_DIM + thread_i;
            const auto& env = envs[full_id];
            STATE state;
            STATE next_state;
            state = states_input[full_id];
            for(TI iteration_i=0; iteration_i<SPEC_SIMULATE::N_ITERATIONS; iteration_i++){
                T action[ENVIRONMENT::ACTION_DIM];
//        evaluate(policy, state, action);
                for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
                    action[action_i] = 0;
                }
                bpt::utils::integrators::rk4<DEVICE, T, typename SPEC::PARAMETERS, STATE, ENVIRONMENT::ACTION_DIM, bpt::rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, env.parameters, state, action, env.parameters.integration.dt, next_state);
                state = next_state;
            }
            next_states_output[full_id] = state;
        }
    }
}

template <typename DEVICE, typename SPEC, typename SPEC_SIMULATE>
__global__ void
__launch_bounds__(SPEC_SIMULATE::BLOCK_DIM)//, minBlocksPerMultiprocessor, maxBlocksPerCluster)
simulate_parallel(DEVICE& device, const bpt::rl::environments::Multirotor<SPEC>* envs, const typename bpt::rl::environments::Multirotor<SPEC>::State* states_input, typename bpt::rl::environments::Multirotor<SPEC>::State* next_states_output, const SPEC_SIMULATE) {
    using ENVIRONMENT = bpt::rl::environments::Multirotor<SPEC>;
    using STATE = typename ENVIRONMENT::State;
    using TI = typename DEVICE::index_t;
    const TI full_id = blockIdx.x * blockDim.x + threadIdx.x;
    const TI thread_id = threadIdx.x;
    const TI block_id = blockIdx.x;
    __shared__ ENVIRONMENT env;
    if(thread_id == 0){
        env = envs[block_id];
    }
    __syncthreads();
    STATE state;
    STATE next_state;
    state = states_input[full_id];
    for(TI iteration_i=0; iteration_i<SPEC_SIMULATE::N_ITERATIONS; iteration_i++){
        T action[ENVIRONMENT::ACTION_DIM];
//        evaluate(policy, state, action);
        for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
            action[action_i] = 0;
        }
        bpt::utils::integrators::rk4<DEVICE, T, typename SPEC::PARAMETERS, STATE, ENVIRONMENT::ACTION_DIM, bpt::rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, env.parameters, state, action, env.parameters.integration.dt, next_state);
        state = next_state;
    }
    next_states_output[full_id] = state;
}

int main(void) {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "\tName: " << prop.name << std::endl;
        std::cout << "\tCompute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "\tNumber of SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "\tRegisters per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "\tMax threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    }
    TI_CPU chosen_device = 0;
    cudaSetDevice(chosen_device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, chosen_device);


    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;

    STATE initial_states[N_BLOCKS][N_THREADS];
    STATE final_states_gpu[N_BLOCKS][N_THREADS];
    STATE final_states_cpu[N_BLOCKS][N_THREADS];
    ENVIRONMENT envs[N_BLOCKS][N_THREADS];
    for(TI_CPU block_i=0; block_i<N_BLOCKS; block_i++){
        for(TI_CPU thread_i=0; thread_i<N_THREADS; thread_i++){
            envs[block_i][thread_i].parameters = penv::parameters;
            envs[block_i][thread_i].parameters.integration.dt = DT;
            bpt::initial_state(device_cpu, envs[block_i][thread_i], initial_states[block_i][thread_i]);
        }
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        simulate_sequential(device_cpu, &envs[0][0], &initial_states[0][0], &final_states_cpu[0][0], SimulateParallelSpec<N_BLOCKS_CPU, N_THREADS_CPU, N_ITERATIONS>{});
        auto end = std::chrono::high_resolution_clock::now();

        double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Simulation time (CPU):  " << elapsedTime << " ms (" << N_BLOCKS_CPU * N_THREADS_CPU * N_ITERATIONS / (elapsedTime / 1000.0) / 1e6 << " Msteps/s)" << std::endl;
    }
    {
        ENVIRONMENT *d_envs;
        STATE* d_states;
        STATE* d_next_states;

        cudaMalloc((void **)&d_envs, N_BLOCKS * sizeof(ENVIRONMENT));
        cudaMemcpy(d_envs, envs, N_BLOCKS * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_states, N_BLOCKS * N_THREADS * sizeof(STATE));
        cudaMemcpy(d_states, &initial_states, N_BLOCKS * N_THREADS * sizeof(STATE), cudaMemcpyHostToDevice);

        cudaMemcpy(&initial_states, d_states, N_BLOCKS * N_THREADS * sizeof(STATE), cudaMemcpyDeviceToHost);

        cudaMalloc((void **)&d_next_states, N_BLOCKS * N_THREADS * sizeof(STATE));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        dim3 grid(N_BLOCKS);
        dim3 threadsPerBlock(N_THREADS);
        simulate_parallel<<<grid, threadsPerBlock>>>(device_gpu, d_envs, d_states, d_next_states, SimulateParallelSpec<N_BLOCKS, N_THREADS, N_ITERATIONS>{});
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        auto err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        }
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaDeviceSynchronize();
        cudaMemcpy(final_states_gpu, d_next_states, N_BLOCKS * N_THREADS * sizeof(STATE), cudaMemcpyDeviceToHost);
        cudaFree(d_envs); cudaFree(d_states); cudaFree(d_next_states);
        cudaDeviceSynchronize();

        std::cout << "Simulation time (GPU):  " << elapsedTime << " ms (" << N_BLOCKS * N_THREADS * N_ITERATIONS / (elapsedTime / 1000.0) / 1e6 << " Msteps/s, " << N_BLOCKS * N_THREADS * N_ITERATIONS * DT / (elapsedTime / 1000.0) / (365 * 24 * 3600) << " Years/s)" << std::endl;
    }

//    // Copy a & b from the host to the device
//    T diff = 0;
//    for(int i=N_BLOCKS-N_BLOCKS_EVAL; i < N_BLOCKS; i++){
//        for(int j=N_THREADS-N_THREADS_EVAL; j < N_THREADS; j++) {
//            for (int k = 0; k < STATE_DIM; k++) {
//                diff += std::fabs(state_cpu[i][j][k] - state_gpu[i][j][k]);
//            }
//        }
//    }
//    std::cout << "Average diff (cpu-gpu): " << diff/(T)(N_BLOCKS * N_THREADS) << std::endl;
//
//    std::cout << "Final state:" << std::endl;
//    std::cout.precision(17);
//    for(int i=0; i < STATE_DIM; i++){
//        std::cout << state_cpu[N_BLOCKS-1][N_THREADS-1][i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "Final state comparison cpu <-> gpu:" << std::endl;
//    for(int i=0; i < STATE_DIM; i++){
//        std::cout << state_cpu[N_BLOCKS-1][N_THREADS-1][i] - state_gpu[N_BLOCKS-1][N_THREADS-1][i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "Final state comparison gpu <-> jax:" << std::endl;
//    T acc = 0;
//    for(int i=0; i < N_BLOCKS; i++){
//        for(int j=0; j < N_THREADS; j++) {
//            for (int i = 0; i < STATE_DIM; i++) {
//                T diff = state_gpu[N_BLOCKS - 1][N_THREADS - 1][i] - expected_state2[i];
//                acc += std::abs(diff);
//                if(i == 0 && j == 0){
//                    std::cout << diff << " ";
//                }
//            }
//        }
//    }
//    std::cout << std::endl;
//    std::cout << "Final state comparison gpu <-> jax (cumulative): " << acc / ((T)N_BLOCKS * N_THREADS) << std::endl;


    return 0;
}
