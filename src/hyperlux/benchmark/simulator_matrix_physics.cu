#include <rl_tools/operations/cuda.h>

#include "simulator_matrix_physics_contract.h"

#include <memory>
#include <vector>

namespace rl_tools::hyperlux::benchmark::detail {
    using namespace paper;
    using DEVICE = devices::CUDA<devices::DefaultCUDASpecification>;
    using DEVICE_TAG = devices::cuda::TAG<DEVICE, true>;

    struct PhysicsSimulationImpl {
        int num_envs;
        float fov_degrees;
        float aspect;
        std::uint32_t seed;
        ENVIRONMENT* environments = nullptr;
        PARAMETERS* parameters = nullptr;
        STATE* states = nullptr;
        STATE* initial_states = nullptr;
        hyperlux::Camera<float>* scratch_cameras = nullptr;
    };

    bool cuda_ok(PhysicsSimulation& simulation, cudaError_t status, const char* operation) {
        if(status == cudaSuccess) {
            return true;
        }
        simulation.error = std::string(operation) + ": " + cudaGetErrorString(status);
        return false;
    }

    __global__ void step_cameras_kernel(DEVICE_TAG device, ENVIRONMENT* environments, PARAMETERS* parameters, STATE* states, const STATE* initial_states, hyperlux::Camera<float>* cameras, int num_envs, float fov_degrees, float aspect, int iteration, std::uint32_t seed) {
        const int env_i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if(env_i >= num_envs) {
            return;
        }
        STATE state = states[env_i];
        curandState rng;
        if(parameters[env_i].mdp.action_noise.normalized_rpm != 0) {
            curand_init(static_cast<unsigned long long>(seed), static_cast<unsigned long long>(env_i), static_cast<unsigned long long>(iteration), &rng);
        }
        paper_benchmark_step(device, environments[env_i], parameters[env_i], state, initial_states[env_i], env_i, iteration, rng);
        states[env_i] = state;
        cameras[env_i] = paper_benchmark_camera(device, state, fov_degrees, aspect);
    }

    void free_physics_simulation(PhysicsSimulation& simulation) {
        auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
        if(impl == nullptr) {
            return;
        }
        cudaFree(impl->environments);
        cudaFree(impl->parameters);
        cudaFree(impl->states);
        cudaFree(impl->initial_states);
        cudaFree(impl->scratch_cameras);
        delete impl;
        simulation.impl = nullptr;
    }

    bool init_physics_simulation(PhysicsSimulation& simulation, int num_envs, const float position[3], const PhysicsCameraPose* poses, float fov_degrees, float aspect, std::uint32_t seed) {
        simulation.error.clear();
        if(num_envs <= 0 || poses == nullptr || position == nullptr) {
            simulation.error = "Physics initialization requires a positive camera count, position and poses";
            return false;
        }
        free_physics_simulation(simulation);
        auto impl = std::make_unique<PhysicsSimulationImpl>();
        impl->num_envs = num_envs;
        impl->fov_degrees = fov_degrees;
        impl->aspect = aspect;
        impl->seed = seed;
        DEVICE device;
        std::vector<ENVIRONMENT> environments(static_cast<std::size_t>(num_envs));
        std::vector<PARAMETERS> parameters(static_cast<std::size_t>(num_envs), NOMINAL_PARAMETERS);
        std::vector<STATE> states(static_cast<std::size_t>(num_envs));
        for(int env_i = 0; env_i < num_envs; env_i++) {
            environments[env_i].parameters = NOMINAL_PARAMETERS;
            states[env_i] = paper_benchmark_initial_state(device, position, poses[env_i]);
        }
        const auto environment_bytes = static_cast<std::size_t>(num_envs) * sizeof(ENVIRONMENT);
        const auto parameter_bytes = static_cast<std::size_t>(num_envs) * sizeof(PARAMETERS);
        const auto state_bytes = static_cast<std::size_t>(num_envs) * sizeof(STATE);
        simulation.impl = impl.release();
        auto& storage = *static_cast<PhysicsSimulationImpl*>(simulation.impl);
        const bool allocated =
            cuda_ok(simulation, cudaMalloc(&storage.environments, environment_bytes), "cudaMalloc environments") &&
            cuda_ok(simulation, cudaMalloc(&storage.parameters, parameter_bytes), "cudaMalloc parameters") &&
            cuda_ok(simulation, cudaMalloc(&storage.states, state_bytes), "cudaMalloc states") &&
            cuda_ok(simulation, cudaMalloc(&storage.initial_states, state_bytes), "cudaMalloc initial states") &&
            cuda_ok(simulation, cudaMalloc(&storage.scratch_cameras, static_cast<std::size_t>(num_envs) * sizeof(hyperlux::Camera<float>)), "cudaMalloc scratch cameras");
        if(!allocated) {
            free_physics_simulation(simulation);
            return false;
        }
        const bool copied =
            cuda_ok(simulation, cudaMemcpy(storage.environments, environments.data(), environment_bytes, cudaMemcpyHostToDevice), "cudaMemcpy environments") &&
            cuda_ok(simulation, cudaMemcpy(storage.parameters, parameters.data(), parameter_bytes, cudaMemcpyHostToDevice), "cudaMemcpy parameters") &&
            cuda_ok(simulation, cudaMemcpy(storage.states, states.data(), state_bytes, cudaMemcpyHostToDevice), "cudaMemcpy states") &&
            cuda_ok(simulation, cudaMemcpy(storage.initial_states, states.data(), state_bytes, cudaMemcpyHostToDevice), "cudaMemcpy initial states");
        if(!copied) {
            free_physics_simulation(simulation);
            return false;
        }
        return true;
    }

    bool physics_step_cameras(PhysicsSimulation& simulation, void* destination, void* stream, int iteration) {
        auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
        if(impl == nullptr) {
            simulation.error = "Physics simulation is not initialized";
            return false;
        }
        constexpr int BLOCK = 256;
        const int grid = (impl->num_envs + BLOCK - 1) / BLOCK;
        auto* cameras = destination == nullptr ? impl->scratch_cameras : static_cast<hyperlux::Camera<float>*>(destination);
        step_cameras_kernel<<<grid, BLOCK, 0, reinterpret_cast<cudaStream_t>(stream)>>>(DEVICE_TAG{}, impl->environments, impl->parameters, impl->states, impl->initial_states, cameras, impl->num_envs, impl->fov_degrees, impl->aspect, iteration, impl->seed);
        return cuda_ok(simulation, cudaPeekAtLastError(), "step_cameras_kernel launch");
    }

    bool synchronize_physics(PhysicsSimulation& simulation) {
        return cuda_ok(simulation, cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}
