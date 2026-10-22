#include <rl_tools/operations/cpu.h>

#include "simulator_matrix_physics_contract.h"

#include <memory>
#include <vector>

namespace rl_tools::hyperlux::benchmark::detail {
    using namespace paper;
    using DEVICE = devices::DefaultCPU;

    struct PhysicsSimulationImpl {
        DEVICE device;
        DEVICE::SPEC::RANDOM::ENGINE<> rng;
        float fov_degrees;
        float aspect;
        std::vector<ENVIRONMENT> environments;
        std::vector<PARAMETERS> parameters;
        std::vector<STATE> states;
        std::vector<STATE> initial_states;
        std::vector<hyperlux::Camera<float>> scratch_cameras;
    };

    void free_physics_simulation(PhysicsSimulation& simulation) {
        auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
        if(impl == nullptr) {
            return;
        }
        free(impl->device, impl->rng);
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
        impl->fov_degrees = fov_degrees;
        impl->aspect = aspect;
        impl->environments.resize(static_cast<std::size_t>(num_envs));
        impl->parameters.assign(static_cast<std::size_t>(num_envs), NOMINAL_PARAMETERS);
        impl->states.resize(static_cast<std::size_t>(num_envs));
        impl->scratch_cameras.resize(static_cast<std::size_t>(num_envs));
        init(impl->device);
        malloc(impl->device, impl->rng);
        init(impl->device, impl->rng, seed);
        for(int env_i = 0; env_i < num_envs; env_i++) {
            impl->environments[env_i].parameters = NOMINAL_PARAMETERS;
            impl->states[env_i] = paper_benchmark_initial_state(impl->device, position, poses[env_i]);
        }
        impl->initial_states = impl->states;
        simulation.impl = impl.release();
        return true;
    }

    bool physics_step_cameras(PhysicsSimulation& simulation, hyperlux::Camera<float>* destination, int iteration) {
        auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
        if(impl == nullptr) {
            simulation.error = "Physics simulation is not initialized";
            return false;
        }
        auto* cameras = destination == nullptr ? impl->scratch_cameras.data() : destination;
        for(std::size_t env_i = 0; env_i < impl->states.size(); env_i++) {
            paper_benchmark_step(impl->device, impl->environments[env_i], impl->parameters[env_i], impl->states[env_i], impl->initial_states[env_i], static_cast<int>(env_i), iteration, impl->rng);
            cameras[env_i] = paper_benchmark_camera(impl->device, impl->states[env_i], impl->fov_degrees, impl->aspect);
        }
        return true;
    }

    bool synchronize_physics(PhysicsSimulation&) {
        return true;
    }
}
