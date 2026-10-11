#pragma once

#include <rl_tools/rendering/camera.h>

#include <cstdint>
#include <string>

namespace rl_tools::rendering::raytracing::benchmark {
    struct PhysicsCameraPose {
        float direction[3];
        float up[3];
    };

    struct PhysicsSimulation {
        void* impl = nullptr;
        std::string error;
    };

    namespace detail {
        bool init_physics_simulation(PhysicsSimulation&, int, const float[3], const PhysicsCameraPose*, float, float, std::uint32_t);
        bool physics_step_cameras(PhysicsSimulation&, void*, void*, int);
        bool physics_step_cameras(PhysicsSimulation&, rendering::Camera<float>*, int);
        bool synchronize_physics(PhysicsSimulation&);
        void free_physics_simulation(PhysicsSimulation&);
    }
}

namespace rl_tools {
    template <typename DEVICE>
    bool init_physics_simulation(DEVICE&, rendering::raytracing::benchmark::PhysicsSimulation& simulation, int num_envs, const float position[3], const rendering::raytracing::benchmark::PhysicsCameraPose* poses, float fov_degrees, float aspect, std::uint32_t seed) {
        return rendering::raytracing::benchmark::detail::init_physics_simulation(simulation, num_envs, position, poses, fov_degrees, aspect, seed);
    }

    template <typename DEVICE>
    bool physics_step_cameras(DEVICE&, rendering::raytracing::benchmark::PhysicsSimulation& simulation, void* destination, void* stream, int iteration) {
        return rendering::raytracing::benchmark::detail::physics_step_cameras(simulation, destination, stream, iteration);
    }

    template <typename DEVICE>
    bool physics_step_cameras(DEVICE&, rendering::raytracing::benchmark::PhysicsSimulation& simulation, rendering::Camera<float>* destination, int iteration) {
        return rendering::raytracing::benchmark::detail::physics_step_cameras(simulation, destination, iteration);
    }

    template <typename DEVICE>
    bool synchronize_physics(DEVICE&, rendering::raytracing::benchmark::PhysicsSimulation& simulation) {
        return rendering::raytracing::benchmark::detail::synchronize_physics(simulation);
    }

    template <typename DEVICE>
    void free_physics_simulation(DEVICE&, rendering::raytracing::benchmark::PhysicsSimulation& simulation) {
        rendering::raytracing::benchmark::detail::free_physics_simulation(simulation);
    }

    template <typename DEVICE>
    const char* physics_last_error(DEVICE&, const rendering::raytracing::benchmark::PhysicsSimulation& simulation) {
        return simulation.error.c_str();
    }
}
