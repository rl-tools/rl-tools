#pragma once

#include <cstdint>

namespace rl_tools::rendering::raytracing::benchmark {

struct PhysicsCameraPose {
    float direction[3];
    float up[3];
};

struct PhysicsSimulation {
    void* impl = nullptr;
};

bool init_physics_simulation(
    PhysicsSimulation& simulation,
    int num_envs,
    const float camera_position[3],
    const PhysicsCameraPose* poses,
    float fov,
    float aspect,
    std::uint32_t seed
);

bool physics_step_cameras(
    PhysicsSimulation& simulation,
    void* camera_buffer_device_ptr,
    void* cuda_stream,
    int iteration
);

void free_physics_simulation(PhysicsSimulation& simulation);

const char* physics_last_error();

} // namespace rl_tools::rendering::raytracing::benchmark
