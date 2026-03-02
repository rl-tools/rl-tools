#include "environment/environment.h"
#include "environment/operations.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <type_traits>

namespace ex = raytracing_example;

struct Device {};

int main() {
    using T = float;
    using TI = int;
    static constexpr TI NUM_ENVS = 4096;
    using SPEC = ex::Specification<T, TI, NUM_ENVS, 64, 64, 64>;
    static_assert(std::is_standard_layout_v<ex::Parameters<SPEC>>);
    static_assert(std::is_trivially_copyable_v<ex::Parameters<SPEC>>);
    static_assert(std::is_standard_layout_v<ex::State<SPEC>>);
    static_assert(std::is_trivially_copyable_v<ex::State<SPEC>>);

    Device device;
    ex::Environment<SPEC> env;
    env.scene_path = "/home/jonas/phd/projects/render/ProcTHOR-Test-0-new.glb";

    ex::malloc(device, env);
    if (!ex::init(device, env)) {
        std::cerr << "failed to initialize raytracing environment" << std::endl;
        return 1;
    }

    std::vector<ex::Parameters<SPEC>> parameters(NUM_ENVS);
    std::vector<ex::State<SPEC>> states(NUM_ENVS);
    std::vector<ex::State<SPEC>> next_states(NUM_ENVS);
    std::vector<T> actions(NUM_ENVS * 3);
    std::vector<uint32_t> pixels(static_cast<size_t>(NUM_ENVS) * SPEC::RAYTRACING_SPEC::CAM_PIXELS);

    for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
        ex::sample_initial_parameters(parameters[env_i]);
        ex::sample_initial_state(parameters[env_i], states[env_i], env_i);
    }

    constexpr TI STEPS = 64;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (TI step_i = 0; step_i < STEPS; step_i++) {
        for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
            const T phase = static_cast<T>(0.01 * env_i + 0.05 * step_i);
            actions[env_i * 3 + 0] = std::cos(phase);
            actions[env_i * 3 + 1] = std::sin(phase);
            actions[env_i * 3 + 2] = static_cast<T>(0.3) * std::sin(static_cast<T>(0.5) * phase);
        }

        ex::step_batch(env, parameters.data(), states.data(), actions.data(), next_states.data(), NUM_ENVS);
        ex::observe_batch(device, env, parameters.data(), next_states.data(), NUM_ENVS, pixels.data(), static_cast<TI>(pixels.size()));

        states.swap(next_states);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    uint64_t checksum = 0;
    for (size_t i = 0; i < std::min<size_t>(pixels.size(), 10000); i++) {
        checksum += pixels[i];
    }

    const double fps = (static_cast<double>(NUM_ENVS) * STEPS) / elapsed;
    std::cout << "raytracing_example: " << NUM_ENVS << " envs, " << STEPS << " batched observe steps" << std::endl;
    std::cout << "elapsed: " << elapsed << " s, effective frame throughput: " << fps << " frames/s" << std::endl;
    std::cout << "checksum(sample): " << checksum << std::endl;

    rl_tools::save_image(device, *env.renderer, "raytracing_example_grid.png");

    ex::free(device, env);
    return 0;
}
