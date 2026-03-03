#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"


#include <iostream>
#include <chrono>
#include <type_traits>
#include <cmath>

namespace rlt = rl_tools;

int main() {
    using T = float;
    using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
    static constexpr TI NUM_ENVS = 1024;
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, 256, 256, 64>;

    static_assert(std::is_standard_layout_v<rlt::rl::environments::raytracing_example::Parameters<SPEC>>);
    static_assert(std::is_trivially_copyable_v<rlt::rl::environments::raytracing_example::Parameters<SPEC>>);
    static_assert(std::is_standard_layout_v<rlt::rl::environments::raytracing_example::State<SPEC>>);
    static_assert(std::is_trivially_copyable_v<rlt::rl::environments::raytracing_example::State<SPEC>>);

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::init(device, rng, 0);

    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = "ProcTHOR-Test-0-new.glb";

    using PARAMETERS_SPEC = rlt::tensor::Specification<rlt::rl::environments::raytracing_example::Parameters<SPEC>, TI, rlt::tensor::Shape<TI, NUM_ENVS>>;
    using STATE_SPEC = rlt::tensor::Specification<rlt::rl::environments::raytracing_example::State<SPEC>, TI, rlt::tensor::Shape<TI, NUM_ENVS>>;
    using ACTIONS_SPEC = rlt::matrix::Specification<T, TI, NUM_ENVS, 3>;
    using PIXELS_SPEC = rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, NUM_ENVS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>>;

    rlt::Tensor<PARAMETERS_SPEC> parameters;
    rlt::Tensor<STATE_SPEC> states;
    rlt::Tensor<STATE_SPEC> next_states;
    rlt::Matrix<ACTIONS_SPEC> actions;
    rlt::Tensor<PIXELS_SPEC> pixels;

    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, next_states);
    rlt::malloc(device, actions);
    rlt::malloc(device, pixels);

    rlt::malloc(device, env);
    rlt::init(device, env);

    for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
        rlt::rl::environments::raytracing_example::Parameters<SPEC> p;
        rlt::sample_initial_parameters(device, env, p, rng);
        rlt::rl::environments::raytracing_example::State<SPEC> s;
        rlt::sample_initial_state(device, env, p, s, rng);
        // const T angle = static_cast<T>(env_i) * static_cast<T>(0.01);
        // s.position[0] = -0.937;
        // s.position[1] = 1.690;
        // s.position[2] = 8.410;
        // s.velocity[0] = 0;
        // s.velocity[1] = 0;
        // s.velocity[2] = 0;
        // s.yaw = angle + static_cast<T>(3.14159265358979323846) / static_cast<T>(2.0);
        rlt::set(device, parameters, p, env_i);
        rlt::set(device, states, s, env_i);
        rlt::set(device, next_states, s, env_i);
    }


    constexpr TI STEPS = 256;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (TI step_i = 0; step_i < STEPS; step_i++) {
        for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
            const T phase = static_cast<T>(0.01 * env_i + 0.05 * step_i);
            rlt::set(actions, env_i, 0, std::cos(phase));
            rlt::set(actions, env_i, 1, std::sin(phase));
            rlt::set(actions, env_i, 2, static_cast<T>(0.3) * std::sin(static_cast<T>(0.5) * phase));
        }

        rlt::step_batch(device, env, parameters, states, actions, next_states, rng, NUM_ENVS);
        rlt::observe_batch(device, env, parameters, next_states, NUM_ENVS, pixels);

        rlt::copy(device, device, states, next_states);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const double fps = (static_cast<double>(NUM_ENVS) * STEPS) / elapsed;
    std::cout << "raytracing_example: " << NUM_ENVS << " envs, " << STEPS << " batched observe steps" << std::endl;
    std::cout << "elapsed: " << elapsed << " s, effective frame throughput: " << fps << " frames/s" << std::endl;

    rlt::save_image(device, *env.renderer, "raytracing_example_grid.png");

    rlt::free(device, env);
    rlt::free(device, pixels);
    rlt::free(device, actions);
    rlt::free(device, next_states);
    rlt::free(device, states);
    rlt::free(device, parameters);
    return 0;
}
