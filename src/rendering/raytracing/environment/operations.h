#pragma once

#include "environment.h"

#include <rl_tools/rendering/raytracing/operations_optix.h>

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

namespace raytracing_example {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            env.renderer = new typename Environment<SPEC>::Renderer{};
            env.owns_renderer = true;
            rl_tools::malloc(device, *env.renderer);
        }
    }

    template <typename DEVICE, typename SPEC>
    bool init(DEVICE& device, Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            return false;
        }

        bool loaded = false;
        if (env.scene_path != nullptr) {
            loaded = rl_tools::load_model(device, *env.renderer, std::string(env.scene_path));
        }
        if (!loaded) {
            rl_tools::load_default_cube(device, *env.renderer);
        }

        rl_tools::upload_geometry(device, *env.renderer);
        {
            vec3f center(env.renderer->scene_center[0], env.renderer->scene_center[1], env.renderer->scene_center[2]);
            const vec3f up(0.f, 1.f, 0.f);
            rl_tools::generate_cameras(device, *env.renderer, center, env.renderer->camera_radius, up, SPEC::RAYTRACING_SPEC::COS_FOVY);
        }
        rl_tools::generate_probe_directions(device, *env.renderer);
        rl_tools::build_pipeline(device, *env.renderer);
        return true;
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            return;
        }
        rl_tools::free(device, *env.renderer);
        if (env.owns_renderer) {
            delete env.renderer;
        }
        env.renderer = nullptr;
        env.owns_renderer = false;
    }

    template <typename SPEC>
    void sample_initial_parameters(Parameters<SPEC>& parameters) {
        parameters.scene_translation[0] = 0;
        parameters.scene_translation[1] = 0;
        parameters.scene_translation[2] = 0;
        parameters.base_height = 0;
    }

    template <typename SPEC>
    void sample_initial_state(const Parameters<SPEC>& parameters, State<SPEC>& state, typename SPEC::TI env_i) {
        using T = typename SPEC::T;
        const T angle = static_cast<T>(env_i) * static_cast<T>(0.01);
        state.position[0] = static_cast<T>(4.0) * std::cos(angle);
        state.position[1] = parameters.base_height;
        state.position[2] = static_cast<T>(4.0) * std::sin(angle);
        state.velocity[0] = 0;
        state.velocity[1] = 0;
        state.velocity[2] = 0;
        state.yaw = angle + static_cast<T>(3.14159265358979323846) / static_cast<T>(2.0);
    }

    template <typename SPEC>
    void step(const Environment<SPEC>& env, const Parameters<SPEC>& parameters, const State<SPEC>& state, const typename SPEC::T* action, State<SPEC>& next_state) {
        using T = typename SPEC::T;

        T desired_vx = action[0] * env.max_velocity;
        T desired_vz = action[1] * env.max_velocity;
        T yaw_rate = action[2];

        T alpha = env.acceleration * env.dt;
        if (alpha > static_cast<T>(1)) {
            alpha = static_cast<T>(1);
        }

        next_state.velocity[0] = state.velocity[0] + alpha * (desired_vx - state.velocity[0]);
        next_state.velocity[1] = 0;
        next_state.velocity[2] = state.velocity[2] + alpha * (desired_vz - state.velocity[2]);

        next_state.position[0] = state.position[0] + next_state.velocity[0] * env.dt;
        next_state.position[1] = parameters.base_height;
        next_state.position[2] = state.position[2] + next_state.velocity[2] * env.dt;
        next_state.yaw = state.yaw + yaw_rate * env.dt;
    }

    template <typename SPEC>
    void step_batch(const Environment<SPEC>& env,
                    const Parameters<SPEC>* parameters,
                    const State<SPEC>* states,
                    const typename SPEC::T* actions,
                    State<SPEC>* next_states,
                    typename SPEC::TI num_envs) {
        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            step(env, parameters[env_i], states[env_i], actions + env_i * 3, next_states[env_i]);
        }
    }

    template <typename SPEC>
    CameraData make_camera_for_state(const Environment<SPEC>& env, const Parameters<SPEC>& parameters, const State<SPEC>& state) {
        using T = typename SPEC::T;
        const T cy = std::cos(state.yaw);
        const T sy = std::sin(state.yaw);

        const vec3f position(
            parameters.scene_translation[0] + state.position[0],
            parameters.scene_translation[1] + state.position[1] + env.eye_height,
            parameters.scene_translation[2] + state.position[2]
        );

        const vec3f look_at(
            position.x + env.look_ahead * cy,
            position.y,
            position.z + env.look_ahead * sy
        );

        const vec3f up(0.f, 1.f, 0.f);
        const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);

        return rl_tools::make_camera_data(position, look_at, up, SPEC::RAYTRACING_SPEC::COS_FOVY, aspect);
    }

    template <typename DEVICE, typename SPEC>
    void observe_batch(DEVICE& device,
                       Environment<SPEC>& env,
                       const Parameters<SPEC>* parameters,
                       const State<SPEC>* states,
                       typename SPEC::TI num_envs,
                       uint32_t* out_pixels,
                       typename SPEC::TI out_pixels_count) {
        if (env.renderer == nullptr || num_envs != SPEC::NUM_ENVS) {
            return;
        }

        std::vector<CameraData> cameras;
        cameras.reserve(num_envs);
        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            cameras.push_back(make_camera_for_state(env, parameters[env_i], states[env_i]));
        }

        rl_tools::set_cameras(device, *env.renderer, cameras.data(), num_envs);
        rl_tools::render(device, *env.renderer);
        rl_tools::read_frame_buffer(device, *env.renderer, out_pixels, out_pixels_count);
    }
}
