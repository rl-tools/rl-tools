#pragma once

#include "environment.h"

#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <rl_tools/operations/cpu_mux.h>

#include <array>
#include <cmath>
#include <string>
#include <cstdint>

namespace rl_tools::raytracing_example {
    template <typename SPEC>
    struct ObservationRGB {
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH;
    };
}

namespace rl_tools::raytracing_example {
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            env.renderer = new typename Environment<SPEC>::Renderer{};
            env.owns_renderer = true;
            rl_tools::malloc(device, *env.renderer);
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, Environment<SPEC>& env) {
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

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, Environment<SPEC>& env) {
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
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters) {
        parameters.scene_translation[0] = 0;
        parameters.scene_translation[1] = 0;
        parameters.scene_translation[2] = 0;
        parameters.base_height = 0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters, RNG& rng) {
        initial_parameters(device, env, parameters);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters, State<SPEC>& state, RNG& rng) {
        using T = typename SPEC::T;
        const T angle = rl_tools::random::uniform_real_distribution(device.random, static_cast<T>(0), static_cast<T>(2.0 * 3.14159265358979323846), rng);
        const T radius = rl_tools::random::uniform_real_distribution(device.random, static_cast<T>(2.0), static_cast<T>(6.0), rng);

        state.position[0] = radius * std::cos(angle);
        state.position[1] = parameters.base_height;
        state.position[2] = radius * std::sin(angle);
        state.velocity[0] = 0;
        state.velocity[1] = 0;
        state.velocity[2] = 0;
        state.yaw = angle + static_cast<T>(3.14159265358979323846) / static_cast<T>(2.0);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters, const State<SPEC>& state, const rl_tools::Matrix<ACTION_SPEC>& action, State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 3);

        T desired_vx = rl_tools::get(action, 0, 0) * env.max_velocity;
        T desired_vz = rl_tools::get(action, 0, 1) * env.max_velocity;
        T yaw_rate = rl_tools::get(action, 0, 2);

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

        return env.dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters, const State<SPEC>& state, const rl_tools::Matrix<ACTION_SPEC>& action, const State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        const T v = std::sqrt(next_state.velocity[0] * next_state.velocity[0] + next_state.velocity[2] * next_state.velocity[2]);
        const T control = std::abs(rl_tools::get(action, 0, 0)) + std::abs(rl_tools::get(action, 0, 1)) + static_cast<T>(0.1) * std::abs(rl_tools::get(action, 0, 2));
        return v - static_cast<T>(0.05) * control;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool terminated(DEVICE& device, const Environment<SPEC>& env, Parameters<SPEC>& parameters, const State<SPEC>& state, RNG& rng) {
        return false;
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, Environment<SPEC>& env, const Parameters<SPEC>& parameters, const State<SPEC>& state, const ObservationRGB<SPEC>&, rl_tools::Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == ObservationRGB<SPEC>::DIM);

        std::array<CameraData, 1> cameras{};
        const auto camera = rl_tools::make_camera_data(
            vec3f(
                parameters.scene_translation[0] + state.position[0],
                parameters.scene_translation[1] + state.position[1] + env.eye_height,
                parameters.scene_translation[2] + state.position[2]
            ),
            vec3f(
                parameters.scene_translation[0] + state.position[0] + env.look_ahead * std::cos(state.yaw),
                parameters.scene_translation[1] + state.position[1] + env.eye_height,
                parameters.scene_translation[2] + state.position[2] + env.look_ahead * std::sin(state.yaw)
            ),
            vec3f(0.f, 1.f, 0.f),
            SPEC::RAYTRACING_SPEC::COS_FOVY,
            static_cast<typename SPEC::T>(SPEC::CAM_WIDTH) / static_cast<typename SPEC::T>(SPEC::CAM_HEIGHT)
        );
        cameras[0] = camera;

        rl_tools::set_cameras(device, *env.renderer, cameras.data(), 1);
        rl_tools::render(device, *env.renderer);

        std::array<uint32_t, SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT> pixels{};
        rl_tools::read_frame_buffer(device, *env.renderer, pixels.data(), pixels.size());
        for (typename SPEC::TI i = 0; i < static_cast<typename SPEC::TI>(pixels.size()); i++) {
            rl_tools::set(observation, 0, i, static_cast<typename OBS_SPEC::T>(pixels[i]));
        }
    }

    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT CameraData make_camera_for_state(const Environment<SPEC>& env, const Parameters<SPEC>& parameters, const State<SPEC>& state) {
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

    template <typename DEVICE, typename SPEC, typename PARAMETERS_SPEC, typename STATE_SPEC, typename ACTIONS_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void step_batch(DEVICE& device, const Environment<SPEC>& env, rl_tools::Tensor<PARAMETERS_SPEC>& parameters, const rl_tools::Tensor<STATE_SPEC>& states, const rl_tools::Matrix<ACTIONS_SPEC>& actions, rl_tools::Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng, typename SPEC::TI num_envs) {
        static_assert(rl_tools::utils::typing::is_same_v<typename PARAMETERS_SPEC::T, Parameters<SPEC>>);
        static_assert(rl_tools::utils::typing::is_same_v<typename STATE_SPEC::T, State<SPEC>>);
        static_assert(rl_tools::utils::typing::is_same_v<typename NEXT_STATE_SPEC::T, State<SPEC>>);
        static_assert(ACTIONS_SPEC::ROWS == SPEC::NUM_ENVS);
        static_assert(ACTIONS_SPEC::COLS == 3);

        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            auto& parameters_env = rl_tools::get_ref(device, parameters, env_i);
            auto& state_env = rl_tools::get_ref(device, states, env_i);
            auto action_env = rl_tools::row(device, actions, env_i);
            auto& next_state_env = rl_tools::get_ref(device, next_states, env_i);
            step(device, env, parameters_env, state_env, action_env, next_state_env, rng);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETERS_SPEC, typename STATE_SPEC, typename OUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_batch(DEVICE& device, Environment<SPEC>& env, const rl_tools::Tensor<PARAMETERS_SPEC>& parameters, const rl_tools::Tensor<STATE_SPEC>& states, typename SPEC::TI num_envs, rl_tools::Tensor<OUT_SPEC>& out_pixels) {
        static_assert(rl_tools::utils::typing::is_same_v<typename PARAMETERS_SPEC::T, Parameters<SPEC>>);
        static_assert(rl_tools::utils::typing::is_same_v<typename STATE_SPEC::T, State<SPEC>>);
        static_assert(rl_tools::utils::typing::is_same_v<typename OUT_SPEC::T, uint32_t>);
        static_assert(rl_tools::get<0>(typename OUT_SPEC::SHAPE{}) == SPEC::NUM_ENVS);
        static_assert(rl_tools::get<1>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(rl_tools::get<2>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);

        if (env.renderer == nullptr || num_envs != SPEC::NUM_ENVS) {
            return;
        }

        std::array<CameraData, SPEC::NUM_ENVS> cameras;
        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            cameras[env_i] = make_camera_for_state(env, rl_tools::get_ref(device, parameters, env_i), rl_tools::get_ref(device, states, env_i));
        }

        rl_tools::set_cameras(device, *env.renderer, cameras.data(), num_envs);
        rl_tools::render(device, *env.renderer);
        rl_tools::read_frame_buffer(device, *env.renderer, rl_tools::data(out_pixels), rl_tools::product(typename OUT_SPEC::SHAPE{}));
    }
}
