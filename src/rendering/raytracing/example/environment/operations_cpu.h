#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include "environment.h"

#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include <array>
#include <cmath>
#include <string>
#include <cstdint>

namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            env.renderer = new typename rl::environments::raytracing_example::Environment<SPEC>::Renderer{};
            env.owns_renderer = true;
            malloc(device, *env.renderer);
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            return;
        }
        free(device, *env.renderer);
        if (env.owns_renderer) {
            delete env.renderer;
        }
        env.renderer = nullptr;
        env.owns_renderer = false;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        bool loaded = false;
        if (env.scene_path != nullptr) {
            loaded = load_model(device, *env.renderer, std::string(env.scene_path));
        }
        if (!loaded) {
            load_default_cube(device, *env.renderer);
        }

        upload_geometry(device, *env.renderer);
        {
            owl::vec3f center(env.renderer->scene_center[0], env.renderer->scene_center[1], env.renderer->scene_center[2]);
            const owl::vec3f up(0.f, 1.f, 0.f);
            generate_cameras(device, *env.renderer, center, env.renderer->camera_radius, up, SPEC::RAYTRACING_SPEC::COS_FOVY);
        }
        generate_probe_directions(device, *env.renderer);
        build_pipeline(device, *env.renderer);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters) {
        parameters.scene_translation[0] = 0;
        parameters.scene_translation[1] = 0;
        parameters.scene_translation[2] = 0;
        parameters.base_height = 0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, RNG& rng) {
        initial_parameters(device, env, parameters);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, rl::environments::raytracing_example::State<SPEC>& state, RNG& rng) {
        using T = typename SPEC::T;
        const T angle = random::uniform_real_distribution(device.random, static_cast<T>(0), static_cast<T>(2.0 * 3.14159265358979323846), rng);
        const T radius = random::uniform_real_distribution(device.random, static_cast<T>(2.0), static_cast<T>(6.0), rng);

        state.position[0] = radius * std::cos(angle);
        state.position[1] = parameters.base_height;
        state.position[2] = radius * std::sin(angle);
        state.velocity[0] = 0;
        state.velocity[1] = 0;
        state.velocity[2] = 0;
        state.yaw = angle + static_cast<T>(3.14159265358979323846) / static_cast<T>(2.0);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::raytracing_example::State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 3);

        T desired_vx = get(action, 0, 0) * env.max_velocity;
        T desired_vz = get(action, 0, 1) * env.max_velocity;
        T yaw_rate = get(action, 0, 2);

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
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, const rl::environments::raytracing_example::State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        const T v = std::sqrt(next_state.velocity[0] * next_state.velocity[0] + next_state.velocity[2] * next_state.velocity[2]);
        const T control = std::abs(get(action, 0, 0)) + std::abs(get(action, 0, 1)) + static_cast<T>(0.1) * std::abs(get(action, 0, 2));
        return v - static_cast<T>(0.05) * control;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool terminated(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, RNG& rng) {
        return false;
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env, const rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, const rl::environments::raytracing_example::ObservationRGB<SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == rl::environments::raytracing_example::ObservationRGB<SPEC>::DIM);

        std::array<CameraData, 1> cameras{};
        const auto camera = make_camera_data(
            owl::vec3f(
                parameters.scene_translation[0] + state.position[0],
                parameters.scene_translation[1] + state.position[1] + env.eye_height,
                parameters.scene_translation[2] + state.position[2]
            ),
            owl::vec3f(
                parameters.scene_translation[0] + state.position[0] + env.look_ahead * std::cos(state.yaw),
                parameters.scene_translation[1] + state.position[1] + env.eye_height,
                parameters.scene_translation[2] + state.position[2] + env.look_ahead * std::sin(state.yaw)
            ),
            owl::vec3f(0.f, 1.f, 0.f),
            SPEC::RAYTRACING_SPEC::COS_FOVY,
            static_cast<typename SPEC::T>(SPEC::CAM_WIDTH) / static_cast<typename SPEC::T>(SPEC::CAM_HEIGHT)
        );
        cameras[0] = camera;

        set_cameras(device, *env.renderer, cameras.data(), 1);
        render(device, *env.renderer);

        std::array<uint32_t, SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT> pixels{};
        read_frame_buffer(device, *env.renderer, pixels.data(), pixels.size());
        for (typename SPEC::TI i = 0; i < static_cast<typename SPEC::TI>(pixels.size()); i++) {
            set(observation, 0, i, static_cast<typename OBS_SPEC::T>(pixels[i]));
        }
    }

    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT CameraData make_camera_for_state(const rl::environments::raytracing_example::Environment<SPEC>& env, const rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state) {
        using T = typename SPEC::T;
        const T cy = std::cos(state.yaw);
        const T sy = std::sin(state.yaw);

        const owl::vec3f position(
            parameters.scene_translation[0] + state.position[0],
            parameters.scene_translation[1] + state.position[1] + env.eye_height,
            parameters.scene_translation[2] + state.position[2]
        );

        const owl::vec3f look_at(
            position.x + env.look_ahead * cy,
            position.y,
            position.z + env.look_ahead * sy
        );

        const owl::vec3f up(0.f, 1.f, 0.f);
        const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);

        return make_camera_data(position, look_at, up, SPEC::RAYTRACING_SPEC::COS_FOVY, aspect);
    }

    template <typename DEVICE, typename SPEC, typename PARAMETERS_SPEC, typename STATE_SPEC, typename ACTIONS_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void step_batch(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, Tensor<PARAMETERS_SPEC>& parameters, const Tensor<STATE_SPEC>& states, const Matrix<ACTIONS_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng, typename SPEC::TI num_envs) {
        static_assert(utils::typing::is_same_v<typename PARAMETERS_SPEC::T, rl::environments::raytracing_example::Parameters<SPEC>>);
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, rl::environments::raytracing_example::State<SPEC>>);
        static_assert(utils::typing::is_same_v<typename NEXT_STATE_SPEC::T, rl::environments::raytracing_example::State<SPEC>>);
        static_assert(ACTIONS_SPEC::ROWS == SPEC::NUM_ENVS);
        static_assert(ACTIONS_SPEC::COLS == 3);

        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            auto& parameters_env = get_ref(device, parameters, env_i);
            auto& state_env = get_ref(device, states, env_i);
            auto action_env = row(device, actions, env_i);
            auto& next_state_env = get_ref(device, next_states, env_i);
            step(device, env, parameters_env, state_env, action_env, next_state_env, rng);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETERS_SPEC, typename STATE_SPEC, typename OUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_batch(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env, const Tensor<PARAMETERS_SPEC>& parameters, const Tensor<STATE_SPEC>& states, typename SPEC::TI num_envs, Tensor<OUT_SPEC>& out_pixels) {
        static_assert(utils::typing::is_same_v<typename PARAMETERS_SPEC::T, rl::environments::raytracing_example::Parameters<SPEC>>);
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, rl::environments::raytracing_example::State<SPEC>>);
        static_assert(utils::typing::is_same_v<typename OUT_SPEC::T, uint32_t>);
        static_assert(get<0>(typename OUT_SPEC::SHAPE{}) == SPEC::NUM_ENVS);
        static_assert(get<1>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);

        if (env.renderer == nullptr || num_envs != SPEC::NUM_ENVS) {
            return;
        }

        std::array<CameraData, SPEC::NUM_ENVS> cameras;
        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            cameras[env_i] = make_camera_for_state(env, get_ref(device, parameters, env_i), get_ref(device, states, env_i));
        }

        set_cameras(device, *env.renderer, cameras.data(), num_envs);
        render(device, *env.renderer);
        read_frame_buffer(device, *env.renderer, data(out_pixels), product(typename OUT_SPEC::SHAPE{}));
    }
}
