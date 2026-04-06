#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include "environment.h"

#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <rl_tools/rendering/raytracing/scene/procthor/scene.h>
#include <rl_tools/rendering/raytracing/scene/procthor/operations_cpu.h>

#include <cmath>
#include <string>

namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void precompute_indoor_initial_states(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using SCENE_SPEC = rendering::raytracing::scene::SceneSpecification<T, TI>;
        rendering::raytracing::scene::procthor::Scene<SCENE_SPEC> scene;
        T fov = SPEC::RAYTRACING_SPEC::COS_FOVY;
        T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
        rendering::raytracing::scene::procthor::precompute_indoor_positions(device, scene, *env.renderer, fov, aspect);
        TI take_n = std::min(scene.num_indoor_positions, rl::environments::raytracing_example::Environment<SPEC>::NUM_INITIAL_STATES);
        for(TI i = 0; i < take_n; i++){
            auto& s = env.indoor_initial_states[i];
            s.position[0] = scene.indoor_positions[i].position[0];
            s.position[1] = scene.indoor_positions[i].position[1];
            s.position[2] = scene.indoor_positions[i].position[2];
            s.velocity[0] = static_cast<T>(0);
            s.velocity[1] = static_cast<T>(0);
            s.velocity[2] = static_cast<T>(0);
            s.yaw = scene.indoor_positions[i].yaw;
        }
        env.num_indoor_initial_states = take_n;
    }

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
        utils::assert_exit(device, env.scene_path != nullptr, "raytracing_example::init: scene_path is null");
        const bool loaded = load_model(device, *env.renderer, std::string(env.scene_path));
        utils::assert_exit(device, loaded, "raytracing_example::init: failed to load scene");

        upload_geometry(device, *env.renderer);
        {
            using T = typename SPEC::T;
            const T up[3] = {0, 0, 1};
            generate_cameras(device, *env.renderer, env.renderer->scene_center, env.renderer->camera_radius, up, SPEC::RAYTRACING_SPEC::COS_FOVY);
        }
        generate_probe_directions(device, *env.renderer);
        build_pipeline(device, *env.renderer);
        precompute_indoor_initial_states(device, env);
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
        using TI = typename SPEC::TI;

        utils::assert_exit(device, env.num_indoor_initial_states > 0, "raytracing_example::sample_initial_state: no indoor initial states available");
        const TI index = random::uniform_int_distribution(device.random, static_cast<TI>(0), static_cast<TI>(env.num_indoor_initial_states - 1), rng);
        state = env.indoor_initial_states[index];
        state.position[2] = parameters.base_height;
        state.velocity[0] = static_cast<T>(0);
        state.velocity[1] = static_cast<T>(0);
        state.velocity[2] = static_cast<T>(0);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::raytracing_example::State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 3);

        T desired_vx = get(action, 0, 0) * env.max_velocity;
        T desired_vy = get(action, 0, 1) * env.max_velocity;
        T yaw_rate = get(action, 0, 2);

        T alpha = env.acceleration * env.dt;
        if (alpha > static_cast<T>(1)) {
            alpha = static_cast<T>(1);
        }

        next_state.velocity[0] = state.velocity[0] + alpha * (desired_vx - state.velocity[0]);
        next_state.velocity[1] = state.velocity[1] + alpha * (desired_vy - state.velocity[1]);
        next_state.velocity[2] = 0;

        next_state.position[0] = state.position[0] + next_state.velocity[0] * env.dt;
        next_state.position[1] = state.position[1] + next_state.velocity[1] * env.dt;
        next_state.position[2] = parameters.base_height;
        next_state.yaw = state.yaw + yaw_rate * env.dt;

        return env.dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::raytracing_example::Environment<SPEC>& env, rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, const rl::environments::raytracing_example::State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        const T v = std::sqrt(next_state.velocity[0] * next_state.velocity[0] + next_state.velocity[1] * next_state.velocity[1]);
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

        using T = typename SPEC::T;
        const T position[3] = {
            parameters.scene_translation[0] + state.position[0],
            parameters.scene_translation[1] + state.position[1],
            parameters.scene_translation[2] + state.position[2] + env.eye_height
        };
        const T look_at[3] = {
            position[0] + env.look_ahead * std::cos(state.yaw),
            position[1] + env.look_ahead * std::sin(state.yaw),
            position[2]
        };
        const T up[3] = {0, 0, 1};
        set(device, env.renderer->cameras, make_camera_data(position, look_at, up,
            SPEC::RAYTRACING_SPEC::COS_FOVY,
            static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT)), static_cast<typename SPEC::TI>(0));

        set_cameras(device, *env.renderer, env.renderer->cameras);
        render(device, *env.renderer);

        read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);
        const uint32_t* fb_data = data(env.renderer->frame_buffer);
        for (typename SPEC::TI i = 0; i < static_cast<typename SPEC::TI>(SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT); i++) {
            set(observation, 0, i, static_cast<typename OBS_SPEC::T>(fb_data[i]));
        }
    }

    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::CameraData<typename SPEC::T> make_camera_for_state(const rl::environments::raytracing_example::Environment<SPEC>& env, const rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state) {
        using T = typename SPEC::T;
        const T cy = std::cos(state.yaw);
        const T sy = std::sin(state.yaw);

        const T position[3] = {
            parameters.scene_translation[0] + state.position[0],
            parameters.scene_translation[1] + state.position[1],
            parameters.scene_translation[2] + state.position[2] + env.eye_height
        };

        const T look_at[3] = {
            position[0] + env.look_ahead * cy,
            position[1] + env.look_ahead * sy,
            position[2]
        };

        const T up[3] = {0, 0, 1};
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

        for (typename SPEC::TI env_i = 0; env_i < num_envs; env_i++) {
            set(device, env.renderer->cameras, make_camera_for_state(env, get_ref(device, parameters, env_i), get_ref(device, states, env_i)), env_i);
        }

        set_cameras(device, *env.renderer, env.renderer->cameras);
        render(device, *env.renderer);
        read_frame_buffer(device, *env.renderer, out_pixels);
    }
}
