#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include "environment.h"

#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/procthor.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <cmath>
#include <string>

namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void precompute_indoor_initial_states(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ANNOTATIONS_SPEC = rendering::datasets::procthor::AnnotationsSpecification<T, TI>;
        rendering::datasets::procthor::Annotations<ANNOTATIONS_SPEC> annotations;
        T fov = SPEC::FOV;
        T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
        rendering::datasets::procthor::annotate(device, annotations, env.bundle->metadata, *env.renderer, fov, aspect);
        TI take_n = std::min(annotations.num_indoor_positions, rl::environments::raytracing_example::Environment<SPEC>::NUM_INITIAL_STATES);
        for(TI i = 0; i < take_n; i++){
            auto& s = env.indoor_initial_states[i];
            s.position[0] = annotations.indoor_positions[i].position[0];
            s.position[1] = annotations.indoor_positions[i].position[1];
            s.position[2] = annotations.indoor_positions[i].position[2];
            s.velocity[0] = static_cast<T>(0);
            s.velocity[1] = static_cast<T>(0);
            s.velocity[2] = static_cast<T>(0);
            s.yaw = annotations.indoor_positions[i].yaw;
        }
        env.num_indoor_initial_states = take_n;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        if (env.renderer == nullptr) {
            env.renderer = new typename rl::environments::raytracing_example::Environment<SPEC>::Renderer{};
            env.owns_renderer = true;
            malloc(device, *env.renderer);
            malloc(device, env.camera_staging);
        }
        if (env.bundle == nullptr) {
            env.bundle = new rendering::Bundle<typename SPEC::T>{};
            env.owns_bundle = true;
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        if (env.renderer != nullptr) {
            free(device, *env.renderer);
            if (env.owns_renderer) {
                delete env.renderer;
                free(device, env.camera_staging);
            }
            env.renderer = nullptr;
            env.owns_renderer = false;
        }
        if (env.bundle != nullptr) {
            if (env.owns_bundle) {
                delete env.bundle;
            }
            env.bundle = nullptr;
            env.owns_bundle = false;
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::environments::raytracing_example::Environment<SPEC>& env) {
        utils::assert_exit(device, env.scene_path != nullptr, "raytracing_example::init: scene_path is null");
        bool loaded;
        if constexpr (SPEC::RAYTRACING_SPEC::HAS_SEGMENTATION) {
            rendering::raytracing::ObjectAssembly assembly;
            loaded = load<typename SPEC::RAYTRACING_SPEC::SHADING, SPEC::RAYTRACING_SPEC::HAS_RGB>(device, assembly, std::string(env.scene_path));
            if(loaded){
                add(device, env.bundle->scene, assembly);
                rendering::datasets::compute_bounds(device, *env.bundle);
            }
        }
        else {
            loaded = load<typename SPEC::RAYTRACING_SPEC::SHADING, SPEC::RAYTRACING_SPEC::HAS_RGB>(device, *env.bundle, std::string(env.scene_path));
        }
        utils::assert_exit(device, loaded, "raytracing_example::init: failed to load scene");

        init(device, *env.renderer, *env.bundle);
        {
            using T = typename SPEC::T;
            const T up[3] = {0, 0, 1};
            generate_cameras(device, *env.renderer, env.bundle->metadata.center, (env.bundle->metadata.max_ray_length / 2), up, SPEC::FOV);
        }
        generate_probe_directions(device, *env.renderer);
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
        const auto camera = make_camera_data(position, look_at, up,
            SPEC::FOV,
            static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT));
        set(device, env.camera_staging, camera, 0);
        auto camera_staging_slot = view_range(device, env.camera_staging, 0, tensor::ViewSpec<0, 1>{});
        auto camera_slot = view_range(device, cameras(device, *env.renderer), 0, tensor::ViewSpec<0, 1>{});
        copy(device, env.renderer->device, camera_staging_slot, camera_slot);
        render(device, *env.renderer);

        std::vector<uint32_t> frame_staging((size_t)SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT);
        auto frame_camera_0 = view(device, frame_buffer(device, *env.renderer), 0);
        Tensor<tensor::Specification<uint32_t, typename SPEC::TI, typename decltype(frame_camera_0)::SPEC::SHAPE>> frame_alias;
        frame_alias._data = frame_staging.data();
        copy(env.renderer->device, device, frame_camera_0, frame_alias);
        for (typename SPEC::TI i = 0; i < static_cast<typename SPEC::TI>(SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT); i++) {
            set(observation, 0, i, static_cast<typename OBS_SPEC::T>(frame_staging[i]));
        }
    }

    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_camera_for_state(const rl::environments::raytracing_example::Environment<SPEC>& env, const rl::environments::raytracing_example::Parameters<SPEC>& parameters, const rl::environments::raytracing_example::State<SPEC>& state) {
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

        return make_camera_data(position, look_at, up, SPEC::FOV, aspect);
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
            set(device, env.camera_staging, make_camera_for_state(env, get_ref(device, parameters, env_i), get_ref(device, states, env_i)), env_i);
        }
        copy(device, env.renderer->device, env.camera_staging, cameras(device, *env.renderer));
        render(device, *env.renderer);
        copy(env.renderer->device, device, frame_buffer(device, *env.renderer), out_pixels);
    }
}
