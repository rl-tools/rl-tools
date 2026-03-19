#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H

#include "multirotor_visual.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <rl_tools/rendering/raytracing/scene/procthor/operations_cpu.h>

#include <array>
#include <cmath>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    // =========================================================================
    // Lifecycle
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        if (env.renderer == nullptr) {
            env.renderer = new rendering::raytracing::Renderer<typename SPEC::RENDERER_SPEC>{};
            env.owns_renderer = true;
            malloc(device, *env.renderer);
        }
        if (env.scene == nullptr) {
            env.scene = new rendering::raytracing::scene::procthor::Scene<typename SPEC::SCENE_SPEC>{};
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        if (env.renderer != nullptr) {
            free(device, *env.renderer);
            if (env.owns_renderer) {
                delete env.renderer;
            }
            env.renderer = nullptr;
            env.owns_renderer = false;
        }
        if (env.scene != nullptr) {
            delete env.scene;
            env.scene = nullptr;
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        using T = typename SPEC::T;
        init(device, env.dynamics);

        if (env.renderer_initialized) {
            return;
        }

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
            generate_cameras(device, *env.renderer, center, env.renderer->camera_radius, up, env.cos_fov);
        }
        generate_probe_directions(device, *env.renderer);
        build_pipeline(device, *env.renderer);

        T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
        rendering::raytracing::scene::procthor::precompute_indoor_positions(device, *env.scene, *env.renderer, env.eye_height, env.cos_fov, aspect);

        env.renderer_initialized = true;
    }

    // =========================================================================
    // Parameters
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters) {
        initial_parameters(device, env.dynamics, parameters.dynamics);
        parameters.scene_translation[0] = 0;
        parameters.scene_translation[1] = 0;
        parameters.scene_translation[2] = 0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, RNG& rng) {
        using TI = typename SPEC::TI;
        sample_initial_parameters(device, env.dynamics, parameters.dynamics, rng);
        if (env.use_target_mode) {
            for (TI i = 0; i < 3; i++) {
                parameters.scene_translation[i] = env.target_scene_translation[i];
            }
        } else {
            parameters.scene_translation[0] = 0;
            parameters.scene_translation[1] = 0;
            parameters.scene_translation[2] = 0;
        }
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        if (env.use_target_mode) {
            sample_initial_state(device, env.dynamics, parameters.dynamics, state, rng);
            return;
        }

        // Sample indoor position from the scene
        auto indoor_pos = rendering::raytracing::scene::procthor::sample_indoor_position(device, *env.scene, rng);

        // Use the L2F sample_initial_state for dynamics state initialization
        sample_initial_state(device, env.dynamics, parameters.dynamics, state, rng);

        // Override position with indoor position (converting from scene Y-up to NED)
        // In the scene: X=forward, Y=up, Z=right
        // In NED: X=north, Y=east, Z=down
        state.position[0] = indoor_pos.position[0];
        state.position[1] = indoor_pos.position[2];
        state.position[2] = -indoor_pos.position[1] - env.eye_height;

        // Set hover orientation (identity quaternion = level)
        state.orientation[0] = static_cast<T>(1);
        state.orientation[1] = static_cast<T>(0);
        state.orientation[2] = static_cast<T>(0);
        state.orientation[3] = static_cast<T>(0);

        // Apply yaw rotation around NED down axis
        T half_yaw = indoor_pos.yaw / static_cast<T>(2);
        state.orientation[0] = std::cos(half_yaw);
        state.orientation[1] = static_cast<T>(0);
        state.orientation[2] = static_cast<T>(0);
        state.orientation[3] = std::sin(half_yaw);

        // Zero velocities for hover start
        for (TI i = 0; i < 3; i++) {
            state.linear_velocity[i] = static_cast<T>(0);
            state.angular_velocity[i] = static_cast<T>(0);
        }
    }

    // =========================================================================
    // Dynamics (delegate to L2F)
    // =========================================================================
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& next_state, RNG& rng) {
        return step(device, env.dynamics, parameters.dynamics, state, action, next_state, rng);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& next_state, RNG& rng) {
        return reward(device, env.dynamics, parameters.dynamics, state, action, next_state, rng);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, RNG& rng) {
        return terminated(device, env.dynamics, parameters.dynamics, state, rng);
    }

    // =========================================================================
    // Camera helper: construct camera from quadrotor state
    // =========================================================================
    namespace rl::environments::l2f_visual {
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT CameraData make_camera_for_state(DEVICE&, const MultirrotorVisual<SPEC>& env, const typename MultirrotorVisual<SPEC>::Parameters& parameters, const typename MultirrotorVisual<SPEC>::State& state) {
            using T = typename SPEC::T;

            T cam_pos_world[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, env.camera_mount.offset_body, cam_pos_world);

            T cam_forward_world[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, env.camera_mount.forward_body, cam_forward_world);

            T cam_up_world[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, env.camera_mount.up_body, cam_up_world);

            // Convert from NED to scene coordinates (Y-up)
            // NED: X=north, Y=east, Z=down
            // Scene: X=north, Y=up, Z=east
            const T px = state.position[0] + cam_pos_world[0] + parameters.scene_translation[0];
            const T py = -state.position[2] + cam_pos_world[2] + parameters.scene_translation[1];
            const T pz = state.position[1] + cam_pos_world[1] + parameters.scene_translation[2];

            const owl::vec3f position(px, py, pz);
            const owl::vec3f look_at(
                px + cam_forward_world[0],
                py + cam_forward_world[2],
                pz + cam_forward_world[1]
            );
            const owl::vec3f up(
                -cam_up_world[0],
                cam_up_world[2],
                -cam_up_world[1]
            );

            const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
            return make_camera_data(position, look_at, up, env.cos_fov, aspect);
        }
    }

    // =========================================================================
    // Observation: Image (RGB)
    // =========================================================================
    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const rl::environments::observation::Image<typename SPEC::TI, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        using T = typename OBS_SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH * 3);

        CameraData camera = rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, state);
        std::array<CameraData, SPEC::NUM_ENVS> cameras;
        for (TI i = 0; i < SPEC::NUM_ENVS; i++) {
            cameras[i] = camera;
        }

        set_cameras(device, *env.renderer, cameras.data(), SPEC::NUM_ENVS);
        render(device, *env.renderer);

        constexpr TI TOTAL_PIXELS = SPEC::NUM_ENVS * SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        std::array<uint32_t, TOTAL_PIXELS> all_pixels{};
        read_frame_buffer(device, *env.renderer, all_pixels.data(), all_pixels.size());

        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        for (TI i = 0; i < CAM_PIXELS; i++) {
            const uint32_t rgba = all_pixels[i];
            const T r = static_cast<T>((rgba >>  0) & 0xFF) / static_cast<T>(255);
            const T g = static_cast<T>((rgba >>  8) & 0xFF) / static_cast<T>(255);
            const T b = static_cast<T>((rgba >> 16) & 0xFF) / static_cast<T>(255);
            set(observation, 0, i * 3 + 0, r);
            set(observation, 0, i * 3 + 1, g);
            set(observation, 0, i * 3 + 2, b);
        }
    }

    // =========================================================================
    // Observation: Privileged (delegate to L2F dense observation)
    // =========================================================================
    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::ObservationPrivileged& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng) {
        observe(device, env.dynamics, parameters.dynamics, state, observation_type, observation, rng);
    }

    // =========================================================================
    // Batch observation
    // =========================================================================
    template <typename DEVICE, typename SPEC, typename PARAMETERS_SPEC, typename STATE_SPEC, typename OUT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_batch(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const Tensor<PARAMETERS_SPEC>& parameters, const Tensor<STATE_SPEC>& states, typename SPEC::TI num_envs, Tensor<OUT_SPEC>& out_pixels) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename PARAMETERS_SPEC::T, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters>);
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State>);
        static_assert(utils::typing::is_same_v<typename OUT_SPEC::T, uint32_t>);
        static_assert(get<0>(typename OUT_SPEC::SHAPE{}) == SPEC::NUM_ENVS);
        static_assert(get<1>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename OUT_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);

        if (env.renderer == nullptr || num_envs != SPEC::NUM_ENVS) {
            return;
        }

        std::array<CameraData, SPEC::NUM_ENVS> cameras;
        for (TI env_i = 0; env_i < num_envs; env_i++) {
            cameras[env_i] = rl::environments::l2f_visual::make_camera_for_state(device, env, get_ref(device, parameters, env_i), get_ref(device, states, env_i));
        }

        set_cameras(device, *env.renderer, cameras.data(), num_envs);
        render(device, *env.renderer);
        read_frame_buffer(device, *env.renderer, data(out_pixels), product(typename OUT_SPEC::SHAPE{}));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
