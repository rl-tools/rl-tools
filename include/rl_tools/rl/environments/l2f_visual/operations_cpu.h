#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H

#include "multirotor_visual.h"
#include "operations_generic.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <rl_tools/rendering/raytracing/scene/procthor/operations_cpu.h>

#include <array>
#include <cmath>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

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

        if(env.scene_path == nullptr){
            return;
        }
        const bool loaded = load_model(device, *env.renderer, std::string(env.scene_path));
        utils::assert_exit(device, loaded, "l2f_visual::init: failed to load scene");

        upload_geometry(device, *env.renderer);
        {
            typename rl::environments::l2f_visual::Parameters<SPEC> default_params;
            const T up[3] = {0, 0, 1};
            generate_cameras(device, *env.renderer, env.renderer->scene_center, env.renderer->camera_radius, up, default_params.fov);
            generate_probe_directions(device, *env.renderer);
            build_pipeline(device, *env.renderer);
            T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
            rendering::raytracing::scene::procthor::precompute_indoor_positions(device, *env.scene, *env.renderer, default_params.fov, aspect);
        }

        env.renderer_initialized = true;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        if (env.use_target_mode) {
            sample_initial_state(device, env.dynamics, parameters.dynamics, state, rng);
            return;
        }

        auto indoor_pos = rendering::raytracing::scene::procthor::sample_indoor_position(device, *env.scene, rng);
        sample_initial_state(device, env.dynamics, parameters.dynamics, state, rng);

        state.position[0] = indoor_pos.position[0];
        state.position[1] = indoor_pos.position[1];
        state.position[2] = indoor_pos.position[2];

        state.orientation[0] = static_cast<T>(1);
        state.orientation[1] = static_cast<T>(0);
        state.orientation[2] = static_cast<T>(0);
        state.orientation[3] = static_cast<T>(0);

        T half_yaw = indoor_pos.yaw / static_cast<T>(2);
        state.orientation[0] = std::cos(half_yaw);
        state.orientation[1] = static_cast<T>(0);
        state.orientation[2] = static_cast<T>(0);
        state.orientation[3] = std::sin(half_yaw);

        for (TI i = 0; i < 3; i++) {
            state.linear_velocity[i] = static_cast<T>(0);
            state.angular_velocity[i] = static_cast<T>(0);
        }
    }

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

    namespace rl::environments::l2f_visual {
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::CameraData<typename SPEC::T> make_camera_for_state(DEVICE&, const MultirrotorVisual<SPEC>&, const typename MultirrotorVisual<SPEC>::Parameters& parameters, const typename MultirrotorVisual<SPEC>::State& state) {
            using T = typename SPEC::T;
            auto rotate_scene_yaw = [&](const T in[3], T out[3]){
                T c = std::cos(parameters.scene_yaw);
                T s = std::sin(parameters.scene_yaw);
                out[0] = c * in[0] - s * in[1];
                out[1] = s * in[0] + c * in[1];
                out[2] = in[2];
            };

            T cam_pos_local[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.offset_body, cam_pos_local);
            T cam_pos_world[3];
            rotate_scene_yaw(cam_pos_local, cam_pos_world);

            T cam_forward_local[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.forward_body, cam_forward_local);
            T cam_forward_world[3];
            rotate_scene_yaw(cam_forward_local, cam_forward_world);

            T cam_up_local[3];
            rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, parameters.camera_mount.up_body, cam_up_local);
            T cam_up_world[3];
            rotate_scene_yaw(cam_up_local, cam_up_world);

            T state_position_world[3];
            rotate_scene_yaw(state.position, state_position_world);

            const T position[3] = {
                state_position_world[0] + cam_pos_world[0] + parameters.scene_translation[0],
                state_position_world[1] + cam_pos_world[1] + parameters.scene_translation[1],
                state_position_world[2] + cam_pos_world[2] + parameters.scene_translation[2]
            };
            const T look_at[3] = {
                position[0] + cam_forward_world[0],
                position[1] + cam_forward_world[1],
                position[2] + cam_forward_world[2]
            };
            const T up[3] = {
                cam_up_world[0],
                cam_up_world[1],
                cam_up_world[2]
            };

            const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
            return make_camera_data(position, look_at, up, parameters.fov, aspect);
        }
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const rl::environments::observation::Image<typename SPEC::TI, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        using T = typename OBS_SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH * 3);

        auto camera = rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, state);
        for (TI i = 0; i < SPEC::NUM_ENVS; i++) {
            set(device, env.renderer->cameras, camera, i);
        }

        set_cameras(device, *env.renderer, env.renderer->cameras);
        render(device, *env.renderer);

        read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);

        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        const uint32_t* fb_data = data(env.renderer->frame_buffer);
        for (TI i = 0; i < CAM_PIXELS; i++) {
            const uint32_t rgba = fb_data[i];
            const T r = static_cast<T>((rgba >>  0) & 0xFF) / static_cast<T>(255);
            const T g = static_cast<T>((rgba >>  8) & 0xFF) / static_cast<T>(255);
            const T b = static_cast<T>((rgba >> 16) & 0xFF) / static_cast<T>(255);
            set(observation, 0, i * 3 + 0, r);
            set(observation, 0, i * 3 + 1, g);
            set(observation, 0, i * 3 + 2, b);
        }
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::ObservationPrivileged& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng) {
        observe(device, env.dynamics, parameters.dynamics, state, observation_type, observation, rng);
    }

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

        for (TI env_i = 0; env_i < num_envs; env_i++) {
            set(device, env.renderer->cameras, rl::environments::l2f_visual::make_camera_for_state(device, env, get_ref(device, parameters, env_i), get_ref(device, states, env_i)), env_i);
        }

        set_cameras(device, *env.renderer, env.renderer->cameras);
        render(device, *env.renderer);
        read_frame_buffer(device, *env.renderer, out_pixels);
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters){
        std::string json_string = "{";
        std::string dynamics_json = rl_tools::json(device, env.dynamics, parameters.dynamics);
        json_string += dynamics_json.substr(1, dynamics_json.size() - 2);
        json_string += ", \"visual\": {";
        json_string += "\"scene_translation\": [";
        for(typename SPEC::TI i = 0; i < 3; i++){
            json_string += std::to_string(parameters.scene_translation[i]);
            if(i < 2) json_string += ", ";
        }
        json_string += "], \"scene_yaw\": " + std::to_string(parameters.scene_yaw);
        json_string += ", \"scene_hash\": \"";
        for(unsigned i = 0; i < rl::environments::l2f_visual::SceneHash::HASH_SIZE; i++){
            char hex[3];
            std::snprintf(hex, sizeof(hex), "%02x", parameters.scene_hash.hash[i]);
            json_string += hex;
        }
        json_string += "\", \"cam_width\": " + std::to_string(SPEC::CAM_WIDTH);
        json_string += ", \"cam_height\": " + std::to_string(SPEC::CAM_HEIGHT);
        json_string += ", \"fov\": " + std::to_string(parameters.fov);
        json_string += "}}";
        return json_string;
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state){
        return rl_tools::json(device, env.dynamics, parameters.dynamics, state);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
