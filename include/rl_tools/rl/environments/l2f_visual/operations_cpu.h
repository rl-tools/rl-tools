#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CPU_H

#include "multirotor_visual.h"
#include "operations_generic.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/raytracing/scene/procthor/operations_cpu.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    // the env owns only its dynamics — the renderer and scene metadata are non-owning
    // references wired in by the target (see rendering::raytracing::AssetLibrary)
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        malloc(device, env.dynamics);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        free(device, env.dynamics);
        env.renderer = nullptr;
        env.scene = nullptr;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env) {
        init(device, env.dynamics);
    }

    // scene-under-drone (matching the CUDA training targets): the dynamics state stays near the
    // origin; the sampled indoor position and yaw place the scene via parameters
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;

        sample_initial_state(device, env.dynamics, parameters.dynamics, state, rng);
        if (env.use_target_mode) {
            return;
        }

        auto indoor_pos = rendering::raytracing::scene::procthor::sample_indoor_position(device, *env.scene, rng);
        parameters.scene_translation[0] = indoor_pos.position[0];
        parameters.scene_translation[1] = indoor_pos.position[1];
        parameters.scene_translation[2] = indoor_pos.position[2];
        parameters.scene_yaw = random::uniform_real_distribution(device.random, static_cast<T>(0), static_cast<T>(2) * math::PI<T>, rng);
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
        RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<typename SPEC::T> make_camera_for_state(DEVICE& device, const MultirrotorVisual<SPEC>&, const typename MultirrotorVisual<SPEC>::Parameters& parameters, const typename MultirrotorVisual<SPEC>::State& state) {
            using T = typename SPEC::T;
            const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
            const T scene_yaw_cos = math::cos(device.math, parameters.scene_yaw);
            const T scene_yaw_sin = math::sin(device.math, parameters.scene_yaw);
            return hyperdrone::make_camera<DEVICE, T>(device, parameters.camera_mount, parameters.fov, state.orientation, state.position, aspect, parameters.scene_translation, scene_yaw_cos, scene_yaw_sin);
        }
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const rl::environments::observation::Image<typename SPEC::TI, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, SPEC::IMAGE_CHANNELS>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        using T = typename OBS_SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == SPEC::CAM_HEIGHT * SPEC::CAM_WIDTH * SPEC::IMAGE_CHANNELS);

        auto camera = rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, state);
        std::array<rendering::raytracing::Camera<typename SPEC::T>, SPEC::NUM_ENVS> camera_staging;
        camera_staging.fill(camera);
        Tensor<tensor::Specification<rendering::raytracing::Camera<typename SPEC::T>, TI, typename decltype(env.renderer->cameras)::SPEC::SHAPE>> camera_alias;
        camera_alias._data = camera_staging.data();
        copy(device, env.renderer->device, camera_alias, cameras(device, *env.renderer));
        if constexpr (SPEC::RENDERER_SPEC::ENABLE_MOTION_BLUR) {
            copy(device, env.renderer->device, camera_alias, cameras_open(device, *env.renderer));
        }
        render(device, *env.renderer);

        // the ray gen writes the float observation directly; host readers stage after the sync
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        std::vector<float> rgb_staging(SPEC::HAS_RGB ? CAM_PIXELS * 3 : 0);
        std::vector<float> depth_staging(SPEC::HAS_DEPTH ? CAM_PIXELS : 0);
        if constexpr (SPEC::HAS_RGB) {
            auto observation_camera_0 = view(device, env.renderer->observation, 0);
            Tensor<tensor::Specification<float, TI, typename decltype(observation_camera_0)::SPEC::SHAPE>> rgb_alias;
            rgb_alias._data = rgb_staging.data();
            copy(env.renderer->device, device, observation_camera_0, rgb_alias);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            auto depth_camera_0 = view(device, env.renderer->depth_buffer, 0);
            Tensor<tensor::Specification<float, TI, typename decltype(depth_camera_0)::SPEC::SHAPE>> depth_alias;
            depth_alias._data = depth_staging.data();
            copy(env.renderer->device, device, depth_camera_0, depth_alias);
        }
        for (TI i = 0; i < CAM_PIXELS; i++) {
            if constexpr (SPEC::HAS_RGB) {
                set(observation, 0, i * SPEC::IMAGE_CHANNELS + 0, static_cast<T>(rgb_staging[i * 3 + 0]));
                set(observation, 0, i * SPEC::IMAGE_CHANNELS + 1, static_cast<T>(rgb_staging[i * 3 + 1]));
                set(observation, 0, i * SPEC::IMAGE_CHANNELS + 2, static_cast<T>(rgb_staging[i * 3 + 2]));
            }
            if constexpr (SPEC::HAS_DEPTH) {
                constexpr TI DEPTH_OFFSET = SPEC::HAS_RGB ? 3 : 0;
                set(observation, 0, i * SPEC::IMAGE_CHANNELS + DEPTH_OFFSET, static_cast<T>(depth_staging[i]));
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::Parameters& parameters, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::State& state, const typename rl::environments::l2f_visual::MultirrotorVisual<SPEC>::ObservationPrivileged& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng) {
        observe(device, env.dynamics, parameters.dynamics, state, observation_type, observation, rng);
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
