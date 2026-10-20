#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CPU_H

#include "spy.h"
#include "operations_cpu_spectator.h"
#include <algorithm>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC, typename BACKEND, typename SNAPSHOT_SPEC>
    void copy_spy_overlay(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex index, rendering::raytracing::OverlaySnapshot<SNAPSHOT_SPEC>& snapshot) {
        copy(device, renderer, index, snapshot);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "capture_optix.h"
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename CONFIG>
    void capture_spy(DEVICE& device, rendering::raytracing::spy::State<CONFIG>& state) {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using SOURCE = typename CONFIG::RENDERER;
        using SPEC = typename SOURCE::SPEC;
        using CAMERA_SPEC = tensor::Specification<rendering::raytracing::Camera<T>, TI, tensor::Shape<TI, 1>>;
        auto& renderer = *state.renderer;
        auto& buffer = state.buffer;
        auto& job = buffer.job;
        Tensor<CAMERA_SPEC> camera_source, camera_destination;
        camera_source._data = data(cameras(device, renderer)) + state.camera;
        camera_destination._data = &buffer.camera;
        copy(renderer.device, device, camera_source, camera_destination);
        job.scene = state.scene;
        job.pool = state.pool;
        job.metadata = state.metadata;
        job.num_overlays = 0;
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            for(size_t slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++) {
                const auto overlay = renderer.attachments[state.camera * SPEC::MAX_OVERLAYS_PER_CAMERA + slot];
                if(overlay != SOURCE::INVALID_OVERLAY) {
                    const auto index = rendering::raytracing::OverlayIndex{overlay};
                    job.overlay_indices[job.num_overlays] = index;
                    copy_spy_overlay(device, renderer, index, job.overlays[job.num_overlays]);
                    job.num_overlays++;
                }
            }
        }
        if(state.reset_camera) {
            auto& host = get_rendering_device(device);
            T direction[3];
            for(size_t axis = 0; axis < 3; axis++) {
                state.position[axis] = buffer.camera.pos[axis];
                direction[axis] = buffer.camera.dir_00[axis] + T(0.5) * (buffer.camera.dir_du[axis] + buffer.camera.dir_dv[axis]);
            }
            state.yaw = math::atan2(host.math, direction[1], direction[0]);
            state.pitch = math::atan2(host.math, direction[2], math::sqrt(host.math, direction[0] * direction[0] + direction[1] * direction[1]));
            state.reset_camera = false;
        }
        auto& host = get_rendering_device(device);
        const T look[3] = {state.position[0] + math::cos(host.math, state.yaw) * math::cos(host.math, state.pitch), state.position[1] + math::sin(host.math, state.yaw) * math::cos(host.math, state.pitch), state.position[2] + math::sin(host.math, state.pitch)};
        const T up[3] = {0, 0, 1};
        job.camera = make_camera_data(state.position, look, up, T(80), T(CONFIG::SPECTATOR_CONFIG::CAM_WIDTH) / T(CONFIG::SPECTATOR_CONFIG::CAM_HEIGHT));
        job.captured_at = spy_time();
    }

    template <typename DEVICE, typename CONFIG>
    void free(DEVICE& device, rendering::raytracing::Spy<CONFIG>& spy) {
        if(spy.state == nullptr) {
            return;
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id()) {
            throw std::runtime_error("Raytracing spy must be closed on its creating thread");
        }
        stop_spy_spectator(device, state.spectator);
        rendering::raytracing::spy::ui::destroy(state.window);
        delete spy.state;
        spy.state = nullptr;
    }

    template <typename DEVICE, typename CONFIG>
    void malloc(DEVICE&, rendering::raytracing::Spy<CONFIG>& spy) {
        using SOURCE = typename CONFIG::RENDERER::SPEC;
        using SPECTATOR = typename CONFIG::SPECTATOR_CONFIG;
        static_assert(SPECTATOR::CAM_WIDTH > 0 && SPECTATOR::CAM_HEIGHT > 0);
        static_assert(SPECTATOR::NUM_OVERLAYS == SOURCE::NUM_OVERLAYS && SPECTATOR::MAX_OVERLAY_INSTANCES == SOURCE::MAX_OVERLAY_INSTANCES && SPECTATOR::MAX_OVERLAYS_PER_CAMERA == SOURCE::MAX_OVERLAYS_PER_CAMERA, "Raytracing spy must preserve its source overlay layout");
        if(spy.state != nullptr) {
            throw std::logic_error("Raytracing spy is already allocated");
        }
        spy.state = new rendering::raytracing::spy::State<CONFIG>;
        spy.state->owner = std::this_thread::get_id();
    }

    template <typename DEVICE, typename CONFIG>
    void init(DEVICE& device, rendering::raytracing::Spy<CONFIG>& spy, typename CONFIG::RENDERER& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::SceneMetadata<typename CONFIG::T>& metadata, size_t camera = 0) {
        if(spy.state == nullptr || spy.state->renderer != nullptr) {
            throw std::logic_error("Raytracing spy must be allocated and initialized exactly once");
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id()) {
            throw std::logic_error("Raytracing spy must be initialized on its creating thread");
        }
        if(camera >= CONFIG::RENDERER::SPEC::NUM_CAMERAS) {
            throw std::out_of_range("Raytracing spy camera index out of range");
        }
        if(renderer.backend == nullptr) {
            throw std::logic_error("Raytracing spy requires an initialized renderer");
        }
        state.renderer = &renderer;
        state.scene = &scene;
        state.pool = &pool;
        state.metadata = metadata;
        state.camera = camera;
        state.window = rendering::raytracing::spy::ui::create("Raytracing spy", {}, 0, 0, 0, 0, CONFIG::SPECTATOR_CONFIG::CAM_WIDTH, CONFIG::SPECTATOR_CONFIG::CAM_HEIGHT);
        start_spy_spectator(device, state.spectator);
        state.open = true;
    }

    template <typename DEVICE, typename CONFIG>
    bool update(DEVICE& device, rendering::raytracing::Spy<CONFIG>& spy, typename CONFIG::RENDERER& renderer) {
        namespace ui = rendering::raytracing::spy::ui;
        using T = typename CONFIG::T;
        if(spy.state == nullptr || !spy.state->open) {
            return false;
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id() || state.renderer != &renderer) {
            throw std::logic_error("Raytracing spy update requires its creating thread and original renderer");
        }
        const double now = spy_time();
        if(now - state.last_update < CONFIG::UI_REFRESH_INTERVAL) {
            return true;
        }
        const auto input = ui::update(state.window, -1, state.captured_at > 0 ? now - state.captured_at : -1, "");
        if(input.closed) {
            free(device, spy);
            return false;
        }
        state.reset_camera = state.reset_camera || input.reset_camera;
        const T dt = T(std::min(0.1, now - state.last_update));
        state.last_update = now;
        if(!state.reset_camera) {
            auto& host = get_rendering_device(device);
            state.yaw += input.yaw_delta;
            state.pitch = std::max(T(-1.55334), std::min(T(1.55334), state.pitch + T(input.pitch_delta)));
            const T cy = math::cos(host.math, state.yaw), sy = math::sin(host.math, state.yaw), cp = math::cos(host.math, state.pitch), sp = math::sin(host.math, state.pitch);
            state.position[0] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * cy * cp + T(input.right) * sy);
            state.position[1] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * sy * cp - T(input.right) * cy);
            state.position[2] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * sp + T(input.up));
        }
        auto& buffer = state.buffer;
        if(poll_spy_spectator(device, state.spectator, buffer.image)) {
            ui::upload_spectator(state.window, buffer.image.pixels.data());
            state.captured_at = buffer.image.captured_at;
        }
        if(now >= state.next_capture) {
            state.next_capture = now + CONFIG::REFRESH_INTERVAL;
            capture_spy(device, state);
            submit_spy_spectator(device, state.spectator, buffer.job);
        }
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
