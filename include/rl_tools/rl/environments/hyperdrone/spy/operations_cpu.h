#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CPU_H

#include "spy.h"
#include "operations_cpu_spectator.h"
#include "capture_cpu.h"
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "capture_optix.h"
#if defined(__CUDACC__)
#include "capture_cuda.h"
#endif
#endif
#include <chrono>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    inline double spy_time() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    template <typename CONFIG>
    void prepare_spy_capture(rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& environment, double now) {
        using WORLD = typename CONFIG::WORLD;
        auto& buffer = state.buffer;
        for(size_t member = 0; member < CONFIG::ENVIRONMENT::N_WORLDS; member++) {
            auto& world = environment.worlds[member];
            auto& frame = buffer.frames[member];
            const bool valid = !world.render_pending && world.history_step > 0;
            if(!frame.valid || frame.scene_version != world.scene_version || frame.sequence != world.history_step) {
                frame.captured_at = now;
            }
            frame.scene_version = world.scene_version;
            frame.sequence = world.history_step;
            frame.valid = valid;
        }
        buffer.selected_camera = state.selected_camera;
        const size_t member = state.selected_camera / (WORLD::INSTANCES * WORLD::N_VIEWS);
        buffer.selected_valid = buffer.frames[member].valid;
        if(buffer.selected_valid) {
            auto& world = environment.worlds[member];
            const auto& slot = world.slots[world.active_slot];
            auto& job = buffer.job;
            job.scene = &environment.shared.library.scenes[slot.scene_id];
            job.pool = &environment.shared.library.pool;
            job.metadata = slot.metadata;
            job.member = member;
            job.instance = (state.selected_camera / WORLD::N_VIEWS) % WORLD::INSTANCES;
            job.scene_version = world.scene_version;
            job.captured_at = buffer.frames[member].captured_at;
        }
    }

    template <typename DEVICE, typename CONFIG>
    void submit_spy_view(DEVICE& device, rl::environments::hyperdrone::spy::State<CONFIG>& state) {
        using T = typename CONFIG::T;
        using WORLD = typename CONFIG::WORLD;
        auto& buffer = state.buffer;
        if(!buffer.selected_valid || buffer.selected_camera != state.selected_camera) {
            return;
        }
        auto& job = buffer.job;
        if(state.environment->worlds[job.member].scene_version != job.scene_version) {
            return;
        }
        auto& view = state.views[job.member * WORLD::INSTANCES + job.instance];
        auto& host = get_rendering_device(device);
        if(!view.initialized || view.scene_version != job.scene_version || state.reset_camera) {
            const auto& camera = buffer.geometry.camera;
            T direction[3];
            for(size_t axis = 0; axis < 3; axis++) {
                view.position[axis] = camera.pos[axis];
                direction[axis] = camera.dir_00[axis] + T(0.5) * (camera.dir_du[axis] + camera.dir_dv[axis]);
            }
            view.yaw = math::atan2(host.math, direction[1], direction[0]);
            view.pitch = math::atan2(host.math, direction[2], math::sqrt(host.math, direction[0] * direction[0] + direction[1] * direction[1]));
            view.scene_version = job.scene_version;
            view.initialized = true;
            state.reset_camera = false;
        }
        const T look[3] = {view.position[0] + math::cos(host.math, view.yaw) * math::cos(host.math, view.pitch), view.position[1] + math::sin(host.math, view.yaw) * math::cos(host.math, view.pitch), view.position[2] + math::sin(host.math, view.pitch)};
        const T up[3] = {0, 0, 1};
        job.camera = make_camera_data(view.position, look, up, T(80), T(CONFIG::SPECTATOR_CONFIG::CAM_WIDTH) / T(CONFIG::SPECTATOR_CONFIG::CAM_HEIGHT));
        job.overlay.assign(buffer.geometry.slots, buffer.geometry.slots + WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES);
        submit_spy_spectator(device, state.spectator, job);
    }

    template <typename DEVICE, typename CONFIG>
    void free(DEVICE& device, rl::environments::hyperdrone::Spy<CONFIG>& spy) {
        if(spy.state == nullptr) {
            return;
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id()) {
            throw std::runtime_error("HyperDrone spy must be closed on its creating thread");
        }
        stop_spy_spectator(device, state.spectator);
        free_spy_capture(device, state, typename CONFIG::BACKEND{});
        rl::environments::hyperdrone::spy::ui::destroy(state.window);
        delete spy.state;
        spy.state = nullptr;
    }

    template <typename DEVICE, typename CONFIG>
    void malloc(DEVICE& device, rl::environments::hyperdrone::Spy<CONFIG>& spy) {
        static_assert(CONFIG::WORLD::RENDERER_SPEC::HAS_RGB, "HyperDrone spy requires RGB output");
        static_assert(CONFIG::ATLAS_WIDTH > 0 && CONFIG::ATLAS_HEIGHT > 0);
        if(spy.state != nullptr) {
            throw std::logic_error("HyperDrone spy is already allocated");
        }
        spy.state = new rl::environments::hyperdrone::spy::State<CONFIG>;
        spy.state->owner = std::this_thread::get_id();
        try {
            constexpr size_t CAMERAS = CONFIG::ENVIRONMENT::N_WORLDS * CONFIG::WORLD::INSTANCES * CONFIG::WORLD::N_VIEWS;
            auto& state = *spy.state;
            while(state.columns * state.columns < CAMERAS) {
                state.columns++;
            }
            state.rows = (CAMERAS + state.columns - 1) / state.columns;
            if(state.columns > CONFIG::ATLAS_WIDTH || state.rows > CONFIG::ATLAS_HEIGHT) {
                throw std::invalid_argument("HyperDrone spy atlas needs at least one pixel per camera");
            }
            state.views.resize(CONFIG::ENVIRONMENT::N_WORLDS * CONFIG::WORLD::INSTANCES);
            state.buffer.job.overlay.reserve(CONFIG::WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES);
            malloc_spy_capture(device, state, typename CONFIG::BACKEND{});
        }
        catch(...) {
            free(device, spy);
            throw;
        }
    }

    template <typename DEVICE, typename CONFIG>
    void init(DEVICE& device, rl::environments::hyperdrone::Spy<CONFIG>& spy, typename CONFIG::ENVIRONMENT& environment) {
        if(spy.state == nullptr || spy.state->environment != nullptr) {
            throw std::logic_error("HyperDrone spy must be allocated and initialized exactly once");
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id()) {
            throw std::logic_error("HyperDrone spy must be initialized on its creating thread");
        }
        state.environment = &environment;
        state.window = rl::environments::hyperdrone::spy::ui::create(CONFIG::ENVIRONMENT::N_WORLDS, CONFIG::WORLD::INSTANCES, CONFIG::WORLD::N_CAMERA_AGENTS, CONFIG::WORLD::N_CAMERAS_PER_AGENT, CONFIG::ATLAS_WIDTH, CONFIG::ATLAS_HEIGHT, state.columns, state.rows, CONFIG::SPECTATOR_CONFIG::CAM_WIDTH, CONFIG::SPECTATOR_CONFIG::CAM_HEIGHT);
        start_spy_spectator(device, state.spectator);
        state.open = true;
    }

    template <typename DEVICE, typename CONFIG>
    bool update(DEVICE& device, rl::environments::hyperdrone::Spy<CONFIG>& spy, typename CONFIG::ENVIRONMENT& environment) {
        namespace ui = rl::environments::hyperdrone::spy::ui;
        using WORLD = typename CONFIG::WORLD;
        using T = typename CONFIG::T;
        if(spy.state == nullptr || !spy.state->open) {
            return false;
        }
        auto& state = *spy.state;
        if(state.owner != std::this_thread::get_id() || state.environment != &environment) {
            throw std::logic_error("HyperDrone spy update requires its creating thread and original environment");
        }
        const double now = spy_time();
        if(now - state.last_update < CONFIG::UI_REFRESH_INTERVAL) {
            return true;
        }
        const size_t member = state.selected_camera / (WORLD::INSTANCES * WORLD::N_VIEWS);
        const auto& frame = state.displayed_frames[member];
        const bool current = frame.valid && frame.scene_version == environment.worlds[member].scene_version;
        if(state.views[state.selected_camera / WORLD::N_VIEWS].scene_version != environment.worlds[member].scene_version) {
            state.spectator_captured_at = 0;
        }
        const auto input = ui::update(state.window, current ? now - frame.captured_at : -1, state.spectator_captured_at > 0 ? now - state.spectator_captured_at : -1, current ? "" : "Waiting for an observed RGB frame");
        if(input.closed) {
            free(device, spy);
            return false;
        }
        if(state.selected_camera != input.camera) {
            state.selected_camera = input.camera;
            state.spectator_captured_at = 0;
        }
        state.reset_camera = state.reset_camera || input.reset_camera;
        const T dt = T(std::min(0.1, now - state.last_update));
        state.last_update = now;
        auto& view = state.views[state.selected_camera / WORLD::N_VIEWS];
        if(view.initialized) {
            auto& host = get_rendering_device(device);
            view.yaw += input.yaw_delta;
            view.pitch = std::max(T(-1.55334), std::min(T(1.55334), view.pitch + T(input.pitch_delta)));
            const T cy = math::cos(host.math, view.yaw), sy = math::sin(host.math, view.yaw), cp = math::cos(host.math, view.pitch), sp = math::sin(host.math, view.pitch);
            view.position[0] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * cy * cp + T(input.right) * sy);
            view.position[1] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * sy * cp - T(input.right) * cy);
            view.position[2] += dt * CONFIG::MOVE_SPEED * (T(input.forward) * sp + T(input.up));
        }
        auto& buffer = state.buffer;
        if(poll_spy_spectator(device, state.spectator, buffer.image)) {
            const auto& image = buffer.image;
            const size_t selected_member = state.selected_camera / (WORLD::INSTANCES * WORLD::N_VIEWS);
            const size_t selected_instance = (state.selected_camera / WORLD::N_VIEWS) % WORLD::INSTANCES;
            if(image.member == selected_member && image.instance == selected_instance && image.scene_version == environment.worlds[selected_member].scene_version) {
                ui::upload_spectator(state.window, image.pixels.data());
                state.spectator_captured_at = image.captured_at;
            }
        }
        if(buffer.pending && poll_spy_capture(device, state, typename CONFIG::BACKEND{})) {
            ui::upload_atlas(state.window, buffer.atlas);
            state.displayed_frames = buffer.frames;
            submit_spy_view(device, state);
            buffer.pending = false;
        }
        if(now >= state.next_capture) {
            state.next_capture = now + CONFIG::REFRESH_INTERVAL;
            if(!buffer.pending) {
                prepare_spy_capture(state, environment, now);
                capture_spy(device, state, environment, typename CONFIG::BACKEND{});
                buffer.pending = true;
            }
        }
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
