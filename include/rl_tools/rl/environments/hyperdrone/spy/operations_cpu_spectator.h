#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CPU_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CPU_SPECTATOR_H

#include "spectator.h"
#include "../../../../rendering/raytracing/operations_cpu_mux.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename BACKEND>
    int spy_spectator_device_id(DEVICE&, BACKEND) {
        return 0;
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void malloc(DEVICE&, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, BACKEND>&, int) {
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void free(DEVICE& device, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, BACKEND>& buffer) {
        if(buffer.renderer.backend != nullptr) {
            free(device, buffer.renderer);
        }
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void render_spy_spectator(DEVICE& device, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, BACKEND>& buffer, const rl::environments::hyperdrone::spy::SpectatorJob<typename CONFIG::T>& job, rl::environments::hyperdrone::spy::SpectatorImage& result) {
        using SPEC = typename rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, BACKEND>::SPEC;
        auto& renderer = buffer.renderer;
        Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
        camera_alias._data = const_cast<rendering::raytracing::Camera<typename CONFIG::T>*>(&job.camera);
        copy(device, renderer.device, camera_alias, cameras(device, renderer));
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            for(size_t slot_i = 0; slot_i < SPEC::MAX_OVERLAY_INSTANCES; slot_i++) {
                const auto& source = job.overlay[slot_i];
                auto& destination = renderer.overlays[0].slots[slot_i];
                destination.active = source.active;
                destination.object = source.object;
                std::copy(source.transform, source.transform + 12, data(transforms(device, renderer)) + slot_i * 12);
            }
            renderer.overlays[0].dirty = true;
            update(device, renderer);
        }
        render(device, renderer);
        result.pixels.resize(SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT);
        Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
        frame_alias._data = result.pixels.data();
        copy(renderer.device, device, frame_buffer(device, renderer), frame_alias);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "operations_cuda_spectator.h"
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename CONFIG>
    void run_spy_spectator(rl::environments::hyperdrone::spy::Spectator<CONFIG>& spectator, int device_id) {
        namespace spy = rl::environments::hyperdrone::spy;
        using BUFFER = spy::SpectatorBuffer<CONFIG>;
        using SPEC = typename BUFFER::SPEC;
        BUFFER buffer{};
        spy::SpectatorJob<typename CONFIG::T> job;
        spy::SpectatorImage result;
        const rendering::raytracing::Scene* current_scene = nullptr;
        const rendering::raytracing::AssetPool* current_pool = nullptr;
        try {
            malloc(buffer.device, buffer, device_id);
            for(;;) {
                {
                    std::unique_lock<std::mutex> lock(spectator.mutex);
                    spectator.wake.wait(lock, [&spectator] { return spectator.stopping || spectator.requested; });
                    if(spectator.stopping) {
                        break;
                    }
                    std::swap(job, spectator.request);
                    spectator.requested = false;
                }
                if(job.scene == nullptr || job.pool == nullptr || job.overlay.size() != SPEC::MAX_OVERLAY_INSTANCES) {
                    throw std::invalid_argument("HyperDrone spy: invalid spectator scene snapshot");
                }
                if(current_scene != job.scene || current_pool != job.pool) {
                    if(buffer.renderer.backend != nullptr) {
                        free(buffer.device, buffer.renderer);
                    }
                    malloc(buffer.device, buffer.renderer);
                    init(buffer.device, buffer.renderer, *job.scene, *job.pool, job.metadata);
                    if constexpr(SPEC::ENABLE_OVERLAYS) {
                        attach(buffer.device, buffer.renderer, 0, rendering::raytracing::OverlayIndex{0});
                    }
                    current_scene = job.scene;
                    current_pool = job.pool;
                }
                render_spy_spectator(buffer.device, buffer, job, result);
                result.member = job.member;
                result.instance = job.instance;
                result.scene_version = job.scene_version;
                result.captured_at = job.captured_at;
                {
                    std::lock_guard<std::mutex> lock(spectator.mutex);
                    std::swap(result, spectator.result);
                    spectator.ready = true;
                    spectator.working = false;
                }
            }
        }
        catch(...) {
            std::lock_guard<std::mutex> lock(spectator.mutex);
            spectator.error = std::current_exception();
            spectator.working = false;
        }
        free(buffer.device, buffer);
    }

    template <typename DEVICE, typename CONFIG>
    void start_spy_spectator(DEVICE& device, rl::environments::hyperdrone::spy::Spectator<CONFIG>& spectator) {
        if(spectator.worker.joinable()) {
            throw std::logic_error("HyperDrone spy: spectator is already running");
        }
        spectator.stopping = false;
        spectator.requested = false;
        spectator.working = false;
        spectator.ready = false;
        spectator.error = nullptr;
        const int device_id = spy_spectator_device_id(device, rendering::raytracing::backends::Default{});
        spectator.worker = std::thread([&spectator, device_id] { run_spy_spectator(spectator, device_id); });
    }

    template <typename DEVICE, typename CONFIG>
    bool submit_spy_spectator(DEVICE&, rl::environments::hyperdrone::spy::Spectator<CONFIG>& spectator, rl::environments::hyperdrone::spy::SpectatorJob<typename CONFIG::T>& job) {
        std::unique_lock<std::mutex> lock(spectator.mutex, std::try_to_lock);
        if(!lock.owns_lock()) {
            return false;
        }
        if(spectator.error) {
            std::rethrow_exception(spectator.error);
        }
        if(spectator.stopping || spectator.working || spectator.ready || !spectator.worker.joinable()) {
            return false;
        }
        std::swap(spectator.request, job);
        spectator.requested = true;
        spectator.working = true;
        lock.unlock();
        spectator.wake.notify_one();
        return true;
    }

    template <typename DEVICE, typename CONFIG>
    bool poll_spy_spectator(DEVICE&, rl::environments::hyperdrone::spy::Spectator<CONFIG>& spectator, rl::environments::hyperdrone::spy::SpectatorImage& result) {
        std::unique_lock<std::mutex> lock(spectator.mutex, std::try_to_lock);
        if(!lock.owns_lock()) {
            return false;
        }
        if(spectator.error) {
            std::rethrow_exception(spectator.error);
        }
        if(!spectator.ready) {
            return false;
        }
        std::swap(result, spectator.result);
        spectator.ready = false;
        return true;
    }

    template <typename DEVICE, typename CONFIG>
    void stop_spy_spectator(DEVICE&, rl::environments::hyperdrone::spy::Spectator<CONFIG>& spectator) {
        {
            std::lock_guard<std::mutex> lock(spectator.mutex);
            spectator.stopping = true;
        }
        spectator.wake.notify_one();
        if(spectator.worker.joinable()) {
            spectator.worker.join();
        }
        spectator.request = {};
        spectator.result = {};
        spectator.error = nullptr;
        spectator.requested = false;
        spectator.working = false;
        spectator.ready = false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
