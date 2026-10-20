#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CPU_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CPU_SPECTATOR_H

#include "spectator.h"
#include "../operations_cpu_mux.h"

#include <chrono>
#include <stdexcept>
#include <utility>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    inline double spy_time() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    template <typename DEVICE, typename BACKEND>
    int spy_spectator_device_id(DEVICE&, BACKEND) {
        return 0;
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void malloc(DEVICE&, rendering::raytracing::spy::SpectatorBuffer<CONFIG, BACKEND>&, int) {
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void free(DEVICE& device, rendering::raytracing::spy::SpectatorBuffer<CONFIG, BACKEND>& buffer) {
        if(buffer.renderer.backend != nullptr) {
            free(device, buffer.renderer);
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void copy_spy_overlays(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::raytracing::spy::SpectatorJob<SPEC>& job) {
        if(job.num_overlays > SPEC::MAX_OVERLAYS_PER_CAMERA) {
            throw std::invalid_argument("Raytracing spy: too many overlay snapshots");
        }
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            bool attachments_changed = false;
            for(size_t slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++) {
                const size_t overlay = slot < job.num_overlays ? job.overlay_indices[slot].index : renderer.INVALID_OVERLAY;
                if(slot < job.num_overlays && overlay >= SPEC::NUM_OVERLAYS) {
                    throw std::invalid_argument("Raytracing spy: overlay index out of range");
                }
                attachments_changed = attachments_changed || renderer.attachments[slot] != overlay;
            }
            if(attachments_changed) {
                for(size_t slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++) {
                    const auto overlay = renderer.attachments[slot];
                    if(overlay != renderer.INVALID_OVERLAY) {
                        detach(device, renderer, 0, rendering::raytracing::OverlayIndex{overlay});
                    }
                }
                for(size_t slot = 0; slot < job.num_overlays; slot++) {
                    attach(device, renderer, 0, job.overlay_indices[slot]);
                }
            }
            for(size_t slot = 0; slot < job.num_overlays; slot++) {
                copy(device, job.overlays[slot], renderer, job.overlay_indices[slot]);
            }
        }
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void render_spy_spectator(DEVICE& device, rendering::raytracing::spy::SpectatorBuffer<CONFIG, BACKEND>& buffer, const rendering::raytracing::spy::SpectatorJob<rendering::raytracing::Specification<CONFIG>>& job, rendering::raytracing::spy::SpectatorImage& result) {
        using SPEC = typename rendering::raytracing::spy::SpectatorBuffer<CONFIG, BACKEND>::SPEC;
        auto& renderer = buffer.renderer;
        Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
        camera_alias._data = const_cast<rendering::raytracing::Camera<typename CONFIG::T>*>(&job.camera);
        copy(device, renderer.device, camera_alias, cameras(device, renderer));
        copy_spy_overlays(device, renderer, job);
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            update(device, renderer);
        }
        render(device, renderer);
        result.pixels.resize(SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT);
        Tensor<rl_tools::rendering::raytracing::outputs::TensorSpecification<SPEC, rl_tools::rendering::raytracing::outputs::RGB<>>> frame_alias;
        frame_alias._data = result.pixels.data();
        copy(renderer.device, device, output(device, renderer, rl_tools::rendering::raytracing::outputs::RGB<>{}), frame_alias);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "operations_cuda_spectator.h"
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename CONFIG>
    void run_spy_spectator(rendering::raytracing::spy::Spectator<CONFIG>& spectator, int device_id) {
        namespace spy = rendering::raytracing::spy;
        using BUFFER = spy::SpectatorBuffer<CONFIG>;
        using SPEC = typename BUFFER::SPEC;
        BUFFER buffer{};
        spy::SpectatorJob<SPEC> job;
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
                if(job.scene == nullptr || job.pool == nullptr) {
                    throw std::invalid_argument("Raytracing spy: invalid spectator scene snapshot");
                }
                if(current_scene != job.scene || current_pool != job.pool) {
                    if(buffer.renderer.backend != nullptr) {
                        free(buffer.device, buffer.renderer);
                    }
                    malloc(buffer.device, buffer.renderer);
                    init(buffer.device, buffer.renderer, *job.scene, *job.pool, job.metadata);
                    current_scene = job.scene;
                    current_pool = job.pool;
                }
                render_spy_spectator(buffer.device, buffer, job, result);
                result.view = job.view;
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
    void start_spy_spectator(DEVICE& device, rendering::raytracing::spy::Spectator<CONFIG>& spectator) {
        if(spectator.worker.joinable()) {
            throw std::logic_error("Raytracing spy: spectator is already running");
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
    bool submit_spy_spectator(DEVICE&, rendering::raytracing::spy::Spectator<CONFIG>& spectator, rendering::raytracing::spy::SpectatorJob<rendering::raytracing::Specification<CONFIG>>& job) {
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
    bool poll_spy_spectator(DEVICE&, rendering::raytracing::spy::Spectator<CONFIG>& spectator, rendering::raytracing::spy::SpectatorImage& result) {
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
    void stop_spy_spectator(DEVICE&, rendering::raytracing::spy::Spectator<CONFIG>& spectator) {
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
