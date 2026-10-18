#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_OPTIX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_OPTIX_H

#include "spy.h"
#include "../../../../rendering/raytracing/backends/optix/operations_cuda.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::spy {
    struct CaptureStorage<rendering::raytracing::backends::Optix>::State {
        uint32_t* device_atlas = nullptr;
        void* device_geometry = nullptr;
        void* host_geometry = nullptr;
        cudaStream_t copy_stream = nullptr;
        cudaStream_t producer_stream = nullptr;
        cudaEvent_t packed = nullptr;
        cudaEvent_t ready = nullptr;
        bool capturing = false;
    };

    template <typename CONFIG>
    struct OptixGeometry {
        static constexpr size_t SLOTS = Geometry<CONFIG>::SLOTS;
        rendering::raytracing::Camera<typename CONFIG::T> camera;
        rendering::raytracing::backends::optix::OverlaySlotStructure structure[SLOTS > 0 ? SLOTS : 1];
        float transforms[SLOTS > 0 ? SLOTS : 1][12];
    };
}

namespace rl_tools {
    inline void spy_capture_cuda_check(cudaError_t status) {
        if(status != cudaSuccess) {
            throw std::runtime_error(std::string("HyperDrone spy capture: ") + cudaGetErrorString(status));
        }
    }

    template <typename DEVICE, typename CONFIG>
    void malloc_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, rendering::raytracing::backends::Optix) {
        namespace spy = rl::environments::hyperdrone::spy;
        auto* capture = new spy::CaptureStorage<rendering::raytracing::backends::Optix>::State{};
        state.buffer.capture.state = capture;
        const size_t atlas_bytes = CONFIG::ATLAS_WIDTH * CONFIG::ATLAS_HEIGHT * sizeof(uint32_t);
        spy_capture_cuda_check(cudaMallocHost(&state.buffer.atlas, atlas_bytes));
        spy_capture_cuda_check(cudaMalloc(&capture->device_atlas, atlas_bytes));
        std::fill_n(state.buffer.atlas, CONFIG::ATLAS_WIDTH * CONFIG::ATLAS_HEIGHT, 0xff202020);
        spy_capture_cuda_check(cudaMemcpy(capture->device_atlas, state.buffer.atlas, atlas_bytes, cudaMemcpyHostToDevice));
        spy_capture_cuda_check(cudaMalloc(&capture->device_geometry, sizeof(spy::OptixGeometry<CONFIG>)));
        spy_capture_cuda_check(cudaMallocHost(&capture->host_geometry, sizeof(spy::OptixGeometry<CONFIG>)));
        spy_capture_cuda_check(cudaStreamCreateWithFlags(&capture->copy_stream, cudaStreamNonBlocking));
        spy_capture_cuda_check(cudaEventCreateWithFlags(&capture->packed, cudaEventDisableTiming));
        spy_capture_cuda_check(cudaEventCreateWithFlags(&capture->ready, cudaEventDisableTiming));
    }

    template <typename DEVICE, typename CONFIG>
    void free_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, rendering::raytracing::backends::Optix) {
        auto* capture = state.buffer.capture.state;
        if(capture != nullptr) {
            if(capture->capturing) {
                cudaStreamSynchronize(capture->producer_stream);
                cudaStreamSynchronize(capture->copy_stream);
            }
            if(capture->packed != nullptr) {
                cudaEventDestroy(capture->packed);
            }
            if(capture->ready != nullptr) {
                cudaEventDestroy(capture->ready);
            }
            if(capture->copy_stream != nullptr) {
                cudaStreamDestroy(capture->copy_stream);
            }
            cudaFree(capture->device_atlas);
            cudaFree(capture->device_geometry);
            cudaFreeHost(capture->host_geometry);
            delete capture;
            state.buffer.capture.state = nullptr;
        }
        cudaFreeHost(state.buffer.atlas);
        state.buffer.atlas = nullptr;
    }

    template <typename DEVICE, typename CONFIG>
    void capture_spy_geometry(DEVICE& device, rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& env, cudaStream_t producer_stream) {
        using WORLD = typename CONFIG::WORLD;
        using GEOMETRY = rl::environments::hyperdrone::spy::OptixGeometry<CONFIG>;
        auto& capture = *state.buffer.capture.state;
        capture.producer_stream = producer_stream;
        capture.capturing = true;
        if(state.buffer.selected_valid) {
            constexpr size_t CAMERAS_PER_MEMBER = WORLD::INSTANCES * WORLD::N_VIEWS;
            const size_t member_i = state.buffer.selected_camera / CAMERAS_PER_MEMBER;
            const size_t local_camera = state.buffer.selected_camera % CAMERAS_PER_MEMBER;
            const size_t instance_i = local_camera / WORLD::N_VIEWS;
            auto& world = env.worlds[member_i];
            auto& renderer = world.slots[world.active_slot].renderer;
            auto* geometry = static_cast<GEOMETRY*>(capture.device_geometry);
            spy_capture_cuda_check(cudaMemcpyAsync(&geometry->camera, data(cameras(device, renderer)) + local_camera, sizeof(geometry->camera), cudaMemcpyDeviceToDevice, producer_stream));
            if constexpr(WORLD::RENDERER_SPEC::ENABLE_OVERLAYS) {
                spy_capture_cuda_check(cudaMemcpyAsync(geometry->structure, overlay_structure(device, renderer) + instance_i * GEOMETRY::SLOTS, sizeof(geometry->structure), cudaMemcpyDeviceToDevice, producer_stream));
                spy_capture_cuda_check(cudaMemcpyAsync(geometry->transforms, data(transforms(device, renderer)) + instance_i * GEOMETRY::SLOTS * 12, sizeof(geometry->transforms), cudaMemcpyDeviceToDevice, producer_stream));
            }
        }
    }

    template <typename DEVICE, typename CONFIG>
    void transfer_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, bool transfer_atlas) {
        auto& capture = *state.buffer.capture.state;
        spy_capture_cuda_check(cudaEventRecord(capture.packed, capture.producer_stream));
        spy_capture_cuda_check(cudaStreamWaitEvent(capture.copy_stream, capture.packed, 0));
        if(transfer_atlas) {
            const size_t atlas_bytes = CONFIG::ATLAS_WIDTH * CONFIG::ATLAS_HEIGHT * sizeof(uint32_t);
            spy_capture_cuda_check(cudaMemcpyAsync(state.buffer.atlas, capture.device_atlas, atlas_bytes, cudaMemcpyDeviceToHost, capture.copy_stream));
        }
        if(state.buffer.selected_valid) {
            spy_capture_cuda_check(cudaMemcpyAsync(capture.host_geometry, capture.device_geometry, sizeof(rl::environments::hyperdrone::spy::OptixGeometry<CONFIG>), cudaMemcpyDeviceToHost, capture.copy_stream));
        }
        spy_capture_cuda_check(cudaEventRecord(capture.ready, capture.copy_stream));
        state.buffer.pending = true;
    }

    template <typename DEVICE, typename CONFIG>
    bool poll_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, rendering::raytracing::backends::Optix) {
        auto& capture = *state.buffer.capture.state;
        const auto status = cudaEventQuery(capture.ready);
        if(status == cudaErrorNotReady) {
            return false;
        }
        spy_capture_cuda_check(status);
        if(state.buffer.selected_valid) {
            const auto& geometry = *static_cast<const rl::environments::hyperdrone::spy::OptixGeometry<CONFIG>*>(capture.host_geometry);
            state.buffer.geometry.camera = geometry.camera;
            if constexpr(CONFIG::WORLD::RENDERER_SPEC::ENABLE_OVERLAYS) {
                for(size_t slot_i = 0; slot_i < geometry.SLOTS; slot_i++) {
                    auto& destination = state.buffer.geometry.slots[slot_i];
                    destination.active = geometry.structure[slot_i].active != 0;
                    destination.object = geometry.structure[slot_i].object;
                    std::memcpy(destination.transform, geometry.transforms[slot_i], sizeof(destination.transform));
                }
            }
        }
        capture.capturing = false;
        return true;
    }

    template <typename DEV_SPEC, typename CONFIG>
    void capture_spy(devices::CPU<DEV_SPEC>& device, rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& env, rendering::raytracing::backends::Optix) {
        pack_spy_atlas_cpu(device, state, env);
        constexpr size_t CAMERAS_PER_MEMBER = CONFIG::WORLD::INSTANCES * CONFIG::WORLD::N_VIEWS;
        const size_t member_i = state.buffer.selected_camera / CAMERAS_PER_MEMBER;
        auto& world = env.worlds[member_i];
        auto& renderer = world.slots[world.active_slot].renderer;
        capture_spy_geometry(device, state, env, stream(device, renderer));
        transfer_spy_capture(device, state, false);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
