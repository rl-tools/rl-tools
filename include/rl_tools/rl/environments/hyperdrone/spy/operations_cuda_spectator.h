#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CUDA_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_OPERATIONS_CUDA_SPECTATOR_H

#include "spectator.h"
#include "../../../../rendering/raytracing/backends/optix/operations_cuda.h"

#include <cstring>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::spy {
    struct SpectatorCudaDeviceSpecification: devices::DefaultCPUSpecification {
        struct DEVICE_EXTENSION {
            cudaStream_t stream = nullptr;
        };
    };

    template <typename CONFIG>
    struct SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix> {
        using SPEC = rendering::raytracing::Specification<CONFIG>;
        devices::CPU<SpectatorCudaDeviceSpecification> device;
        rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix> renderer;
        rendering::raytracing::Camera<typename CONFIG::T>* camera = nullptr;
        rendering::raytracing::backends::optix::OverlaySlotStructure* structure = nullptr;
        float* transforms = nullptr;
        uint32_t* pixels = nullptr;
    };
}

namespace rl_tools {
    inline void spy_spectator_cuda_check(cudaError_t status) {
        if(status != cudaSuccess) {
            throw std::runtime_error(std::string("HyperDrone spy spectator: ") + cudaGetErrorString(status));
        }
    }

    template <typename DEVICE>
    int spy_spectator_device_id(DEVICE&, rendering::raytracing::backends::Optix) {
        int device_id = 0;
        spy_spectator_cuda_check(cudaGetDevice(&device_id));
        return device_id;
    }

    template <typename DEVICE, typename CONFIG>
    void malloc(DEVICE& device, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer, int device_id) {
        using SPEC = typename rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>::SPEC;
        spy_spectator_cuda_check(cudaSetDevice(device_id));
        spy_spectator_cuda_check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
        spy_spectator_cuda_check(cudaMallocHost(&buffer.camera, sizeof(*buffer.camera)));
        spy_spectator_cuda_check(cudaMallocHost(&buffer.pixels, SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT * sizeof(*buffer.pixels)));
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            spy_spectator_cuda_check(cudaMallocHost(&buffer.transforms, SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(*buffer.transforms)));
            spy_spectator_cuda_check(cudaMallocHost(&buffer.structure, SPEC::MAX_OVERLAY_INSTANCES * sizeof(*buffer.structure)));
        }
    }

    template <typename DEVICE, typename CONFIG>
    void free(DEVICE& device, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer) {
        if(device.stream != nullptr) {
            cudaStreamSynchronize(device.stream);
        }
        if(buffer.renderer.backend != nullptr) {
            free(device, buffer.renderer);
        }
        cudaFreeHost(buffer.camera);
        cudaFreeHost(buffer.pixels);
        cudaFreeHost(buffer.transforms);
        cudaFreeHost(buffer.structure);
        if(device.stream != nullptr) {
            cudaStreamDestroy(device.stream);
            device.stream = nullptr;
        }
        buffer.camera = nullptr;
        buffer.pixels = nullptr;
        buffer.transforms = nullptr;
        buffer.structure = nullptr;
    }

    template <typename DEVICE, typename CONFIG>
    void render_spy_spectator(DEVICE& device, rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer, const rl::environments::hyperdrone::spy::SpectatorJob<typename CONFIG::T>& job, rl::environments::hyperdrone::spy::SpectatorImage& result) {
        using SPEC = typename rl::environments::hyperdrone::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>::SPEC;
        auto& renderer = buffer.renderer;
        *buffer.camera = job.camera;
        spy_spectator_cuda_check(cudaMemcpyAsync(data(cameras(device, renderer)), buffer.camera, sizeof(*buffer.camera), cudaMemcpyHostToDevice, device.stream));
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            for(size_t slot_i = 0; slot_i < SPEC::MAX_OVERLAY_INSTANCES; slot_i++) {
                const auto& source = job.overlay[slot_i];
                buffer.structure[slot_i].active = source.active ? 1u : 0u;
                buffer.structure[slot_i].object = source.object;
                buffer.structure[slot_i].segmentation_class = rendering::segmentation::INHERIT;
                std::memcpy(buffer.transforms + slot_i * 12, source.transform, 12 * sizeof(float));
            }
            spy_spectator_cuda_check(cudaMemcpyAsync(data(renderer.transforms), buffer.transforms, SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float), cudaMemcpyHostToDevice, device.stream));
            spy_spectator_cuda_check(cudaMemcpyAsync(overlay_structure(device, renderer), buffer.structure, SPEC::MAX_OVERLAY_INSTANCES * sizeof(*buffer.structure), cudaMemcpyHostToDevice, device.stream));
            renderer.overlays[0].dirty = false;
            update_launch(device, renderer);
        }
        render_launch(device, renderer);
        const size_t pixels = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        spy_spectator_cuda_check(cudaMemcpyAsync(buffer.pixels, data(output(device, renderer, rl_tools::rendering::raytracing::outputs::RGB<>{})), pixels * sizeof(*buffer.pixels), cudaMemcpyDeviceToHost, device.stream));
        spy_spectator_cuda_check(cudaStreamSynchronize(device.stream));
        result.pixels.assign(buffer.pixels, buffer.pixels + pixels);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
