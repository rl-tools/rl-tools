#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CUDA_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SPY_OPERATIONS_CUDA_SPECTATOR_H

#include "spectator.h"
#include "../backends/optix/operations_cuda.h"

#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::spy {
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
        uint32_t* pixels = nullptr;
    };
}

namespace rl_tools {
    inline void spy_spectator_cuda_check(cudaError_t status) {
        if(status != cudaSuccess) {
            throw std::runtime_error(std::string("Raytracing spy spectator: ") + cudaGetErrorString(status));
        }
    }

    template <typename DEVICE>
    int spy_spectator_device_id(DEVICE&, rendering::raytracing::backends::Optix) {
        int device_id = 0;
        spy_spectator_cuda_check(cudaGetDevice(&device_id));
        return device_id;
    }

    template <typename DEVICE, typename CONFIG>
    void malloc(DEVICE& device, rendering::raytracing::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer, int device_id) {
        using SPEC = typename rendering::raytracing::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>::SPEC;
        spy_spectator_cuda_check(cudaSetDevice(device_id));
        spy_spectator_cuda_check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
        spy_spectator_cuda_check(cudaMallocHost(&buffer.camera, sizeof(*buffer.camera)));
        spy_spectator_cuda_check(cudaMallocHost(&buffer.pixels, SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT * sizeof(*buffer.pixels)));
    }

    template <typename DEVICE, typename CONFIG>
    void free(DEVICE& device, rendering::raytracing::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer) {
        if(device.stream != nullptr) {
            cudaStreamSynchronize(device.stream);
        }
        if(buffer.renderer.backend != nullptr) {
            free(device, buffer.renderer);
        }
        cudaFreeHost(buffer.camera);
        cudaFreeHost(buffer.pixels);
        if(device.stream != nullptr) {
            cudaStreamDestroy(device.stream);
            device.stream = nullptr;
        }
        buffer.camera = nullptr;
        buffer.pixels = nullptr;
    }

    template <typename DEVICE, typename CONFIG>
    void render_spy_spectator(DEVICE& device, rendering::raytracing::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>& buffer, const rendering::raytracing::spy::SpectatorJob<rendering::raytracing::Specification<CONFIG>>& job, rendering::raytracing::spy::SpectatorImage& result) {
        using SPEC = typename rendering::raytracing::spy::SpectatorBuffer<CONFIG, rendering::raytracing::backends::Optix>::SPEC;
        auto& renderer = buffer.renderer;
        *buffer.camera = job.camera;
        spy_spectator_cuda_check(cudaMemcpyAsync(data(cameras(device, renderer)), buffer.camera, sizeof(*buffer.camera), cudaMemcpyHostToDevice, device.stream));
        copy_spy_overlays(device, renderer, job);
        if constexpr(SPEC::ENABLE_OVERLAYS) {
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
