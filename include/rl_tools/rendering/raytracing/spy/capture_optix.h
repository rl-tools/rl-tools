#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SPY_CAPTURE_OPTIX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SPY_CAPTURE_OPTIX_H

#include "../backends/optix/operations_cuda.h"
#include "operations_cuda_spectator.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC, typename SNAPSHOT_SPEC>
    void copy_spy_overlay(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, rendering::raytracing::OverlayIndex index, rendering::raytracing::OverlaySnapshot<SNAPSHOT_SPEC>& snapshot) {
        devices::CPU<rendering::raytracing::spy::SpectatorCudaDeviceSpecification> host;
        host.stream = stream(device, renderer);
        copy(host, renderer, index, snapshot);
        spy_spectator_cuda_check(cudaStreamSynchronize(host.stream));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
