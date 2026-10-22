#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_SPY_CAPTURE_OPTIX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_SPY_CAPTURE_OPTIX_H

#include "../backends/optix/operations_cuda.h"
#include "operations_cuda_spectator.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC, typename SNAPSHOT_SPEC>
    void copy_spy_overlay(DEVICE& device, HyperLux<SPEC, hyperlux::backends::Optix>& renderer, hyperlux::OverlayIndex index, hyperlux::OverlaySnapshot<SNAPSHOT_SPEC>& snapshot) {
        devices::CPU<hyperlux::spy::SpectatorCudaDeviceSpecification> host;
        host.stream = stream(device, renderer);
        copy(host, renderer, index, snapshot);
        spy_spectator_cuda_check(cudaStreamSynchronize(host.stream));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
