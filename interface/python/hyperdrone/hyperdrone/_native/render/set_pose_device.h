#pragma once
// the device form of set_pose for the OptiX module: one kernel over a batch of placements
#include "renderer_config.h"
#include <cuda_runtime.h>

namespace HYPERDRONE_MODULE_NAME {
    // one placement per row b: placements[b] = {overlay, first_slot, num_parts, first_part},
    // bases[b][12], scales[b] (nullptr: scale) and coordinates[b][num_coordinates] (nullptr: the
    // rest state); the close set is the shutter-close state when pair is set
    struct PoseBatch {
        const size_t* placements = nullptr;
        const float* bases = nullptr;
        const float* scales = nullptr;
        float scale = 1;
        const float* coordinates = nullptr;
        const float* bases_close = nullptr;
        const float* scales_close = nullptr;
        float scale_close = 1;
        const float* coordinates_close = nullptr;
        size_t count = 0;
        size_t num_coordinates = 0;
        bool pair = false;
    };
    void set_pose_device(const rrt::PoseSink<RENDERER_SPEC>& sink, const PoseBatch& batch, cudaStream_t stream);
}
