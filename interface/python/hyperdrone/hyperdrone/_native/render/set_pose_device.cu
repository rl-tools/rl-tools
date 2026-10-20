#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include "set_pose_device.h"

namespace HYPERDRONE_MODULE_NAME {
    template <typename TAG>
    __global__ void set_pose_kernel(TAG device, rrt::PoseSink<RENDERER_SPEC> sink, PoseBatch batch){
        const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if(b >= batch.count){
            return;
        }
        const size_t* row = batch.placements + b * 4;
        const rrt::OverlayIndex overlay{row[0]};
        const rrt::OverlayPlacement placement{row[1], row[2], row[3]};
        const auto pose = rlt::make_pose(batch.bases + b * 12, batch.scales != nullptr ? batch.scales[b] : batch.scale);
        const float* coordinates = batch.coordinates != nullptr ? batch.coordinates + b * batch.num_coordinates : nullptr;
        if constexpr (RENDERER_SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            if(batch.pair){
                const auto close = rlt::make_pose(batch.bases_close + b * 12, batch.scales_close != nullptr ? batch.scales_close[b] : batch.scale_close);
                const float* coordinates_close = batch.coordinates_close != nullptr ? batch.coordinates_close + b * batch.num_coordinates : nullptr;
                rlt::set_pose(device, sink, overlay, placement, pose, close, coordinates, coordinates_close);
                return;
            }
        }
        rlt::set_pose(device, sink, overlay, placement, pose, coordinates);
    }

    void set_pose_device(const rrt::PoseSink<RENDERER_SPEC>& sink, const PoseBatch& batch, cudaStream_t stream){
        if(batch.count == 0){
            return;
        }
        constexpr unsigned int BLOCK_SIZE = 128;
        const unsigned int blocks = (unsigned int)((batch.count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        rlt::devices::cuda::TAG<rlt::devices::DefaultCUDA, true> tag{};
        set_pose_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(tag, sink, batch);
    }
}
