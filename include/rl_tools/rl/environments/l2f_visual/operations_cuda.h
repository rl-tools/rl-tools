#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_CUDA_H

#include "multirotor_visual.h"
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::cuda{
    __global__ void pixel_to_float_kernel(const uint32_t* __restrict__ frame_buffer, float* __restrict__ output, int num_cameras, int pixels_per_camera){
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = num_cameras * pixels_per_camera;
        if(global_idx >= total_pixels){
            return;
        }
        int cam_idx = global_idx / pixels_per_camera;
        int pixel_idx = global_idx % pixels_per_camera;
        uint32_t rgba = frame_buffer[global_idx];
        float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
        float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
        float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;
        int out_base = cam_idx * (pixels_per_camera * 3) + pixel_idx * 3;
        output[out_base + 0] = r;
        output[out_base + 1] = g;
        output[out_base + 2] = b;
    }
}
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void observe_batch_render_gpu(DEVICE& device, rl::environments::l2f_visual::MultirrotorVisual<SPEC>& env, const CameraData* cameras, typename SPEC::TI num_cameras, float* gpu_output_ptr){
        using TI = typename SPEC::TI;
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        set_cameras_async(device, *env.renderer, cameras, num_cameras);
        render_rgb_only(device, *env.renderer);
        const uint32_t* fb_ptr = (const uint32_t*)owlBufferGetPointer((OWLBuffer)env.renderer->frame_buffer, 0);
        TI total_pixels = num_cameras * CAM_PIXELS;
        int block_size = 256;
        int grid_size = (total_pixels + block_size - 1) / block_size;
        rl::environments::l2f_visual::cuda::pixel_to_float_kernel<<<grid_size, block_size>>>(fb_ptr, gpu_output_ptr, num_cameras, CAM_PIXELS);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
