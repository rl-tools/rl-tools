#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_CUDA_H

#include "capture_optix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::spy {
    template <typename CONFIG>
    __global__ void pack_spy_atlas_kernel(const float* history, uint32_t* atlas, size_t member, size_t columns, size_t rows, size_t first_y, size_t pixels) {
        using WORLD = typename CONFIG::WORLD;
        constexpr size_t CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
        constexpr size_t WIDTH = CONFIG::ATLAS_WIDTH, HEIGHT = CONFIG::ATLAS_HEIGHT;
        const size_t pixel_i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(pixel_i < pixels) {
            const size_t x = pixel_i % WIDTH, y = first_y + pixel_i / WIDTH;
            const size_t column = x * columns / WIDTH, row = y * rows / HEIGHT;
            const size_t camera = row * columns + column;
            if(camera >= member * CAMERAS && camera < (member + 1) * CAMERAS) {
                uint32_t pixel = 0xff202020;
                if(history != nullptr) {
                    const size_t x0 = (column * WIDTH + columns - 1) / columns, x1 = ((column + 1) * WIDTH + columns - 1) / columns;
                    const size_t y0 = (row * HEIGHT + rows - 1) / rows, y1 = ((row + 1) * HEIGHT + rows - 1) / rows;
                    const size_t source_x = (x - x0) * WORLD::RENDERER_CONFIG::CAM_WIDTH / (x1 - x0);
                    const size_t source_y = (y - y0) * WORLD::RENDERER_CONFIG::CAM_HEIGHT / (y1 - y0);
                    const float* rgb = history + (camera - member * CAMERAS) * WORLD::FRAME_DIM + (source_y * WORLD::RENDERER_CONFIG::CAM_WIDTH + source_x) * WORLD::IMAGE_CHANNELS;
                    pixel = 0xff000000;
                    for(size_t channel = 0; channel < 3; channel++) {
                        float value = rgb[channel] < 1.0f ? rgb[channel] : 1.0f;
                        value = value > 0.0f ? value : 0.0f;
                        pixel |= static_cast<uint32_t>(__fadd_rn(__fmul_rn(value, 255.0f), 0.5f)) << (channel * 8);
                    }
                }
                atlas[y * WIDTH + x] = pixel;
            }
        }
    }
}

namespace rl_tools {
    template <typename DEV_SPEC, typename CONFIG>
    void capture_spy(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& env, rendering::raytracing::backends::Optix) {
        using WORLD = typename CONFIG::WORLD;
        constexpr size_t CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
        constexpr size_t WIDTH = CONFIG::ATLAS_WIDTH, HEIGHT = CONFIG::ATLAS_HEIGHT;
        auto& capture = *state.buffer.capture.state;
        capture.producer_stream = device.stream;
        capture.capturing = true;
        for(size_t member = 0; member < CONFIG::ENVIRONMENT::NUMBER_OF_ENVIRONMENTS; member++) {
            const auto& world = env.environments[member];
            const float* history = state.buffer.frames[member].valid ? data(world.history) + ((world.history_step - 1) % get<0>(typename WORLD::HISTORY_SPEC::SHAPE{})) * CAMERAS * WORLD::FRAME_DIM : nullptr;
            const size_t first_row = member * CAMERAS / state.columns, last_row = ((member + 1) * CAMERAS - 1) / state.columns;
            const size_t y0 = (first_row * HEIGHT + state.rows - 1) / state.rows, y1 = ((last_row + 1) * HEIGHT + state.rows - 1) / state.rows;
            const size_t pixels = WIDTH * (y1 - y0);
            if(pixels > 0) {
                constexpr size_t BLOCK_SIZE = 256;
                rl::environments::hyperdrone::spy::pack_spy_atlas_kernel<CONFIG><<<(pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, device.stream>>>(history, capture.device_atlas, member, state.columns, state.rows, y0, pixels);
                spy_capture_cuda_check(cudaGetLastError());
            }
        }
        capture_spy_geometry(device, state, env, device.stream);
        transfer_spy_capture(device, state, true);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
