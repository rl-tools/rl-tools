#include "spy.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_CAPTURE_CPU_H

#include <algorithm>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename CONFIG>
    void pack_spy_atlas_cpu(rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& environment) {
        using WORLD = typename CONFIG::WORLD;
        constexpr size_t CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
        constexpr size_t WIDTH = CONFIG::ATLAS_WIDTH, HEIGHT = CONFIG::ATLAS_HEIGHT;
        for(size_t member = 0; member < CONFIG::ENVIRONMENT::NUMBER_OF_ENVIRONMENTS; member++) {
            auto& world = environment.environments[member];
            const bool valid = state.buffer.frames[member].valid;
            const float* history = valid ? data(world.history) + ((world.history_step - 1) % get<0>(typename WORLD::HISTORY_SPEC::SHAPE{})) * CAMERAS * WORLD::FRAME_DIM : nullptr;
            for(size_t camera = 0; camera < CAMERAS; camera++) {
                const size_t index = member * CAMERAS + camera;
                const size_t column = index % state.columns, row = index / state.columns;
                const size_t x0 = (column * WIDTH + state.columns - 1) / state.columns, x1 = ((column + 1) * WIDTH + state.columns - 1) / state.columns;
                const size_t y0 = (row * HEIGHT + state.rows - 1) / state.rows, y1 = ((row + 1) * HEIGHT + state.rows - 1) / state.rows;
                for(size_t y = y0; y < y1; y++) {
                    for(size_t x = x0; x < x1; x++) {
                        uint32_t pixel = 0xff202020;
                        if(valid) {
                            const size_t source_x = (x - x0) * WORLD::RENDERER_CONFIG::CAM_WIDTH / (x1 - x0);
                            const size_t source_y = (y - y0) * WORLD::RENDERER_CONFIG::CAM_HEIGHT / (y1 - y0);
                            const float* rgb = history + camera * WORLD::FRAME_DIM + (source_y * WORLD::RENDERER_CONFIG::CAM_WIDTH + source_x) * WORLD::IMAGE_CHANNELS;
                            pixel = 0xff000000;
                            for(size_t channel = 0; channel < 3; channel++) {
                                const float value = std::max(0.0f, std::min(1.0f, rgb[channel]));
                                pixel |= static_cast<uint32_t>(value * 255.0f + 0.5f) << (channel * 8);
                            }
                        }
                        state.buffer.atlas[y * WIDTH + x] = pixel;
                    }
                }
            }
        }
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void malloc_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, BACKEND) {
        state.buffer.atlas = new uint32_t[CONFIG::ATLAS_WIDTH * CONFIG::ATLAS_HEIGHT];
        std::fill_n(state.buffer.atlas, CONFIG::ATLAS_WIDTH * CONFIG::ATLAS_HEIGHT, 0xff202020);
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void free_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>& state, BACKEND) {
        delete[] state.buffer.atlas;
        state.buffer.atlas = nullptr;
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    bool poll_spy_capture(DEVICE&, rl::environments::hyperdrone::spy::State<CONFIG>&, BACKEND) {
        return true;
    }

    template <typename DEVICE, typename CONFIG, typename BACKEND>
    void capture_spy(DEVICE& device, rl::environments::hyperdrone::spy::State<CONFIG>& state, typename CONFIG::ENVIRONMENT& environment, BACKEND) {
        using WORLD = typename CONFIG::WORLD;
        pack_spy_atlas_cpu(state, environment);
        if(state.buffer.selected_valid) {
            const size_t camera = state.buffer.selected_camera % (WORLD::INSTANCES * WORLD::N_VIEWS);
            const size_t member = state.buffer.selected_camera / (WORLD::INSTANCES * WORLD::N_VIEWS);
            const size_t instance = camera / WORLD::N_VIEWS;
            auto& world = environment.environments[member];
            auto& renderer = world.slots[world.active_slot].renderer;
            state.buffer.geometry.camera = data(cameras(device, renderer))[camera];
            if constexpr(WORLD::RENDERER_SPEC::ENABLE_OVERLAYS) {
                for(size_t slot = 0; slot < WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES; slot++) {
                    auto& destination = state.buffer.geometry.slots[slot];
                    const auto& source = renderer.overlays[instance].slots[slot];
                    destination.object = source.object;
                    destination.active = source.active;
                    const float* transform = data(transforms(device, renderer)) + (instance * WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES + slot) * 12;
                    std::copy_n(transform, 12, destination.transform);
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
