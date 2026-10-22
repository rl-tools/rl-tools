#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPY_H

#include "../world.h"
#include "../../../../hyperlux/spy/spectator.h"
#include "../../../../hyperlux/spy/ui.h"
#include <array>
#include <thread>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {
    namespace spy {
        template <typename T_ENVIRONMENT>
        struct Config {
            using ENVIRONMENT = T_ENVIRONMENT;
            using WORLD = typename ENVIRONMENT::WORLD;
            using RGB_FLOAT = hyperlux::outputs::RGB<hyperlux::outputs::format::Float32>;
            using RGB_PACKED = hyperlux::outputs::RGB<>;
            using FLOAT_ROWS = world::OutputRows<typename WORLD::RENDERER_SPEC, RGB_FLOAT, WORLD::N_CAMERAS_PER_AGENT>;
            using PACKED_ROWS = world::OutputRows<typename WORLD::RENDERER_SPEC, RGB_PACKED, WORLD::N_CAMERAS_PER_AGENT>;
            using T = typename WORLD::T;
            using TI = typename WORLD::TI;
            using BACKEND = typename WORLD::RENDERER::BACKEND;
            struct SPECTATOR_CONFIG: hyperlux::spy::SpectatorConfig<typename WORLD::RENDERER_CONFIG> {
                static constexpr TI NUM_OVERLAYS = WORLD::RENDERER_SPEC::ENABLE_OVERLAYS ? 1 : 0;
                static constexpr TI MAX_OVERLAYS_PER_CAMERA = NUM_OVERLAYS;
            };
            static constexpr TI ATLAS_WIDTH = 1024;
            static constexpr TI ATLAS_HEIGHT = 1024;
            static constexpr double REFRESH_INTERVAL = 1.0 / 20.0;
            static constexpr double UI_REFRESH_INTERVAL = 1.0 / 60.0;
            static constexpr T MOVE_SPEED = 3;
        };

        struct Frame {
            size_t scene_version = 0;
            size_t sequence = 0;
            double captured_at = 0;
            bool valid = false;
        };

        template <typename T>
        struct View {
            T position[3] = {};
            T yaw = 0;
            T pitch = 0;
            size_t scene_version = 0;
            bool initialized = false;
        };

        template <typename BACKEND>
        struct CaptureStorage {};

        template <>
        struct CaptureStorage<hyperlux::backends::Optix> {
            struct State;
            State* state = nullptr;
        };

        // the selected camera and its instance's overlay, captured from the training renderer in
        // the spectator's snapshot type (the capacities agree by SpectatorConfig)
        template <typename CONFIG>
        struct Geometry {
            using SPECTATOR_SPEC = hyperlux::Specification<typename CONFIG::SPECTATOR_CONFIG>;
            static_assert(SPECTATOR_SPEC::MAX_OVERLAY_INSTANCES == CONFIG::WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES);
            hyperlux::Camera<typename CONFIG::T> camera{};
            hyperlux::OverlaySnapshot<SPECTATOR_SPEC> overlay{};
        };

        template <typename CONFIG>
        struct Buffer {
            uint32_t* atlas = nullptr;
            std::vector<float> rgb_float;
            std::vector<uint32_t> rgb_packed;
            Geometry<CONFIG> geometry{};
            CaptureStorage<typename CONFIG::BACKEND> capture;
            std::array<Frame, CONFIG::ENVIRONMENT::N_WORLDS> frames{};
            hyperlux::spy::SpectatorJob<hyperlux::Specification<typename CONFIG::SPECTATOR_CONFIG>> job;
            hyperlux::spy::SpectatorImage image;
            size_t selected_camera = 0;
            bool selected_valid = false;
            bool pending = false;
        };

        template <typename CONFIG>
        struct State {
            hyperlux::spy::ui::Window* window = nullptr;
            typename CONFIG::ENVIRONMENT* environment = nullptr;
            Buffer<CONFIG> buffer;
            hyperlux::spy::Spectator<typename CONFIG::SPECTATOR_CONFIG> spectator;
            std::vector<View<typename CONFIG::T>> views;
            std::array<Frame, CONFIG::ENVIRONMENT::N_WORLDS> displayed_frames{};
            std::thread::id owner;
            size_t selected_camera = 0;
            size_t columns = 1;
            size_t rows = 1;
            double next_capture = 0;
            double last_update = 0;
            double spectator_captured_at = 0;
            bool reset_camera = true;
            bool open = false;
        };
    }

    template <typename T_CONFIG>
    struct Spy {
        using CONFIG = T_CONFIG;
        spy::State<CONFIG>* state = nullptr;
        Spy() = default;
        Spy(const Spy&) = delete;
        Spy& operator=(const Spy&) = delete;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
