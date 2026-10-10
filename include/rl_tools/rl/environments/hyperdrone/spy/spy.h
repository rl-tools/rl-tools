#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPY_H

#include "../world.h"
#include "spectator.h"
#include "ui.h"
#include <array>
#include <thread>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {
    namespace spy {
        template <typename T_ENVIRONMENT>
        struct Config {
            using ENVIRONMENT = T_ENVIRONMENT;
            using WORLD = typename ENVIRONMENT::ENVIRONMENT;
            using T = typename WORLD::T;
            using TI = typename WORLD::TI;
            using BACKEND = typename WORLD::RENDERER::BACKEND;
            using SPECTATOR_CONFIG = SpectatorConfig<typename WORLD::RENDERER_CONFIG>;
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
        struct CaptureStorage<rendering::raytracing::backends::Optix> {
            struct State;
            State* state = nullptr;
        };

        template <typename CONFIG>
        struct Geometry {
            static constexpr size_t SLOTS = CONFIG::WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES;
            rendering::raytracing::Camera<typename CONFIG::T> camera{};
            OverlaySlot slots[SLOTS > 0 ? SLOTS : 1]{};
        };

        template <typename CONFIG>
        struct Buffer {
            uint32_t* atlas = nullptr;
            Geometry<CONFIG> geometry{};
            CaptureStorage<typename CONFIG::BACKEND> capture;
            std::array<Frame, CONFIG::ENVIRONMENT::NUMBER_OF_ENVIRONMENTS> frames{};
            SpectatorJob<typename CONFIG::T> job;
            SpectatorImage image;
            size_t selected_camera = 0;
            bool selected_valid = false;
            bool pending = false;
        };

        template <typename CONFIG>
        struct State {
            ui::Window* window = nullptr;
            typename CONFIG::ENVIRONMENT* environment = nullptr;
            Buffer<CONFIG> buffer;
            Spectator<typename CONFIG::SPECTATOR_CONFIG> spectator;
            std::vector<View<typename CONFIG::T>> views;
            std::array<Frame, CONFIG::ENVIRONMENT::NUMBER_OF_ENVIRONMENTS> displayed_frames{};
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
