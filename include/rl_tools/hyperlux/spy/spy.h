#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_SPY_SPY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_SPY_SPY_H

#include "spectator.h"
#include "ui.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux {
    namespace spy {
        template <typename T_RENDERER>
        struct Config {
            using RENDERER = T_RENDERER;
            using T = typename RENDERER::SPEC::T;
            using TI = typename RENDERER::SPEC::TI;
            using SPECTATOR_CONFIG = SpectatorConfig<typename RENDERER::SPEC::CONFIG>;
            static constexpr double REFRESH_INTERVAL = 1.0 / 20.0;
            static constexpr double UI_REFRESH_INTERVAL = 1.0 / 60.0;
            static constexpr T MOVE_SPEED = 3;
        };

        template <typename CONFIG>
        struct Buffer {
            Camera<typename CONFIG::T> camera{};
            SpectatorJob<Specification<typename CONFIG::SPECTATOR_CONFIG>> job;
            SpectatorImage image;
        };

        template <typename CONFIG>
        struct State {
            ui::Window* window = nullptr;
            typename CONFIG::RENDERER* renderer = nullptr;
            const Scene* scene = nullptr;
            const AssetPool* pool = nullptr;
            hyperlux::SceneMetadata<typename CONFIG::T> metadata;
            Buffer<CONFIG> buffer;
            Spectator<typename CONFIG::SPECTATOR_CONFIG> spectator;
            typename CONFIG::T position[3] = {};
            typename CONFIG::T yaw = 0;
            typename CONFIG::T pitch = 0;
            std::thread::id owner;
            size_t camera = 0;
            double next_capture = 0;
            double last_update = 0;
            double captured_at = 0;
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
