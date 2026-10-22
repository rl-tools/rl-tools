#include <rl_tools/rl_tools.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_SPY_UI_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_SPY_UI_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::spy::ui {
    struct Window;
    struct Input {
        std::size_t camera = 0;
        float forward = 0;
        float right = 0;
        float up = 0;
        float yaw_delta = 0;
        float pitch_delta = 0;
        bool reset_camera = false;
        bool closed = false;
    };

    Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height);
    Input update(Window* window, double atlas_age, double spectator_age, const char* status);
    void upload_atlas(Window* window, const std::uint32_t* pixels);
    void upload_spectator(Window* window, const std::uint32_t* pixels);
    void destroy(Window* window);
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
