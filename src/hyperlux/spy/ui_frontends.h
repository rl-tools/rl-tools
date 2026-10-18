#pragma once

#include <rl_tools/hyperlux/spy/ui.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::spy::ui {
    namespace glfw {
        struct Window;
        Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height);
        Input update(Window* window, double atlas_age, double spectator_age, const char* status);
        void upload_atlas(Window* window, const std::uint32_t* pixels);
        void upload_spectator(Window* window, const std::uint32_t* pixels);
        void destroy(Window* window);
    }
    namespace kitty {
        struct Window;
        Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height);
        Input update(Window* window, double atlas_age, double spectator_age, const char* status);
        void upload_atlas(Window* window, const std::uint32_t* pixels);
        void upload_spectator(Window* window, const std::uint32_t* pixels);
        void destroy(Window* window);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
