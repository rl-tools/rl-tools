#include "ui_frontends.h"

#include <memory>
#include <stdexcept>
#include <thread>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::spy::ui {
    struct Window {
        std::thread::id owner = std::this_thread::get_id();
#ifdef RL_TOOLS_HYPERLUX_SPY_ENABLE_KITTY
        kitty::Window* kitty = nullptr;
#endif
#ifdef RL_TOOLS_HYPERLUX_SPY_ENABLE_GLFW
        glfw::Window* glfw = nullptr;
#endif
    };

    namespace {
        Window* active_window = nullptr;

        // calls the frontend the window was created with; the generic call is never instantiated for frontends this build lacks
        template <typename CALL>
        auto dispatch(Window* window, CALL&& call) {
            if(window == nullptr || window->owner != std::this_thread::get_id()) {
                throw std::runtime_error("HyperLux spy must be updated and closed on its creating thread");
            }
#if defined(RL_TOOLS_HYPERLUX_SPY_ENABLE_KITTY) && defined(RL_TOOLS_HYPERLUX_SPY_ENABLE_GLFW)
            return window->kitty != nullptr ? call(window->kitty) : call(window->glfw);
#elif defined(RL_TOOLS_HYPERLUX_SPY_ENABLE_KITTY)
            return call(window->kitty);
#else
            return call(window->glfw);
#endif
        }
    }

    Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height, Frontend frontend) {
        if(active_window != nullptr) {
            throw std::runtime_error("Only one HyperLux spy window can be open per process");
        }
        if(title == nullptr || spectator_width <= 0 || spectator_height <= 0 || (!labels.empty() && (columns <= 0 || rows <= 0 || atlas_width < columns || atlas_height < rows || labels.size() > static_cast<std::size_t>(columns) * rows))) {
            throw std::invalid_argument("HyperLux spy received invalid title or image dimensions");
        }
        auto window = std::make_unique<Window>();
        bool created = false;
#ifdef RL_TOOLS_HYPERLUX_SPY_ENABLE_GLFW
        if(frontend == Frontend::WINDOW) {
            window->glfw = glfw::create(title, labels, atlas_width, atlas_height, columns, rows, spectator_width, spectator_height);
            created = true;
        }
#endif
#ifdef RL_TOOLS_HYPERLUX_SPY_ENABLE_KITTY
        if(!created) {
            window->kitty = kitty::create(title, labels, atlas_width, atlas_height, columns, rows, spectator_width, spectator_height);
            created = true;
        }
#endif
        if(!created) {
            throw std::runtime_error("HyperLux spy: this build has no kitty terminal frontend (it needs zlib on a POSIX system)");
        }
        active_window = window.get();
        return window.release();
    }

    Input update(Window* window, double atlas_age, double spectator_age, const char* status) {
        return dispatch(window, [&](auto* frontend) { return update(frontend, atlas_age, spectator_age, status); });
    }

    void upload_atlas(Window* window, const std::uint32_t* pixels) {
        dispatch(window, [&](auto* frontend) { upload_atlas(frontend, pixels); });
    }

    void upload_spectator(Window* window, const std::uint32_t* pixels) {
        dispatch(window, [&](auto* frontend) { upload_spectator(frontend, pixels); });
    }

    void destroy(Window* window) {
        if(window == nullptr) {
            return;
        }
        dispatch(window, [](auto* frontend) { destroy(frontend); });
        active_window = nullptr;
        delete window;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
