#include "ui_frontends.h"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/utils/kitty/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::spy::ui::kitty {
    namespace kitty_protocol = rl_tools::utils::kitty;
    constexpr unsigned int TEXT_ROWS = 2;
    constexpr unsigned int LOG_ROWS = 8;
    constexpr float ATLAS_SHARE = 0.45f;
    constexpr float LOOK_SENSITIVITY = 0.002f;
    constexpr float FAST_SPEED = 3.0f;
    constexpr std::uint32_t BACKGROUND_COLOR = 0xff171717u;
    constexpr std::uint32_t SELECTION_COLOR = 0xff3ec2ffu;

    struct Window {
        devices::DefaultCPU device;
        kitty_protocol::Terminal terminal;
        kitty_protocol::Image atlas, spectator;
        kitty_protocol::Placement atlas_placement, spectator_placement;
        std::vector<std::uint32_t> atlas_pixels, highlighted_atlas, spectator_pixels;
        std::string title;
        std::vector<std::string> labels;
        std::size_t selected = 0;
        int atlas_width = 0, atlas_height = 0, columns = 0, rows = 0, spectator_width = 0, spectator_height = 0;
        bool atlas_pending = false, spectator_pending = false, spectator_visible = false;
        bool dragging = false;
        int mouse_x = 0, mouse_y = 0;
        std::string lines[TEXT_ROWS];
    };

    namespace {
        void layout(Window* window) {
            const kitty_protocol::Terminal& terminal = window->terminal;
            kitty_protocol::Placement area;
            area.row = TEXT_ROWS;
            area.rows = terminal.rows > TEXT_ROWS + terminal.log_rows ? terminal.rows - TEXT_ROWS - terminal.log_rows : 0;
            if(!window->labels.empty()) {
                area.columns = static_cast<unsigned int>(terminal.columns * ATLAS_SHARE);
                window->atlas_placement = fit(window->device, terminal, window->atlas_width, window->atlas_height, area);
                area.column = area.columns + 1;
            }
            area.columns = terminal.columns > area.column ? terminal.columns - area.column : 0;
            window->spectator_placement = fit(window->device, terminal, window->spectator_width, window->spectator_height, area);
            for (std::string& line : window->lines) {
                line.clear();
            }
        }

        void highlight_selection(Window* window) {
            window->highlighted_atlas = window->atlas_pixels;
            const float tile_width = static_cast<float>(window->atlas_width) / window->columns;
            const float tile_height = static_cast<float>(window->atlas_height) / window->rows;
            const float column = static_cast<float>(window->selected % window->columns);
            const float row = static_cast<float>(window->selected / window->columns);
            const int x0 = static_cast<int>(std::lround(column * tile_width)), x1 = static_cast<int>(std::lround((column + 1) * tile_width));
            const int y0 = static_cast<int>(std::lround(row * tile_height)), y1 = static_cast<int>(std::lround((row + 1) * tile_height));
            const float displayed_width = window->atlas_placement.columns * window->terminal.cell_width;
            const int thickness = std::max(1, static_cast<int>(std::lround(2 * window->atlas_width / std::max(1.0f, displayed_width))));
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    if (x - x0 < thickness || x1 - 1 - x < thickness || y - y0 < thickness || y1 - 1 - y < thickness) {
                        window->highlighted_atlas[static_cast<std::size_t>(y) * window->atlas_width + x] = SELECTION_COLOR;
                    }
                }
            }
        }

        void flush(Window* window) {
            if (window->atlas_pending && ready(window->device, window->terminal, window->atlas)) {
                highlight_selection(window);
                window->atlas_pending = !transmit(window->device, window->terminal, window->atlas, window->highlighted_atlas.data(), window->atlas_width, window->atlas_height, window->atlas_placement);
            }
            if (window->spectator_pending && ready(window->device, window->terminal, window->spectator)) {
                window->spectator_pending = !transmit(window->device, window->terminal, window->spectator, window->spectator_pixels.data(), window->spectator_width, window->spectator_height, window->spectator_placement);
                window->spectator_visible = window->spectator_visible || !window->spectator_pending;
            }
        }

        std::string frame_age(const char* label, double age) {
            char buffer[64];
            if (std::isfinite(age) && age >= 0) {
                std::snprintf(buffer, sizeof(buffer), "%s: %.0f ms old", label, age * 1000);
            } else {
                std::snprintf(buffer, sizeof(buffer), "%s: waiting for a frame", label);
            }
            return buffer;
        }

        void show_line(Window* window, unsigned int row, const std::string& line) {
            if (window->lines[row] != line) {
                window->lines[row] = line;
                text(window->device, window->terminal, row, 0, line);
            }
        }
    }

    Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height) {
        auto window = std::make_unique<Window>();
        window->title = title;
        window->labels = labels;
        window->atlas_width = atlas_width;
        window->atlas_height = atlas_height;
        window->columns = columns;
        window->rows = rows;
        window->spectator_width = spectator_width;
        window->spectator_height = spectator_height;
        if(!labels.empty()) {
            window->atlas_pixels.assign(static_cast<std::size_t>(atlas_width) * atlas_height, BACKGROUND_COLOR);
        }
        window->spectator_pixels.assign(static_cast<std::size_t>(spectator_width) * spectator_height, BACKGROUND_COLOR);
        window->terminal.log_rows = LOG_ROWS;
        if (!init(window->device, window->terminal)) {
            throw std::runtime_error("HyperLux spy could not take over the kitty terminal");
        }
        if(!labels.empty()) {
            init(window->device, window->terminal, window->atlas);
        }
        init(window->device, window->terminal, window->spectator);
        layout(window.get());
        window->atlas_pending = !labels.empty();
        flush(window.get());
        return window.release();
    }

    Input update(Window* window, double atlas_age, double spectator_age, const char* status) {
        auto& device = window->device;
        auto& terminal = window->terminal;
        poll(device, terminal, 0);
        if (terminal.resized) {
            layout(window);
            if(!window->labels.empty()) {
                place(device, terminal, window->atlas, window->atlas_placement);
            }
            if (window->spectator_visible) {
                place(device, terminal, window->spectator, window->spectator_placement);
            }
        }
        Input input;
        const std::size_t previous_selection = window->selected;
        for (const kitty_protocol::Event& event : terminal.events) {
            switch (event.type) {
                case kitty_protocol::EventType::KEY_PRESS:
                    if (event.code == 'c' && (event.modifiers & kitty_protocol::modifiers::CONTROL)) {
                        // the terminal sends Ctrl+C as a key while the spy owns it; the host program still gets its interrupt
                        std::raise(SIGINT);
                    } else if (event.code == kitty_protocol::keys::ESCAPE) {
                        input.closed = true;
                    } else if (event.code == 'r') {
                        input.reset_camera = true;
                    }
                    break;
                case kitty_protocol::EventType::MOUSE_PRESS: {
                    if (event.code != kitty_protocol::buttons::LEFT) {
                        break;
                    }
                    float u, v;
                    if (!window->labels.empty() && locate(device, terminal, window->atlas_placement, window->atlas_width, window->atlas_height, event.x, event.y, u, v)) {
                        const std::size_t column = std::min(window->columns - 1, static_cast<int>(u * window->columns));
                        const std::size_t row = std::min(window->rows - 1, static_cast<int>(v * window->rows));
                        const std::size_t camera = row * window->columns + column;
                        if (camera < window->labels.size()) {
                            window->selected = camera;
                        }
                    } else if (locate(device, terminal, window->spectator_placement, window->spectator_width, window->spectator_height, event.x, event.y, u, v)) {
                        window->dragging = true;
                        window->mouse_x = event.x;
                        window->mouse_y = event.y;
                    }
                    break;
                }
                case kitty_protocol::EventType::MOUSE_RELEASE:
                    if (event.code == kitty_protocol::buttons::LEFT) {
                        window->dragging = false;
                    }
                    break;
                case kitty_protocol::EventType::MOUSE_DRAG:
                    if (event.code == kitty_protocol::buttons::LEFT && window->dragging) {
                        input.yaw_delta += static_cast<float>(window->mouse_x - event.x) * LOOK_SENSITIVITY;
                        input.pitch_delta += static_cast<float>(window->mouse_y - event.y) * LOOK_SENSITIVITY;
                        window->mouse_x = event.x;
                        window->mouse_y = event.y;
                    }
                    break;
                case kitty_protocol::EventType::FOCUS_LOST:
                    window->dragging = false;
                    break;
                case kitty_protocol::EventType::HANGUP:
                    input.closed = true;
                    break;
                default: break;
            }
        }
        const float speed = held(device, terminal, kitty_protocol::keys::LEFT_SHIFT) ? FAST_SPEED : 1.0f;
        input.forward = speed * (static_cast<float>(held(device, terminal, 'w')) - static_cast<float>(held(device, terminal, 's')));
        input.right = speed * (static_cast<float>(held(device, terminal, 'd')) - static_cast<float>(held(device, terminal, 'a')));
        input.up = speed * (static_cast<float>(held(device, terminal, 'e')) - static_cast<float>(held(device, terminal, 'q')));
        input.camera = window->selected;
        if (input.closed) {
            return input;
        }
        if (window->selected != previous_selection) {
            window->atlas_pending = true;
            spectator_age = -1;
        }
        if (spectator_age < 0 && window->spectator_visible) {
            hide(device, terminal, window->spectator);
            window->spectator_visible = false;
            window->spectator_pending = false;
        }
        show_line(window, 0, window->labels.empty() ? window->title : frame_age("Cameras", atlas_age) + "  |  " + window->labels[window->selected] + "  |  click a camera to select it");
        show_line(window, 1, frame_age("Spectator", spectator_age) + "  |  drag: look  |  WASD: move  |  Q/E: down/up  |  Shift: faster  |  R: reset camera  |  Esc: close" + (status != nullptr && status[0] != '\0' ? std::string("  |  ") + status : std::string()));
        flush(window);
        return input;
    }

    void upload_atlas(Window* window, const std::uint32_t* pixels) {
        std::copy(pixels, pixels + window->atlas_pixels.size(), window->atlas_pixels.begin());
        window->atlas_pending = true;
        flush(window);
    }

    void upload_spectator(Window* window, const std::uint32_t* pixels) {
        std::copy(pixels, pixels + window->spectator_pixels.size(), window->spectator_pixels.begin());
        window->spectator_pending = true;
        flush(window);
    }

    void destroy(Window* window) {
        free(window->device, window->terminal);
        delete window;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
