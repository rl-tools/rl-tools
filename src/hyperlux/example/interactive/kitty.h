#pragma once

#include "input.h"

#include <rl_tools/utils/kitty/operations_cpu.h>

#include <cstdint>

namespace rl_tools::hyperlux::interactive {
    struct Kitty {
        utils::kitty::Terminal terminal;
        utils::kitty::Image image;
        unsigned int width = 0;
        unsigned int height = 0;
        bool dragging = false;
        int mouse_x = 0, mouse_y = 0;
    };
}

namespace rl_tools {
    template <typename DEVICE>
    bool init(DEVICE& device, hyperlux::interactive::Kitty& kitty, unsigned int width, unsigned int height) {
        kitty.width = width;
        kitty.height = height;
        if (!init(device, kitty.terminal)) {
            return false;
        }
        init(device, kitty.terminal, kitty.image);
        return true;
    }

    // waits until the terminal acknowledged the previous frame, so rendering runs at the rate the terminal (or link) sustains
    template <typename DEVICE>
    void poll(DEVICE& device, hyperlux::interactive::Kitty& kitty, hyperlux::interactive::InputState& input) {
        namespace kitty_protocol = utils::kitty;
        do {
            poll(device, kitty.terminal, ready(device, kitty.terminal, kitty.image) ? 0 : -1);
            for (const kitty_protocol::Event& event : kitty.terminal.events) {
                switch (event.type) {
                    case kitty_protocol::EventType::KEY_PRESS:
                        if (event.code == kitty_protocol::keys::ESCAPE || (event.code == 'c' && (event.modifiers & kitty_protocol::modifiers::CONTROL))) {
                            input.quit = true;
                        } else if (event.code == 'c') {
                            input.capture_pose = true;
                        } else if (event.code == kitty_protocol::keys::TAB) {
                            input.cycle_view = true;
                        }
                        break;
                    case kitty_protocol::EventType::MOUSE_PRESS:
                    case kitty_protocol::EventType::MOUSE_RELEASE:
                        if (event.code == kitty_protocol::buttons::LEFT) {
                            kitty.dragging = event.type == kitty_protocol::EventType::MOUSE_PRESS;
                            kitty.mouse_x = event.x;
                            kitty.mouse_y = event.y;
                        }
                        break;
                    case kitty_protocol::EventType::MOUSE_DRAG:
                        if (event.code == kitty_protocol::buttons::LEFT && kitty.dragging) {
                            hyperlux::interactive::look(input, static_cast<float>(event.x - kitty.mouse_x), static_cast<float>(event.y - kitty.mouse_y));
                            kitty.mouse_x = event.x;
                            kitty.mouse_y = event.y;
                        }
                        break;
                    case kitty_protocol::EventType::FOCUS_LOST:
                        kitty.dragging = false;
                        break;
                    case kitty_protocol::EventType::HANGUP:
                        input.quit = true;
                        break;
                    default: break;
                }
            }
        } while (!input.quit && !ready(device, kitty.terminal, kitty.image));
        input.forward = held(device, kitty.terminal, 'w') || held(device, kitty.terminal, kitty_protocol::keys::UP);
        input.backward = held(device, kitty.terminal, 's') || held(device, kitty.terminal, kitty_protocol::keys::DOWN);
        input.left = held(device, kitty.terminal, 'a') || held(device, kitty.terminal, kitty_protocol::keys::LEFT);
        input.right = held(device, kitty.terminal, 'd') || held(device, kitty.terminal, kitty_protocol::keys::RIGHT);
        input.up = held(device, kitty.terminal, kitty_protocol::keys::SPACE);
        input.down = held(device, kitty.terminal, kitty_protocol::keys::LEFT_SHIFT);
    }

    template <typename DEVICE>
    void present(DEVICE& device, hyperlux::interactive::Kitty& kitty, const uint32_t* pixels) {
        const utils::kitty::Terminal& terminal = kitty.terminal;
        if (terminal.rows <= terminal.log_rows) {
            return;
        }
        utils::kitty::Placement area;
        area.columns = terminal.columns;
        area.rows = terminal.rows - terminal.log_rows;
        transmit(device, kitty.terminal, kitty.image, pixels, kitty.width, kitty.height, fit(device, terminal, kitty.width, kitty.height, area));
    }

    template <typename DEVICE>
    void free(DEVICE& device, hyperlux::interactive::Kitty& kitty) {
        free(device, kitty.terminal);
    }
}
