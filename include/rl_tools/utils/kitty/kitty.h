#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_KITTY_KITTY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_KITTY_KITTY_H

#include "../../rl_tools.h"

#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::kitty {
    // keys are identified by their unshifted unicode code point or by these kitty keyboard protocol codes
    namespace keys {
        constexpr unsigned int TAB = 9, ENTER = 13, ESCAPE = 27, SPACE = 32;
        constexpr unsigned int LEFT = 57350, RIGHT = 57351, UP = 57352, DOWN = 57353, LEFT_SHIFT = 57441;
    }
    namespace modifiers {
        constexpr unsigned int SHIFT = 1, ALT = 2, CONTROL = 4;
    }
    namespace buttons {
        constexpr unsigned int LEFT = 0, MIDDLE = 1, RIGHT = 2;
    }

    enum class EventType {
        KEY_PRESS,
        KEY_REPEAT,
        KEY_RELEASE,
        MOUSE_PRESS,
        MOUSE_RELEASE,
        MOUSE_DRAG,
        FOCUS_LOST,
        HANGUP
    };

    struct Event {
        EventType type = EventType::KEY_PRESS;
        unsigned int code = 0;
        unsigned int modifiers = 0;
        int x = 0, y = 0;
    };

    // an image placed here spans `columns` cells and as many rows as its aspect ratio needs (fit computes them)
    struct Placement {
        unsigned int column = 0, row = 0, columns = 0, rows = 0;
    };

    struct Image {
        unsigned int ids[2] = {};
        unsigned int id = 0;
    };

    struct Transmission {
        unsigned int id = 0;
        std::string shared_memory_name;
    };

    struct Terminal {
        int tty = -1;
        unsigned int log_rows = 2;
        bool shared_memory = false;
        unsigned short rows = 0, columns = 0;
        float cell_width = 0, cell_height = 0;
        bool resized = false;
        unsigned int next_image_id = 0;
        unsigned long transmission_index = 0;
        std::vector<Transmission> in_flight;
        std::vector<unsigned int> held;
        std::vector<Event> events;
        std::string input_buffer;
        std::string output;
        std::string encoded;
        std::vector<unsigned char> rgb;
        std::vector<unsigned char> compressed;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
