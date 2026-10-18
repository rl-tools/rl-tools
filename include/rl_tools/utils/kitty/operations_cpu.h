#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_KITTY_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_KITTY_OPERATIONS_CPU_H

#include "../../rl_tools.h"
#include "../../devices/devices.h"
#include "kitty.h"

#include <zlib.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <termios.h>
#include <unistd.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::kitty {
    // Images go through POSIX shared memory when the terminal runs on this host and are zlib-compressed into the terminal stream
    // otherwise (e.g. over ssh). An image is only re-transmitted after the terminal acknowledged its previous frame (see ready), so
    // the frame rate adapts to the link instead of queueing frames.
    constexpr unsigned int PROBE_SHARED_MEMORY_ID = 1;
    constexpr unsigned int PROBE_DIRECT_ID = 2;
    constexpr unsigned int FIRST_IMAGE_ID = 3;
    constexpr int PROBE_TIMEOUT_MS = 2000;
    constexpr int COMPRESSION_LEVEL = 1;
    constexpr std::size_t CHUNK_SIZE = 4096;
    constexpr float FALLBACK_CELL_WIDTH = 8, FALLBACK_CELL_HEIGHT = 16;

    namespace detail {
        constexpr unsigned int KEY_EVENT_PRESS = 1, KEY_EVENT_REPEAT = 2, KEY_EVENT_RELEASE = 3;
        constexpr unsigned int MOUSE_SHIFT = 4, MOUSE_ALT = 8, MOUSE_CONTROL = 16, MOUSE_MOTION = 32;
        constexpr char ENTER_SCREEN[] =
            "\x1b[?1049h\x1b[?25l"              // alternate screen, hidden cursor
            "\x1b[>11u"                         // keyboard flags: disambiguate | report event types | report all keys as escape codes
            "\x1b[?1002h\x1b[?1006h\x1b[?1016h" // mouse drag tracking in SGR pixel coordinates
            "\x1b[?1004h";                      // focus events
        constexpr char LEAVE_SCREEN[] = "\x1b_Ga=d,d=A,q=2\x1b\\\x1b[<u\x1b[?1004l\x1b[?1016l\x1b[?1006l\x1b[?1002l\x1b[r\x1b[?25h\x1b[?1049l";
        constexpr int RESTORE_SIGNALS[] = {SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};
        constexpr std::size_t RESTORE_SIGNAL_COUNT = sizeof(RESTORE_SIGNALS) / sizeof(RESTORE_SIGNALS[0]);

        enum TerminalState : int {
            UNTOUCHED = 0,
            RAW = 1,
            SCREEN = 2
        };
        inline int restore_tty = -1;
        inline termios original_termios;
        inline struct sigaction previous_actions[RESTORE_SIGNAL_COUNT];
        inline volatile std::sig_atomic_t terminal_state = UNTOUCHED;
        inline volatile std::sig_atomic_t writing = 0;
        inline bool exit_handler_installed = false;

        inline void restore_terminal() {
            if (terminal_state == UNTOUCHED) {
                return;
            }
            if (terminal_state == SCREEN) {
                if (writing) {
                    // a signal interrupted a write: terminate its escape sequence so the terminal parses what follows
                    [[maybe_unused]] const ssize_t terminated = ::write(restore_tty, "\x1b\\", 2);
                }
                [[maybe_unused]] const ssize_t written = ::write(restore_tty, LEAVE_SCREEN, sizeof(LEAVE_SCREEN) - 1);
            }
            ::tcsetattr(restore_tty, TCSAFLUSH, &original_termios);
            for (std::size_t i = 0; i < RESTORE_SIGNAL_COUNT; i++) {
                ::sigaction(RESTORE_SIGNALS[i], &previous_actions[i], nullptr);
            }
            terminal_state = UNTOUCHED;
        }

        // the reinstated previous disposition (default or the application's handler) receives the signal once this returns
        inline void restore_terminal_and_reraise(int signal_number) {
            restore_terminal();
            for (std::size_t i = 0; i < RESTORE_SIGNAL_COUNT; i++) {
                if (RESTORE_SIGNALS[i] == signal_number) {
                    ::sigaction(signal_number, &previous_actions[i], nullptr);
                }
            }
            std::raise(signal_number);
        }

        // one terminal at a time: a second activation would record its own handlers as the ones to reinstate
        inline bool enter_raw_mode(int tty) {
            if (terminal_state != UNTOUCHED || ::tcgetattr(tty, &original_termios) != 0) {
                return false;
            }
            restore_tty = tty;
            if (!exit_handler_installed) {
                exit_handler_installed = true;
                std::atexit(restore_terminal);
            }
            struct sigaction action{};
            action.sa_handler = restore_terminal_and_reraise;
            sigemptyset(&action.sa_mask);
            for (std::size_t i = 0; i < RESTORE_SIGNAL_COUNT; i++) {
                ::sigaction(RESTORE_SIGNALS[i], &action, &previous_actions[i]);
            }
            termios raw = original_termios;
            raw.c_iflag &= ~static_cast<tcflag_t>(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
            raw.c_lflag &= ~static_cast<tcflag_t>(ECHO | ICANON | IEXTEN | ISIG);
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            ::tcsetattr(tty, TCSANOW, &raw);
            terminal_state = RAW;
            return true;
        }

        inline void write_all(int fd, const std::string& data) {
            writing = 1;
            std::size_t offset = 0;
            while (offset < data.size()) {
                const ssize_t written = ::write(fd, data.data() + offset, data.size() - offset);
                if (written < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    break;
                }
                offset += static_cast<std::size_t>(written);
            }
            writing = 0;
        }

        inline void append_base64(std::string& output, const unsigned char* data, std::size_t size) {
            static constexpr char ALPHABET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            output.reserve(output.size() + (size + 2) / 3 * 4);
            std::size_t i = 0;
            for (; i + 2 < size; i += 3) {
                const unsigned int triple = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
                output += ALPHABET[(triple >> 18) & 63];
                output += ALPHABET[(triple >> 12) & 63];
                output += ALPHABET[(triple >> 6) & 63];
                output += ALPHABET[triple & 63];
            }
            if (i < size) {
                const unsigned int triple = (data[i] << 16) | (i + 1 < size ? data[i + 1] << 8 : 0);
                output += ALPHABET[(triple >> 18) & 63];
                output += ALPHABET[(triple >> 12) & 63];
                output += i + 1 < size ? ALPHABET[(triple >> 6) & 63] : '=';
                output += '=';
            }
        }

        inline void append_base64(std::string& output, const std::string& text) {
            append_base64(output, reinterpret_cast<const unsigned char*>(text.data()), text.size());
        }

        inline void pack_rgb(const uint32_t* pixels, unsigned char* rgb, std::size_t count) {
            for (std::size_t i = 0; i < count; i++) {
                const uint32_t pixel = pixels[i];
                rgb[3 * i + 0] = pixel & 0xFF;
                rgb[3 * i + 1] = (pixel >> 8) & 0xFF;
                rgb[3 * i + 2] = (pixel >> 16) & 0xFF;
            }
        }

        inline unsigned char* create_shared_memory(const std::string& name, std::size_t size) {
            const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd < 0) {
                return nullptr;
            }
            void* data = ::ftruncate(fd, static_cast<off_t>(size)) == 0 ? ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
            ::close(fd);
            if (data == MAP_FAILED) {
                ::shm_unlink(name.c_str());
                return nullptr;
            }
            return static_cast<unsigned char*>(data);
        }

        inline std::string cursor_position(unsigned int row, unsigned int column) {
            return "\x1b[" + std::to_string(row + 1) + ";" + std::to_string(column + 1) + "H";
        }

        struct Sequence {
            char introducer = 0;
            char final = 0;
            std::string body;
        };

        // pops the next complete CSI (ESC [) or APC (ESC _) sequence, skipping everything else
        inline bool pop_sequence(std::string& buffer, Sequence& sequence) {
            while (!buffer.empty()) {
                const std::size_t escape = buffer.find('\x1b');
                if (escape == std::string::npos) {
                    buffer.clear();
                    return false;
                }
                buffer.erase(0, escape);
                if (buffer.size() < 2) {
                    return false;
                }
                if (buffer[1] == '[') {
                    std::size_t end = 2;
                    while (end < buffer.size() && (buffer[end] < 0x40 || buffer[end] > 0x7E)) {
                        end++;
                    }
                    if (end == buffer.size()) {
                        return false;
                    }
                    sequence.introducer = '[';
                    sequence.final = buffer[end];
                    sequence.body.assign(buffer, 2, end - 2);
                    buffer.erase(0, end + 1);
                    return true;
                }
                if (buffer[1] == '_') {
                    const std::size_t end = buffer.find("\x1b\\", 2);
                    if (end == std::string::npos) {
                        return false;
                    }
                    sequence.introducer = '_';
                    sequence.final = 0;
                    sequence.body.assign(buffer, 2, end - 2);
                    buffer.erase(0, end + 2);
                    return true;
                }
                buffer.erase(0, 2);
            }
            return false;
        }

        // numeric CSI parameter: fields are separated by ';', subfields by ':'
        inline unsigned int parameter(const std::string& body, unsigned int field, unsigned int subfield, unsigned int default_value) {
            unsigned int current_field = 0, current_subfield = 0, value = 0;
            bool present = false;
            for (const char c : body) {
                if (c == ';') {
                    current_field++;
                    current_subfield = 0;
                } else if (c == ':') {
                    current_subfield++;
                } else if (c >= '0' && c <= '9' && current_field == field && current_subfield == subfield) {
                    value = value * 10 + static_cast<unsigned int>(c - '0');
                    present = true;
                }
            }
            return present ? value : default_value;
        }

        inline bool graphics_response(const Sequence& sequence, unsigned int& id, std::string& message) {
            if (sequence.introducer != '_' || sequence.body.empty() || sequence.body[0] != 'G') {
                return false;
            }
            const std::size_t separator = sequence.body.find(';');
            if (separator == std::string::npos) {
                return false;
            }
            id = 0;
            for (std::size_t key = 1; key < separator;) {
                const std::size_t end = std::min(sequence.body.find(',', key), separator);
                if (sequence.body.compare(key, 2, "i=") == 0) {
                    id = static_cast<unsigned int>(std::strtoul(sequence.body.c_str() + key + 2, nullptr, 10));
                }
                key = end + 1;
            }
            message.assign(sequence.body, separator + 1, std::string::npos);
            return true;
        }

        struct Capabilities {
            bool graphics = false;
            bool keyboard = false;
            bool shared_memory = false;
        };

        // the shared memory query only succeeds if the terminal can open objects of this host; the device attributes reply
        // (CSI c) terminates the probe because every terminal answers it after the queries before it
        inline Capabilities probe(int tty, std::string& input_buffer) {
            const std::string probe_name = "/rlt-kitty-" + std::to_string(::getpid()) + "-probe";
            std::string queries;
            if (unsigned char* probe_pixel = create_shared_memory(probe_name, 3)) {
                ::munmap(probe_pixel, 3);
                queries += "\x1b_Gi=" + std::to_string(PROBE_SHARED_MEMORY_ID) + ",a=q,t=s,f=24,s=1,v=1;";
                append_base64(queries, probe_name);
                queries += "\x1b\\";
            }
            queries += "\x1b_Gi=" + std::to_string(PROBE_DIRECT_ID) + ",a=q,t=d,f=24,s=1,v=1;AAAA\x1b\\\x1b[?u\x1b[c";
            write_all(tty, queries);
            Capabilities capabilities;
            bool attributes = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(PROBE_TIMEOUT_MS);
            char buffer[4096];
            while (!attributes) {
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
                if (remaining <= 0) {
                    break;
                }
                pollfd descriptor{tty, POLLIN, 0};
                const int ready = ::poll(&descriptor, 1, static_cast<int>(remaining));
                if (ready < 0 && errno == EINTR) {
                    continue;
                }
                if (ready <= 0) {
                    break;
                }
                const ssize_t count = ::read(tty, buffer, sizeof(buffer));
                if (count <= 0) {
                    break;
                }
                input_buffer.append(buffer, static_cast<std::size_t>(count));
                Sequence sequence;
                unsigned int id;
                std::string message;
                while (!attributes && pop_sequence(input_buffer, sequence)) {
                    if (graphics_response(sequence, id, message)) {
                        capabilities.shared_memory = capabilities.shared_memory || (id == PROBE_SHARED_MEMORY_ID && message == "OK");
                        capabilities.graphics = capabilities.graphics || (id == PROBE_DIRECT_ID && message == "OK");
                    } else if (sequence.introducer == '[' && !sequence.body.empty() && sequence.body[0] == '?') {
                        capabilities.keyboard = capabilities.keyboard || sequence.final == 'u';
                        attributes = sequence.final == 'c';
                    }
                }
            }
            ::shm_unlink(probe_name.c_str());
            return capabilities;
        }

        inline void update_size(Terminal& terminal) {
            winsize size{};
            if (::ioctl(terminal.tty, TIOCGWINSZ, &size) != 0 || size.ws_row == 0 || size.ws_col == 0) {
                return;
            }
            const float cell_width = size.ws_xpixel > 0 ? static_cast<float>(size.ws_xpixel) / size.ws_col : FALLBACK_CELL_WIDTH;
            const float cell_height = size.ws_ypixel > 0 ? static_cast<float>(size.ws_ypixel) / size.ws_row : FALLBACK_CELL_HEIGHT;
            if (size.ws_row == terminal.rows && size.ws_col == terminal.columns && cell_width == terminal.cell_width && cell_height == terminal.cell_height) {
                return;
            }
            terminal.rows = size.ws_row;
            terminal.columns = size.ws_col;
            terminal.cell_width = cell_width;
            terminal.cell_height = cell_height;
            terminal.resized = true;
            const unsigned int log_start = terminal.rows > terminal.log_rows ? terminal.rows - terminal.log_rows + 1 : 1;
            // log lines scroll inside the bottom rows (cursor parked there), so they never move the images and text above
            write_all(terminal.tty, "\x1b[2J\x1b[" + std::to_string(log_start) + ";" + std::to_string(terminal.rows) + "r" + cursor_position(terminal.rows - 1u, 0));
        }

        inline void key_event(Terminal& terminal, unsigned int code, const std::string& body) {
            const unsigned int event = parameter(body, 1, 1, KEY_EVENT_PRESS);
            Event key;
            key.type = event == KEY_EVENT_RELEASE ? EventType::KEY_RELEASE : event == KEY_EVENT_REPEAT ? EventType::KEY_REPEAT : EventType::KEY_PRESS;
            key.code = code;
            key.modifiers = parameter(body, 1, 0, 1) - 1;
            const auto held = std::find(terminal.held.begin(), terminal.held.end(), code);
            if (key.type == EventType::KEY_RELEASE) {
                if (held != terminal.held.end()) {
                    terminal.held.erase(held);
                }
            } else if (held == terminal.held.end()) {
                terminal.held.push_back(code);
            }
            terminal.events.push_back(key);
        }

        template <typename DEVICE>
        void handle_sequence(DEVICE& device, Terminal& terminal, const Sequence& sequence) {
            unsigned int id;
            std::string message;
            if (graphics_response(sequence, id, message)) {
                for (auto transmission = terminal.in_flight.begin(); transmission != terminal.in_flight.end(); ++transmission) {
                    if (transmission->id == id) {
                        if (!transmission->shared_memory_name.empty()) {
                            ::shm_unlink(transmission->shared_memory_name.c_str());
                        }
                        terminal.in_flight.erase(transmission);
                        if (message != "OK") {
                            log(device, device.logger, devices::logging::Verbosity::error, "kitty: image rejected: ", message);
                        }
                        break;
                    }
                }
                return;
            }
            if (sequence.introducer != '[' || (!sequence.body.empty() && sequence.body[0] == '?')) {
                return;
            }
            switch (sequence.final) {
                case 'u':
                    key_event(terminal, parameter(sequence.body, 0, 0, 0), sequence.body);
                    break;
                case 'A': key_event(terminal, keys::UP, sequence.body); break;
                case 'B': key_event(terminal, keys::DOWN, sequence.body); break;
                case 'C': key_event(terminal, keys::RIGHT, sequence.body); break;
                case 'D': key_event(terminal, keys::LEFT, sequence.body); break;
                case 'O': { // focus lost: releases of keys held at that moment never arrive
                    terminal.held.clear();
                    Event focus;
                    focus.type = EventType::FOCUS_LOST;
                    terminal.events.push_back(focus);
                    break;
                }
                case 'M': case 'm': {
                    if (sequence.body.empty() || sequence.body[0] != '<') {
                        break;
                    }
                    const unsigned int button = parameter(sequence.body, 0, 0, 0);
                    Event mouse;
                    mouse.type = (button & MOUSE_MOTION) ? EventType::MOUSE_DRAG : sequence.final == 'M' ? EventType::MOUSE_PRESS : EventType::MOUSE_RELEASE;
                    mouse.code = button & ~(MOUSE_SHIFT | MOUSE_ALT | MOUSE_CONTROL | MOUSE_MOTION);
                    mouse.modifiers = ((button & MOUSE_SHIFT) ? modifiers::SHIFT : 0) | ((button & MOUSE_ALT) ? modifiers::ALT : 0) | ((button & MOUSE_CONTROL) ? modifiers::CONTROL : 0);
                    mouse.x = static_cast<int>(parameter(sequence.body, 1, 0, 0));
                    mouse.y = static_cast<int>(parameter(sequence.body, 2, 0, 0));
                    terminal.events.push_back(mouse);
                    break;
                }
                default: break;
            }
        }
    }
}

namespace rl_tools {
    template <typename DEVICE>
    bool init(DEVICE& device, utils::kitty::Terminal& terminal) {
        namespace kitty = utils::kitty;
        terminal.tty = ::open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC);
        if (terminal.tty < 0 || !kitty::detail::enter_raw_mode(terminal.tty)) {
            log(device, device.logger, devices::logging::Verbosity::error, "kitty: no controlling terminal");
            return false;
        }
        const kitty::detail::Capabilities capabilities = kitty::detail::probe(terminal.tty, terminal.input_buffer);
        if (!capabilities.graphics || !capabilities.keyboard) {
            kitty::detail::restore_terminal();
            log(device, device.logger, devices::logging::Verbosity::error, "kitty: the terminal does not support the kitty ", !capabilities.graphics ? "graphics" : "keyboard", " protocol (tmux and screen do not pass it through)");
            return false;
        }
        terminal.shared_memory = capabilities.shared_memory;
        terminal.next_image_id = kitty::FIRST_IMAGE_ID;
        log(device, device.logger, devices::logging::Verbosity::info, "kitty: transmitting images ", terminal.shared_memory ? "through shared memory" : "zlib-compressed through the terminal stream (no shared memory with the terminal host)");
        kitty::detail::write_all(terminal.tty, kitty::detail::ENTER_SCREEN);
        kitty::detail::terminal_state = kitty::detail::SCREEN;
        terminal.rows = terminal.columns = 0;
        kitty::detail::update_size(terminal);
        return true;
    }

    template <typename DEVICE>
    void free(DEVICE& /*device*/, utils::kitty::Terminal& terminal) {
        for (const auto& transmission : terminal.in_flight) {
            if (!transmission.shared_memory_name.empty()) {
                ::shm_unlink(transmission.shared_memory_name.c_str());
            }
        }
        terminal.in_flight.clear();
        utils::kitty::detail::restore_terminal();
        if (terminal.tty >= 0) {
            ::close(terminal.tty);
            terminal.tty = -1;
        }
    }

    template <typename DEVICE>
    void init(DEVICE& /*device*/, utils::kitty::Terminal& terminal, utils::kitty::Image& image) {
        image.ids[0] = terminal.next_image_id++;
        image.ids[1] = terminal.next_image_id++;
        image.id = 0;
    }

    // collects the events since the last poll, waiting up to timeout_ms (-1: indefinitely) for the first input
    template <typename DEVICE>
    void poll(DEVICE& device, utils::kitty::Terminal& terminal, int timeout_ms) {
        namespace kitty = utils::kitty;
        terminal.events.clear();
        terminal.resized = false;
        kitty::Event hangup;
        hangup.type = kitty::EventType::HANGUP;
        if (kitty::detail::terminal_state != kitty::detail::SCREEN) {
            terminal.events.push_back(hangup);
            return;
        }
        kitty::detail::update_size(terminal);
        char buffer[4096];
        kitty::detail::Sequence sequence;
        int timeout = timeout_ms;
        while (true) {
            while (kitty::detail::pop_sequence(terminal.input_buffer, sequence)) {
                kitty::detail::handle_sequence(device, terminal, sequence);
            }
            pollfd descriptor{terminal.tty, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, timeout);
            if (ready == 0) {
                return;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                terminal.events.push_back(hangup);
                return;
            }
            const ssize_t count = ::read(terminal.tty, buffer, sizeof(buffer));
            if (count <= 0) {
                if (count < 0 && (errno == EINTR || errno == EAGAIN)) {
                    continue;
                }
                terminal.events.push_back(hangup);
                return;
            }
            terminal.input_buffer.append(buffer, static_cast<std::size_t>(count));
            timeout = 0;
        }
    }

    template <typename DEVICE>
    bool ready(DEVICE& /*device*/, const utils::kitty::Terminal& terminal, const utils::kitty::Image& image) {
        return std::none_of(terminal.in_flight.begin(), terminal.in_flight.end(), [&](const utils::kitty::Transmission& transmission) { return transmission.id == image.id; });
    }

    template <typename DEVICE>
    bool held(DEVICE& /*device*/, const utils::kitty::Terminal& terminal, unsigned int key) {
        return std::find(terminal.held.begin(), terminal.held.end(), key) != terminal.held.end();
    }

    // largest placement of a width x height image inside area that keeps its aspect ratio, centered
    template <typename DEVICE>
    utils::kitty::Placement fit(DEVICE& /*device*/, const utils::kitty::Terminal& terminal, unsigned int width, unsigned int height, const utils::kitty::Placement& area) {
        utils::kitty::Placement placement;
        if (area.columns == 0 || area.rows == 0) {
            return placement;
        }
        const float cell_aspect = terminal.cell_height / terminal.cell_width;
        const float image_aspect = static_cast<float>(width) / static_cast<float>(height);
        placement.columns = std::max(1u, std::min(area.columns, static_cast<unsigned int>(area.rows * cell_aspect * image_aspect)));
        placement.rows = std::min(area.rows, static_cast<unsigned int>(std::ceil(placement.columns / (cell_aspect * image_aspect))));
        placement.column = area.column + (area.columns - placement.columns) / 2;
        placement.row = area.row + (area.rows - placement.rows) / 2;
        return placement;
    }

    // maps a mouse position to normalized coordinates of the image shown at placement; false if outside
    template <typename DEVICE>
    bool locate(DEVICE& /*device*/, const utils::kitty::Terminal& terminal, const utils::kitty::Placement& placement, unsigned int width, unsigned int height, int x, int y, float& u, float& v) {
        const float image_width = placement.columns * terminal.cell_width;
        const float image_height = image_width * static_cast<float>(height) / static_cast<float>(width);
        u = (x - placement.column * terminal.cell_width) / image_width;
        v = (y - placement.row * terminal.cell_height) / image_height;
        return u >= 0 && u < 1 && v >= 0 && v < 1;
    }

    // pixels are packed RGBA (red in the lowest byte); returns false without sending while the previous frame is unacknowledged
    template <typename DEVICE>
    bool transmit(DEVICE& device, utils::kitty::Terminal& terminal, utils::kitty::Image& image, const uint32_t* pixels, unsigned int width, unsigned int height, const utils::kitty::Placement& placement) {
        namespace kitty = utils::kitty;
        if (!ready(device, terminal, image) || placement.columns == 0) {
            return false;
        }
        // frames alternate between two image ids and the previous one is deleted only after its successor is complete: re-transmitting
        // an existing id blanks its placements until the last chunk arrives, which flickers whenever chunks span a screen refresh
        const unsigned int previous_id = image.id;
        image.id = previous_id == image.ids[0] ? image.ids[1] : image.ids[0];
        kitty::Transmission transmission;
        transmission.id = image.id;
        std::string& output = terminal.output;
        output = "\x1b" "7" + kitty::detail::cursor_position(placement.row, placement.column);
        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        const std::size_t rgb_size = pixel_count * 3;
        const std::string control = "a=T,f=24,s=" + std::to_string(width) + ",v=" + std::to_string(height) + ",i=" + std::to_string(image.id) + ",p=1,c=" + std::to_string(placement.columns) + ",C=1";
        if (terminal.shared_memory) {
            const std::string name = "/rlt-kitty-" + std::to_string(::getpid()) + "-" + std::to_string(terminal.transmission_index++);
            if (unsigned char* data = kitty::detail::create_shared_memory(name, rgb_size)) {
                kitty::detail::pack_rgb(pixels, data, pixel_count);
                ::munmap(data, rgb_size);
                transmission.shared_memory_name = name;
                output += "\x1b_G" + control + ",t=s,S=" + std::to_string(rgb_size) + ";";
                kitty::detail::append_base64(output, name);
                output += "\x1b\\";
            } else {
                log(device, device.logger, devices::logging::Verbosity::error, "kitty: creating shared memory failed, switching to the terminal stream");
                terminal.shared_memory = false;
            }
        }
        if (!terminal.shared_memory) {
            terminal.rgb.resize(rgb_size);
            kitty::detail::pack_rgb(pixels, terminal.rgb.data(), pixel_count);
            uLongf compressed_size = compressBound(static_cast<uLong>(rgb_size));
            terminal.compressed.resize(compressed_size);
            compress2(terminal.compressed.data(), &compressed_size, terminal.rgb.data(), static_cast<uLong>(rgb_size), kitty::COMPRESSION_LEVEL);
            terminal.encoded.clear();
            kitty::detail::append_base64(terminal.encoded, terminal.compressed.data(), compressed_size);
            for (std::size_t offset = 0; offset < terminal.encoded.size(); offset += kitty::CHUNK_SIZE) {
                const char* more = offset + kitty::CHUNK_SIZE < terminal.encoded.size() ? "1" : "0";
                output += offset == 0 ? "\x1b_G" + control + ",o=z,m=" + more + ";" : std::string("\x1b_Gm=") + more + ";";
                output.append(terminal.encoded, offset, kitty::CHUNK_SIZE);
                output += "\x1b\\";
            }
        }
        if (previous_id != 0) {
            output += "\x1b_Ga=d,d=I,i=" + std::to_string(previous_id) + ",q=2\x1b\\";
        }
        output += "\x1b" "8";
        kitty::detail::write_all(terminal.tty, output);
        terminal.in_flight.push_back(transmission);
        return true;
    }

    // shows the last transmitted frame at a new placement without re-sending it (e.g. after a resize)
    template <typename DEVICE>
    void place(DEVICE& /*device*/, utils::kitty::Terminal& terminal, const utils::kitty::Image& image, const utils::kitty::Placement& placement) {
        if (image.id == 0) {
            return;
        }
        terminal.output = "\x1b" "7" + utils::kitty::detail::cursor_position(placement.row, placement.column) + "\x1b_Ga=p,i=" + std::to_string(image.id) + ",p=1,c=" + std::to_string(placement.columns) + ",C=1,q=2\x1b\\\x1b" "8";
        utils::kitty::detail::write_all(terminal.tty, terminal.output);
    }

    template <typename DEVICE>
    void hide(DEVICE& /*device*/, utils::kitty::Terminal& terminal, const utils::kitty::Image& image) {
        if (image.id == 0) {
            return;
        }
        terminal.output = "\x1b_Ga=d,d=i,i=" + std::to_string(image.id) + ",q=2\x1b\\";
        utils::kitty::detail::write_all(terminal.tty, terminal.output);
    }

    // replaces the rest of the row, truncated to the terminal width
    template <typename DEVICE>
    void text(DEVICE& /*device*/, utils::kitty::Terminal& terminal, unsigned int row, unsigned int column, const std::string& line) {
        if (row >= terminal.rows || column >= terminal.columns) {
            return;
        }
        terminal.output = "\x1b" "7" + utils::kitty::detail::cursor_position(row, column) + line.substr(0, terminal.columns - column) + "\x1b[K\x1b" "8";
        utils::kitty::detail::write_all(terminal.tty, terminal.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
