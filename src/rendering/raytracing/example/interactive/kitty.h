#pragma once

#include "input.h"

#include <zlib.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <termios.h>
#include <unistd.h>

namespace rl_tools::rendering::raytracing::interactive {
    // Frames go through POSIX shared memory when the terminal runs on this host and are zlib-compressed into the terminal
    // stream otherwise (e.g. over ssh). Each frame waits for the terminal's acknowledgment of the previous one, so the frame
    // rate adapts to the link instead of queueing frames.
    struct Kitty {
        unsigned int width = 0;
        unsigned int height = 0;
        int tty = -1;
        bool shared_memory = false;
        bool frame_in_flight = false;
        unsigned long frame_index = 0;
        std::string frame_shared_memory_name;
        unsigned short terminal_rows = 0;
        unsigned short terminal_columns = 0;
        bool dragging = false;
        unsigned int mouse_x = 0, mouse_y = 0;
        std::string input_buffer;
        std::string output;
        std::string encoded;
        std::vector<unsigned char> rgb;
        std::vector<unsigned char> compressed;
    };
    namespace kitty {
        constexpr unsigned int IMAGE_ID = 1;
        constexpr unsigned int PROBE_SHARED_MEMORY_ID = 2;
        constexpr unsigned int PROBE_DIRECT_ID = 3;
        constexpr int PROBE_TIMEOUT_MS = 2000;
        constexpr int COMPRESSION_LEVEL = 1;
        constexpr std::size_t CHUNK_SIZE = 4096;
        constexpr unsigned int KEY_TAB = 9, KEY_ESCAPE = 27, KEY_SPACE = 32, KEY_A = 97, KEY_C = 99, KEY_D = 100, KEY_S = 115, KEY_W = 119, KEY_LEFT_SHIFT = 57441;
        constexpr unsigned int MODIFIER_CONTROL = 4;
        constexpr unsigned int EVENT_PRESS = 1, EVENT_RELEASE = 3;
        constexpr unsigned int MOUSE_MOTION = 32, MOUSE_WHEEL = 64;
        constexpr char ENTER_SCREEN[] =
            "\x1b[?1049h\x1b[?25l"              // alternate screen, hidden cursor
            "\x1b[>11u"                         // keyboard flags: disambiguate | report event types | report all keys as escape codes
            "\x1b[?1002h\x1b[?1006h\x1b[?1016h" // mouse drag tracking in SGR pixel coordinates
            "\x1b[?1004h";                      // focus events
        constexpr char LEAVE_SCREEN[] = "\x1b_Ga=d,d=A,q=2\x1b\\\x1b[<u\x1b[?1004l\x1b[?1016l\x1b[?1006l\x1b[?1002l\x1b[r\x1b[?25h\x1b[?1049l";

        enum TerminalState : int {
            UNTOUCHED = 0,
            RAW = 1,
            SCREEN = 2
        };
        inline int restore_tty = -1;
        inline termios original_termios;
        inline volatile std::sig_atomic_t terminal_state = UNTOUCHED;
        inline bool restore_handlers_installed = false;

        inline void restore_terminal() {
            if (terminal_state == SCREEN) {
                [[maybe_unused]] const ssize_t written = ::write(restore_tty, LEAVE_SCREEN, sizeof(LEAVE_SCREEN) - 1);
            }
            if (terminal_state != UNTOUCHED) {
                ::tcsetattr(restore_tty, TCSAFLUSH, &original_termios);
            }
            terminal_state = UNTOUCHED;
        }

        inline void restore_terminal_and_reraise(int signal_number) {
            restore_terminal();
            std::raise(signal_number);
        }

        inline void write_all(int fd, const std::string& data) {
            std::size_t offset = 0;
            while (offset < data.size()) {
                const ssize_t written = ::write(fd, data.data() + offset, data.size() - offset);
                if (written < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return;
                }
                offset += static_cast<std::size_t>(written);
            }
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
                std::size_t end = std::min(sequence.body.find(',', key), separator);
                if (sequence.body.compare(key, 2, "i=") == 0) {
                    id = static_cast<unsigned int>(std::strtoul(sequence.body.c_str() + key + 2, nullptr, 10));
                }
                key = end + 1;
            }
            message.assign(sequence.body, separator + 1, std::string::npos);
            return true;
        }

        inline bool enter_raw_mode(int tty) {
            if (::tcgetattr(tty, &original_termios) != 0) {
                return false;
            }
            restore_tty = tty;
            if (!restore_handlers_installed) {
                restore_handlers_installed = true;
                std::atexit(restore_terminal);
                struct sigaction action{};
                action.sa_handler = restore_terminal_and_reraise;
                sigemptyset(&action.sa_mask);
                action.sa_flags = SA_RESETHAND;
                for (const int signal_number : {SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL}) {
                    ::sigaction(signal_number, &action, nullptr);
                }
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

        // whether the controlling terminal speaks the kitty graphics and keyboard protocols; dumb terminals are not probed and
        // neither are background processes, which the terminal would stop for touching it
        inline bool available() {
            const char* term = std::getenv("TERM");
            if (term != nullptr && std::strcmp(term, "dumb") == 0) {
                return false;
            }
            const int tty = ::open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC);
            if (tty < 0) {
                return false;
            }
            bool supported = false;
            if (::tcgetpgrp(tty) == ::getpgrp() && enter_raw_mode(tty)) {
                std::string input_buffer;
                const Capabilities capabilities = probe(tty, input_buffer);
                supported = capabilities.graphics && capabilities.keyboard;
                restore_terminal();
            }
            ::close(tty);
            return supported;
        }

        inline void set_key(InputState& input, unsigned int key, unsigned int event, unsigned int modifiers) {
            const bool pressed = event != EVENT_RELEASE;
            switch (key) {
                case KEY_W: input.forward = pressed; break;
                case KEY_S: input.backward = pressed; break;
                case KEY_A: input.left = pressed; break;
                case KEY_D: input.right = pressed; break;
                case KEY_SPACE: input.up = pressed; break;
                case KEY_LEFT_SHIFT: input.down = pressed; break;
                case KEY_ESCAPE:
                    if (event == EVENT_PRESS) {
                        input.quit = true;
                    }
                    break;
                case KEY_C:
                    if (event == EVENT_PRESS) {
                        if (modifiers & MODIFIER_CONTROL) {
                            input.quit = true;
                        } else {
                            input.capture_pose = true;
                        }
                    }
                    break;
                case KEY_TAB:
                    if (event == EVENT_PRESS) {
                        input.cycle_view = true;
                    }
                    break;
                default: break;
            }
        }

        inline void set_arrow(InputState& input, char final, unsigned int event) {
            const bool pressed = event != EVENT_RELEASE;
            switch (final) {
                case 'A': input.forward = pressed; break;
                case 'B': input.backward = pressed; break;
                case 'C': input.right = pressed; break;
                case 'D': input.left = pressed; break;
                default: break;
            }
        }

        template <typename DEVICE>
        void handle_sequence(DEVICE& device, Kitty& kitty, InputState& input, const Sequence& sequence) {
            unsigned int id;
            std::string message;
            if (graphics_response(sequence, id, message)) {
                if (id == IMAGE_ID && kitty.frame_in_flight) {
                    kitty.frame_in_flight = false;
                    if (!kitty.frame_shared_memory_name.empty()) {
                        ::shm_unlink(kitty.frame_shared_memory_name.c_str());
                        kitty.frame_shared_memory_name.clear();
                    }
                    if (message != "OK") {
                        log(device, device.logger, devices::logging::Verbosity::error, "kitty: frame rejected: ", message);
                    }
                }
                return;
            }
            if (sequence.introducer != '[' || (!sequence.body.empty() && sequence.body[0] == '?')) {
                return;
            }
            const unsigned int event = parameter(sequence.body, 1, 1, EVENT_PRESS);
            switch (sequence.final) {
                case 'u':
                    set_key(input, parameter(sequence.body, 0, 0, 0), event, parameter(sequence.body, 1, 0, 1) - 1);
                    break;
                case 'A': case 'B': case 'C': case 'D':
                    set_arrow(input, sequence.final, event);
                    break;
                case 'O': // focus lost: releases of keys held at that moment never arrive
                    input.forward = input.backward = input.left = input.right = input.up = input.down = false;
                    kitty.dragging = false;
                    break;
                case 'M': case 'm': {
                    if (sequence.body.empty() || sequence.body[0] != '<') {
                        break;
                    }
                    const unsigned int button = parameter(sequence.body, 0, 0, 0);
                    const unsigned int x = parameter(sequence.body, 1, 0, 0);
                    const unsigned int y = parameter(sequence.body, 2, 0, 0);
                    if ((button & 3) != 0 || (button & MOUSE_WHEEL)) {
                        break;
                    }
                    if (button & MOUSE_MOTION) {
                        if (kitty.dragging) {
                            look(input, static_cast<float>(x) - static_cast<float>(kitty.mouse_x), static_cast<float>(y) - static_cast<float>(kitty.mouse_y));
                        }
                    } else {
                        kitty.dragging = sequence.final == 'M';
                    }
                    kitty.mouse_x = x;
                    kitty.mouse_y = y;
                    break;
                }
                default: break;
            }
        }
    }
}

namespace rl_tools {
    template <typename DEVICE>
    bool init(DEVICE& device, rendering::raytracing::interactive::Kitty& kitty, unsigned int width, unsigned int height) {
        namespace kitty_protocol = rendering::raytracing::interactive::kitty;
        kitty.width = width;
        kitty.height = height;
        kitty.tty = ::open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC);
        if (kitty.tty < 0 || !kitty_protocol::enter_raw_mode(kitty.tty)) {
            log(device, device.logger, devices::logging::Verbosity::error, "kitty: no controlling terminal");
            return false;
        }
        const kitty_protocol::Capabilities capabilities = kitty_protocol::probe(kitty.tty, kitty.input_buffer);
        if (!capabilities.graphics || !capabilities.keyboard) {
            kitty_protocol::restore_terminal();
            log(device, device.logger, devices::logging::Verbosity::error, "kitty: the terminal does not support the kitty ", !capabilities.graphics ? "graphics" : "keyboard", " protocol (tmux and screen do not pass it through)");
            return false;
        }
        kitty.shared_memory = capabilities.shared_memory;
        log(device, device.logger, devices::logging::Verbosity::info, "kitty: transmitting frames ", kitty.shared_memory ? "through shared memory" : "zlib-compressed through the terminal stream (no shared memory with the terminal host)");
        kitty_protocol::write_all(kitty.tty, kitty_protocol::ENTER_SCREEN);
        kitty_protocol::terminal_state = kitty_protocol::SCREEN;
        return true;
    }

    template <typename DEVICE>
    void poll(DEVICE& device, rendering::raytracing::interactive::Kitty& kitty, rendering::raytracing::interactive::InputState& input) {
        namespace kitty_protocol = rendering::raytracing::interactive::kitty;
        char buffer[4096];
        kitty_protocol::Sequence sequence;
        while (!input.quit) {
            while (!input.quit && kitty_protocol::pop_sequence(kitty.input_buffer, sequence)) {
                kitty_protocol::handle_sequence(device, kitty, input, sequence);
            }
            pollfd descriptor{kitty.tty, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, kitty.frame_in_flight ? -1 : 0);
            if (ready == 0 || input.quit) {
                break;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                input.quit = true;
                break;
            }
            const ssize_t count = ::read(kitty.tty, buffer, sizeof(buffer));
            if (count <= 0) {
                if (count < 0 && (errno == EINTR || errno == EAGAIN)) {
                    continue;
                }
                input.quit = true;
                break;
            }
            kitty.input_buffer.append(buffer, static_cast<std::size_t>(count));
        }
    }

    template <typename DEVICE>
    void present(DEVICE& device, rendering::raytracing::interactive::Kitty& kitty, const uint32_t* pixels) {
        namespace kitty_protocol = rendering::raytracing::interactive::kitty;
        winsize size{};
        if (::ioctl(kitty.tty, TIOCGWINSZ, &size) != 0 || size.ws_row < 3 || size.ws_col < 1) {
            return;
        }
        std::string& output = kitty.output;
        output.clear();
        if (size.ws_row != kitty.terminal_rows || size.ws_col != kitty.terminal_columns) {
            kitty.terminal_rows = size.ws_row;
            kitty.terminal_columns = size.ws_col;
            // log lines scroll inside the two bottom rows (cursor parked on the last), so the latest one stays visible without moving the image
            output += "\x1b[2J\x1b[" + std::to_string(size.ws_row - 1) + ";" + std::to_string(size.ws_row) + "r";
        }
        const unsigned int image_rows = size.ws_row - 2u;
        const float cell_aspect = size.ws_xpixel > 0 && size.ws_ypixel > 0 ? (static_cast<float>(size.ws_ypixel) / size.ws_row) / (static_cast<float>(size.ws_xpixel) / size.ws_col) : 2.0f;
        const float image_aspect = static_cast<float>(kitty.width) / static_cast<float>(kitty.height);
        const unsigned int columns = std::max(1u, std::min<unsigned int>(size.ws_col, static_cast<unsigned int>(image_rows * cell_aspect * image_aspect)));
        const unsigned int rows = std::min<unsigned int>(image_rows, static_cast<unsigned int>(std::ceil(columns / (cell_aspect * image_aspect))));
        output += "\x1b[" + std::to_string((image_rows - rows) / 2 + 1) + ";" + std::to_string((size.ws_col - columns) / 2 + 1) + "H";
        const std::size_t pixel_count = static_cast<std::size_t>(kitty.width) * kitty.height;
        const std::size_t rgb_size = pixel_count * 3;
        const std::string control = "a=T,f=24,s=" + std::to_string(kitty.width) + ",v=" + std::to_string(kitty.height) + ",i=" + std::to_string(kitty_protocol::IMAGE_ID) + ",p=1,c=" + std::to_string(columns) + ",C=1";
        if (kitty.shared_memory) {
            const std::string name = "/rlt-kitty-" + std::to_string(::getpid()) + "-" + std::to_string(kitty.frame_index++);
            if (unsigned char* data = kitty_protocol::create_shared_memory(name, rgb_size)) {
                kitty_protocol::pack_rgb(pixels, data, pixel_count);
                ::munmap(data, rgb_size);
                kitty.frame_shared_memory_name = name;
                output += "\x1b_G" + control + ",t=s,S=" + std::to_string(rgb_size) + ";";
                kitty_protocol::append_base64(output, name);
                output += "\x1b\\";
            } else {
                log(device, device.logger, devices::logging::Verbosity::error, "kitty: creating shared memory failed, switching to the terminal stream");
                kitty.shared_memory = false;
            }
        }
        if (!kitty.shared_memory) {
            kitty.rgb.resize(rgb_size);
            kitty_protocol::pack_rgb(pixels, kitty.rgb.data(), pixel_count);
            uLongf compressed_size = compressBound(static_cast<uLong>(rgb_size));
            kitty.compressed.resize(compressed_size);
            compress2(kitty.compressed.data(), &compressed_size, kitty.rgb.data(), static_cast<uLong>(rgb_size), kitty_protocol::COMPRESSION_LEVEL);
            kitty.encoded.clear();
            kitty_protocol::append_base64(kitty.encoded, kitty.compressed.data(), compressed_size);
            for (std::size_t offset = 0; offset < kitty.encoded.size(); offset += kitty_protocol::CHUNK_SIZE) {
                const char* more = offset + kitty_protocol::CHUNK_SIZE < kitty.encoded.size() ? "1" : "0";
                output += offset == 0 ? "\x1b_G" + control + ",o=z,m=" + more + ";" : std::string("\x1b_Gm=") + more + ";";
                output.append(kitty.encoded, offset, kitty_protocol::CHUNK_SIZE);
                output += "\x1b\\";
            }
        }
        output += "\x1b[" + std::to_string(size.ws_row) + ";1H";
        kitty_protocol::write_all(kitty.tty, output);
        kitty.frame_in_flight = true;
    }

    template <typename DEVICE>
    void free(DEVICE& /*device*/, rendering::raytracing::interactive::Kitty& kitty) {
        if (!kitty.frame_shared_memory_name.empty()) {
            ::shm_unlink(kitty.frame_shared_memory_name.c_str());
        }
        rendering::raytracing::interactive::kitty::restore_terminal();
        ::close(kitty.tty);
    }
}
