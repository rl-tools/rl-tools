#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0

#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION 3

#ifndef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"

#include <GLFW/glfw3.h>

#include <conta/conta.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace rlt = rl_tools;

static constexpr bool OUTPUT_RGB = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB || RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD;
static constexpr bool OUTPUT_DEPTH = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD || RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH;
static constexpr bool OUTPUT_SEGMENTATION = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION;

// Minimal 5x7 bitmap font for overlay text
struct FontGlyph {
    char ch;
    uint8_t rows[7];
};

static const FontGlyph FONT_GLYPHS[] = {
    {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}},
    {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
    {'2', {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}},
    {'3', {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}},
    {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}},
    {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
    {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}},
    {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}},
    {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
    {'.', {0x00,0x00,0x00,0x00,0x00,0x00,0x04}},
    {'-', {0x00,0x00,0x00,0x0E,0x00,0x00,0x00}},
    {'+', {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}},
    {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
    {':', {0x00,0x00,0x04,0x00,0x04,0x00,0x00}},
    {'(', {0x02,0x04,0x08,0x08,0x08,0x04,0x02}},
    {')', {0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
    {',', {0x00,0x00,0x00,0x00,0x00,0x04,0x08}},
    {'=', {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}},
    {'P', {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q', {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}},
    {'o', {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}},
    {'s', {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E}},
    {'u', {0x00,0x00,0x11,0x11,0x11,0x11,0x0F}},
    {'a', {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}},
    {'t', {0x08,0x08,0x1C,0x08,0x08,0x09,0x06}},
};

static const uint8_t* font_lookup(char ch) {
    static const uint8_t empty[7] = {};
    for (const auto& g : FONT_GLYPHS) {
        if (g.ch == ch) return g.rows;
    }
    return empty;
}

static void draw_char(uint32_t* pixels, int width, int height, int x0, int y0, char ch, uint32_t color) {
    const uint8_t* glyph = font_lookup(ch);
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[row] & (0x10 >> col)) {
                int px = x0 + col;
                int py = y0 + row;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    pixels[py * width + px] = color;
                }
            }
        }
    }
}

static void draw_string(uint32_t* pixels, int width, int height, int x0, int y0, const char* str, uint32_t color) {
    int x = x0;
    for (int i = 0; str[i] != '\0'; i++) {
        draw_char(pixels, width, height, x, y0, str[i], color);
        x += 6;
    }
}

#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD || RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
static void depth_to_rgba(const float* depth, uint32_t* pixels, int count, float max_depth) {
    for (int i = 0; i < count; i++) {
        float normalized = std::fmin(std::fmax(depth[i] / max_depth, 0.0f), 1.0f);
        uint8_t value = static_cast<uint8_t>(normalized * 255.0f);
        pixels[i] = (0xFFu << 24) | (uint32_t(value) << 16) | (uint32_t(value) << 8) | uint32_t(value);
    }
}
#endif

static void draw_overlay_background(uint32_t* pixels, int width, int /*height*/, int x0, int y0, int w, int h) {
    for (int row = y0; row < y0 + h; row++) {
        for (int col = x0; col < x0 + w; col++) {
            if (row >= 0 && col >= 0) {
                uint32_t pixel = pixels[row * width + col];
                uint8_t r = (pixel >> 0) & 0xFF;
                uint8_t g = (pixel >> 8) & 0xFF;
                uint8_t b = (pixel >> 16) & 0xFF;
                r = r / 3;
                g = g / 3;
                b = b / 3;
                pixels[row * width + col] = (0xFF << 24) | (b << 16) | (g << 8) | r;
            }
        }
    }
}

struct Quaternion {
    float w, x, y, z;
};

static void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static bool normalize3(float v[3]) {
    const float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if(norm <= 1e-6f) {
        return false;
    }
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    return true;
}

static Quaternion normalize_quaternion(Quaternion q) {
    const float norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if(norm <= 1e-6f) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    return {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

static Quaternion quaternion_from_rotation_columns(const float x_axis[3], const float y_axis[3], const float z_axis[3]) {
    const float m00 = x_axis[0], m01 = y_axis[0], m02 = z_axis[0];
    const float m10 = x_axis[1], m11 = y_axis[1], m12 = z_axis[1];
    const float m20 = x_axis[2], m21 = y_axis[2], m22 = z_axis[2];
    Quaternion q;
    const float trace = m00 + m11 + m22;
    if(trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    }
    else if(m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / s;
        q.x = 0.25f * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    }
    else if(m11 > m22) {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25f * s;
        q.z = (m12 + m21) / s;
    }
    else {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25f * s;
    }
    return normalize_quaternion(q);
}

static Quaternion l2f_camera_quaternion(const float forward_in[3], const float up_reference_in[3]) {
    float x_axis[3] = {forward_in[0], forward_in[1], forward_in[2]};
    if(!normalize3(x_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
    }
    float up_reference[3] = {up_reference_in[0], up_reference_in[1], up_reference_in[2]};
    if(!normalize3(up_reference)) {
        up_reference[0] = 0.0f;
        up_reference[1] = 0.0f;
        up_reference[2] = 1.0f;
    }
    float y_axis[3];
    cross3(up_reference, x_axis, y_axis);
    if(!normalize3(y_axis)) {
        const float fallback_up[3] = {0.0f, 1.0f, 0.0f};
        cross3(fallback_up, x_axis, y_axis);
        if(!normalize3(y_axis)) {
            y_axis[0] = 0.0f;
            y_axis[1] = 1.0f;
            y_axis[2] = 0.0f;
        }
    }
    float z_axis[3];
    cross3(x_axis, y_axis, z_axis);
    normalize3(z_axis);
    return quaternion_from_rotation_columns(x_axis, y_axis, z_axis);
}

struct InputState {
    bool forward = false, backward = false, left = false, right = false;
    bool up = false, down = false;
    double mouse_x = 0, mouse_y = 0;
    double last_mouse_x = 0, last_mouse_y = 0;
    bool first_mouse = true;
    bool cursor_captured = true;
    float yaw = 0;
    float pitch = 0;
};

static InputState g_input;
static bool g_capture_pose_requested = false;
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
static bool g_show_depth = true;
#endif

static constexpr char CAPTURED_POSE_PATH[] = "camera_poses.txt";

struct InteractiveOptions {
    std::string scene_arg;
    std::string record_camera_pose_path;
    float linear_velocity = 3.0f;
    bool help = false;
};

struct RecordedCameraPose {
    double timestamp_s;
    size_t frame_index;
    float position[3];
    float look_at[3];
    float up[3];
    float yaw;
    float pitch;
};

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [conta:HASH | scene.glb] [--record-camera-pose trace.json] [--linear-velocity m/s]" << std::endl;
    std::cerr << "  Without arguments the default scene is fetched via conta (downloaded into the cache if required)" << std::endl;
}

static bool parse_float_arg(const std::string& value, float& out) {
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if(end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

static bool parse_options(int argc, char** argv, InteractiveOptions& options) {
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "-h" || arg == "--help") {
            options.help = true;
            return true;
        }
        const std::string record_prefix = "--record-camera-pose=";
        const std::string record_trace_prefix = "--record-camera-trace=";
        const std::string linear_velocity_prefix = "--linear-velocity=";
        if(arg == "--record-camera-pose" || arg == "--record-camera-trace") {
            if(i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return false;
            }
            options.record_camera_pose_path = argv[++i];
        }
        else if(arg.compare(0, record_prefix.size(), record_prefix) == 0) {
            options.record_camera_pose_path = arg.substr(record_prefix.size());
        }
        else if(arg.compare(0, record_trace_prefix.size(), record_trace_prefix) == 0) {
            options.record_camera_pose_path = arg.substr(record_trace_prefix.size());
        }
        else if(arg == "--linear-velocity") {
            if(i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return false;
            }
            if(!parse_float_arg(argv[++i], options.linear_velocity) || options.linear_velocity < 0.0f) {
                std::cerr << "Invalid --linear-velocity" << std::endl;
                return false;
            }
        }
        else if(arg.compare(0, linear_velocity_prefix.size(), linear_velocity_prefix) == 0) {
            if(!parse_float_arg(arg.substr(linear_velocity_prefix.size()), options.linear_velocity) || options.linear_velocity < 0.0f) {
                std::cerr << "Invalid --linear-velocity" << std::endl;
                return false;
            }
        }
        else if(!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
        else if(options.scene_arg.empty()) {
            options.scene_arg = arg;
        }
        else {
            std::cerr << "Multiple scene arguments provided: " << options.scene_arg << " and " << arg << std::endl;
            return false;
        }
    }
    return true;
}

static std::string camera_cache_path(const std::string& scene_path) {
    const char* home = std::getenv("HOME");
    if (!home) return "";
    std::string cache_dir = std::string(home) + "/.cache/rl_tools_viewer";
    mkdir(cache_dir.c_str(), 0755);
    std::string key;
    for (char c : scene_path) key += (c == '/' || c == '\\') ? '_' : c;
    return cache_dir + "/" + key + ".cam";
}

static void save_camera_state(const std::string& path, float x, float y, float z, float yaw, float pitch) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::binary);
    if (f) f.write(reinterpret_cast<const char*>(&x), 4)
            .write(reinterpret_cast<const char*>(&y), 4)
            .write(reinterpret_cast<const char*>(&z), 4)
            .write(reinterpret_cast<const char*>(&yaw), 4)
            .write(reinterpret_cast<const char*>(&pitch), 4);
}

static bool load_camera_state(const std::string& path, float& x, float& y, float& z, float& yaw, float& pitch) {
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    return f.read(reinterpret_cast<char*>(&x), 4)
            .read(reinterpret_cast<char*>(&y), 4)
            .read(reinterpret_cast<char*>(&z), 4)
            .read(reinterpret_cast<char*>(&yaw), 4)
            .read(reinterpret_cast<char*>(&pitch), 4)
            .good();
}

static std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for(char c : value) {
        switch(c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if(static_cast<unsigned char>(c) < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(c));
                    escaped += buffer;
                }
                else {
                    escaped += c;
                }
        }
    }
    return escaped;
}

static void write_json_number(std::ostream& out, float value) {
    if(std::isfinite(value)) {
        out << value;
    }
    else {
        out << "null";
    }
}

static void write_json_vec3(std::ostream& out, const float v[3]) {
    out << "[";
    write_json_number(out, v[0]);
    out << ", ";
    write_json_number(out, v[1]);
    out << ", ";
    write_json_number(out, v[2]);
    out << "]";
}

static void write_json_quaternion(std::ostream& out, const Quaternion& q) {
    out << "[";
    write_json_number(out, q.w);
    out << ", ";
    write_json_number(out, q.x);
    out << ", ";
    write_json_number(out, q.y);
    out << ", ";
    write_json_number(out, q.z);
    out << "]";
}

static bool write_camera_trace_json(const std::string& path, const std::string& scene_path, const std::vector<RecordedCameraPose>& trace) {
    if(path.empty()) return true;
    const std::string tmp_path = path + ".tmp";

    std::ofstream f(tmp_path);
    if(!f) {
        std::cerr << "Failed to write camera pose trace JSON: " << tmp_path << std::endl;
        return false;
    }

    f << std::setprecision(9);
    f << "{\n";
    f << "  \"scene_path\": \"" << json_escape(scene_path) << "\",\n";
    f << "  \"yaw_pitch_units\": \"radians\",\n";
    f << "  \"frame_count\": " << trace.size() << ",\n";
    f << "  \"poses\": [\n";
    for(size_t i = 0; i < trace.size(); i++) {
        const RecordedCameraPose& pose = trace[i];
        const float forward[3] = {
            pose.look_at[0] - pose.position[0],
            pose.look_at[1] - pose.position[1],
            pose.look_at[2] - pose.position[2]
        };
        const Quaternion q = l2f_camera_quaternion(forward, pose.up);

        f << "    {\n";
        f << "      \"timestamp_s\": " << pose.timestamp_s << ",\n";
        f << "      \"frame_index\": " << pose.frame_index << ",\n";
        f << "      \"renderer\": {\n";
        f << "        \"frame\": \"interactive_camera_input\",\n";
        f << "        \"position\": ";
        write_json_vec3(f, pose.position);
        f << ",\n";
        f << "        \"yaw\": ";
        write_json_number(f, pose.yaw);
        f << ",\n";
        f << "        \"pitch\": ";
        write_json_number(f, pose.pitch);
        f << ",\n";
        f << "        \"quaternion_wxyz\": ";
        write_json_quaternion(f, q);
        f << ",\n";
        f << "        \"forward\": ";
        write_json_vec3(f, forward);
        f << ",\n";
        f << "        \"up\": ";
        write_json_vec3(f, pose.up);
        f << ",\n";
        f << "        \"look_at\": ";
        write_json_vec3(f, pose.look_at);
        f << "\n";
        f << "      }\n";
        f << "    }" << (i + 1 < trace.size() ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    f.close();
    if(!f) {
        std::cerr << "Failed while writing camera pose trace JSON: " << tmp_path << std::endl;
        return false;
    }
    if(std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "Failed to move camera pose trace JSON into place: " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

static bool append_captured_pose(const char* path, const float position[3], const float look_at[3], const float up[3], float yaw, float pitch) {
    std::ofstream f(path, std::ios::app);
    if(!f) {
        return false;
    }
    const float forward[3] = {
        look_at[0] - position[0],
        look_at[1] - position[1],
        look_at[2] - position[2]
    };
    const Quaternion q = l2f_camera_quaternion(forward, up);
    f << std::setprecision(9);
    f << "position=" << position[0] << "," << position[1] << "," << position[2];
    f << " yaw=" << yaw;
    f << " pitch=" << pitch;
    f << " quaternion_wxyz=" << q.w << "," << q.x << "," << q.y << "," << q.z;
    f << "\n";
    f.close();
    return static_cast<bool>(f);
}

static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        g_capture_pose_requested = true;
        return;
    }
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        g_show_depth = !g_show_depth;
        return;
    }
#endif
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    if (key == GLFW_KEY_W || key == GLFW_KEY_UP) g_input.forward = pressed;
    if (key == GLFW_KEY_S || key == GLFW_KEY_DOWN) g_input.backward = pressed;
    if (key == GLFW_KEY_A || key == GLFW_KEY_LEFT) g_input.left = pressed;
    if (key == GLFW_KEY_D || key == GLFW_KEY_RIGHT) g_input.right = pressed;
    if (key == GLFW_KEY_SPACE) g_input.up = pressed;
    if (key == GLFW_KEY_LEFT_SHIFT) g_input.down = pressed;
}

static void cursor_pos_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (!g_input.cursor_captured) return;
    if (g_input.first_mouse) {
        g_input.last_mouse_x = xpos;
        g_input.last_mouse_y = ypos;
        g_input.first_mouse = false;
        return;
    }
    float dx = static_cast<float>(xpos - g_input.last_mouse_x);
    float dy = static_cast<float>(ypos - g_input.last_mouse_y);
    g_input.last_mouse_x = xpos;
    g_input.last_mouse_y = ypos;

    constexpr float sensitivity = 0.002f;
    g_input.yaw -= dx * sensitivity;
    g_input.pitch -= dy * sensitivity;
    constexpr float max_pitch = static_cast<float>(M_PI) * 0.49f;
    if (g_input.pitch > max_pitch) g_input.pitch = max_pitch;
    if (g_input.pitch < -max_pitch) g_input.pitch = -max_pitch;
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        if (g_input.cursor_captured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            g_input.cursor_captured = false;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            g_input.cursor_captured = true;
            g_input.first_mouse = true;
        }
    }
}

int main(int argc, char** argv) {
    using T = float;
    using TI = unsigned int;
    constexpr TI CAM_WIDTH = 1280;
    constexpr TI CAM_HEIGHT = 960;
    constexpr TI NUM_ENVS = 1;
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, 64, rlt::rendering::raytracing::High, false, 1, true, 2, OUTPUT_RGB, OUTPUT_DEPTH, OUTPUT_SEGMENTATION>;
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;

    InteractiveOptions options;
    if(!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    if(options.help) {
        print_usage(argv[0]);
        return 0;
    }

    DEVICE device;
    rlt::init(device);

    static constexpr char DEFAULT_CONTA_HASH[] = "7f1c9129532798e0b63bc41edb6b4c09251cf8a0";
    std::string resolved_scene_path;
    std::string conta_error;
    if (!options.scene_arg.empty()) {
        const char* scene_arg = options.scene_arg.c_str();
        if (std::strncmp(scene_arg, "conta:", 6) == 0) {
            if (!conta::resolve(options.scene_arg.substr(6), resolved_scene_path, conta_error)) {
                std::cerr << conta_error << std::endl;
                return 1;
            }
        } else {
            resolved_scene_path = scene_arg;
        }
    } else {
        std::cout << "No scene argument given, using default: conta:" << DEFAULT_CONTA_HASH << std::endl;
        if (!conta::resolve(DEFAULT_CONTA_HASH, resolved_scene_path, conta_error)) {
            std::cerr << conta_error << std::endl;
            return 1;
        }
    }
    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = resolved_scene_path.c_str();

    rlt::malloc(device, env);
    rlt::init(device, env);
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
    std::cout << "RGBD target enabled. Press Tab to toggle RGB/depth display." << std::endl;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    std::cout << "Depth target enabled." << std::endl;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION
    std::cout << "Segmentation target enabled." << std::endl;
#endif

    rlt::rl::environments::raytracing_example::State<SPEC> state{};
    if (env.num_indoor_initial_states > 0) {
        state = env.indoor_initial_states[0];
    }

    std::string cam_cache = camera_cache_path(resolved_scene_path);
    float cached_yaw = state.yaw, cached_pitch = 0;
    if (load_camera_state(cam_cache, state.position[0], state.position[1], state.position[2], cached_yaw, cached_pitch)) {
        std::cout << "Restored camera from " << cam_cache << std::endl;
    }

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(CAM_WIDTH, CAM_HEIGHT, "RLtools Interactive Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    g_input.yaw = cached_yaw;
    g_input.pitch = cached_pitch;

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CAM_WIDTH, CAM_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    std::vector<uint32_t> pixels(CAM_WIDTH * CAM_HEIGHT);
    std::vector<RecordedCameraPose> camera_trace;
    rlt::Tensor<typename decltype(env.renderer->cameras)::SPEC> camera_staging;
    rlt::malloc(device, camera_staging);

    constexpr float LINEAR_VELOCITY_RAMP_S = 0.5f;
    float linear_velocity_ramp_elapsed_s = 0.0f;
    auto last_time = std::chrono::steady_clock::now();
    const auto trace_start_time = last_time;
    size_t frame_index = 0;

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        glfwPollEvents();

        float cos_yaw = std::cos(g_input.yaw);
        float sin_yaw = std::sin(g_input.yaw);
        float dx = 0, dy = 0, dz = 0;
        if (g_input.forward)  { dx += cos_yaw; dy += sin_yaw; }
        if (g_input.backward) { dx -= cos_yaw; dy -= sin_yaw; }
        if (g_input.left)     { dx -= sin_yaw; dy += cos_yaw; }
        if (g_input.right)    { dx += sin_yaw; dy -= cos_yaw; }
        if (g_input.up)       { dz += 1; }
        if (g_input.down)     { dz -= 1; }
        float move_len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (move_len > 0) {
            linear_velocity_ramp_elapsed_s = std::min(linear_velocity_ramp_elapsed_s + dt, LINEAR_VELOCITY_RAMP_S);
            const float ramp = LINEAR_VELOCITY_RAMP_S > 0.0f ? linear_velocity_ramp_elapsed_s / LINEAR_VELOCITY_RAMP_S : 1.0f;
            float speed = options.linear_velocity * ramp * dt / move_len;
            state.position[0] += dx * speed;
            state.position[1] += dy * speed;
            state.position[2] += dz * speed;
        }
        else {
            linear_velocity_ramp_elapsed_s = 0.0f;
        }
        state.yaw = g_input.yaw;

        T eye[3] = {state.position[0], state.position[1], state.position[2]};
        T look_at[3] = {
            eye[0] + std::cos(g_input.yaw) * std::cos(g_input.pitch),
            eye[1] + std::sin(g_input.yaw) * std::cos(g_input.pitch),
            eye[2] + std::sin(g_input.pitch)
        };
        T up[3] = {0, 0, 1};
        T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        const auto camera = rlt::make_camera_data(eye, look_at, up, SPEC::FOV, aspect);

        if (g_capture_pose_requested) {
            g_capture_pose_requested = false;
            if (append_captured_pose(CAPTURED_POSE_PATH, eye, look_at, up, g_input.yaw, g_input.pitch)) {
                std::cout << "Appended camera pose to " << CAPTURED_POSE_PATH << std::endl;
            }
            else {
                std::cerr << "Failed to append camera pose to " << CAPTURED_POSE_PATH << std::endl;
            }
        }

        if(!options.record_camera_pose_path.empty()) {
            RecordedCameraPose pose;
            pose.timestamp_s = std::chrono::duration<double>(now - trace_start_time).count();
            pose.frame_index = frame_index;
            pose.position[0] = eye[0];
            pose.position[1] = eye[1];
            pose.position[2] = eye[2];
            pose.look_at[0] = look_at[0];
            pose.look_at[1] = look_at[1];
            pose.look_at[2] = look_at[2];
            pose.up[0] = up[0];
            pose.up[1] = up[1];
            pose.up[2] = up[2];
            pose.yaw = g_input.yaw;
            pose.pitch = g_input.pitch;
            camera_trace.push_back(pose);
        }
        frame_index++;

        rlt::set(device, camera_staging, camera, 0);
        rlt::copy(device, env.renderer->device, camera_staging, rlt::cameras(device, *env.renderer));
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        rlt::render(device, *env.renderer);
        {
            std::vector<float> depth_staging(pixels.size());
            {
                rlt::Tensor<typename decltype(env.renderer->depth_buffer)::SPEC> depth_alias;
                depth_alias._data = depth_staging.data();
                rlt::copy(env.renderer->device, device, rlt::depth_buffer(device, *env.renderer), depth_alias);
            }
            const float max_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            depth_to_rgba(depth_staging.data(), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
        if (g_show_depth) {
            rlt::render(device, *env.renderer);
            std::vector<float> depth_staging(pixels.size());
            {
                rlt::Tensor<typename decltype(env.renderer->depth_buffer)::SPEC> depth_alias;
                depth_alias._data = depth_staging.data();
                rlt::copy(env.renderer->device, device, rlt::depth_buffer(device, *env.renderer), depth_alias);
            }
            const float max_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            depth_to_rgba(depth_staging.data(), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
        else {
            rlt::render(device, *env.renderer);
            {
                rlt::Tensor<typename decltype(env.renderer->frame_buffer)::SPEC> frame_alias;
                frame_alias._data = pixels.data();
                rlt::copy(env.renderer->device, device, rlt::frame_buffer(device, *env.renderer), frame_alias);
            }
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION
        rlt::render(device, *env.renderer);
        {
            std::vector<uint32_t> segmentation_staging(pixels.size());
            {
                rlt::Tensor<typename decltype(env.renderer->segmentation_buffer)::SPEC> segmentation_alias;
                segmentation_alias._data = segmentation_staging.data();
                rlt::copy(env.renderer->device, device, rlt::segmentation_buffer(device, *env.renderer), segmentation_alias);
            }
            for (size_t pixel_i = 0; pixel_i < pixels.size(); pixel_i++) {
                pixels[pixel_i] = rlt::rendering::raytracing::detail::segmentation_id_to_rgba(segmentation_staging[pixel_i]);
            }
        }
#else
        rlt::render(device, *env.renderer);
            {
                rlt::Tensor<typename decltype(env.renderer->frame_buffer)::SPEC> frame_alias;
                frame_alias._data = pixels.data();
                rlt::copy(env.renderer->device, device, rlt::frame_buffer(device, *env.renderer), frame_alias);
            }
#endif

        {
            const float forward[3] = {
                look_at[0] - eye[0],
                look_at[1] - eye[1],
                look_at[2] - eye[2]
            };
            const Quaternion q = l2f_camera_quaternion(forward, up);
            char line_flu_pos[128];
            char line_flu_quat[128];
            std::snprintf(line_flu_pos, sizeof(line_flu_pos), "Pos: (%.2f, %.2f, %.2f)", state.position[0], state.position[1], state.position[2]);
            std::snprintf(line_flu_quat, sizeof(line_flu_quat), "Quat: (%.3f, %.3f, %.3f, %.3f)", q.w, q.x, q.y, q.z);
            int overlay_x = 4;
            int overlay_y = 4;
            int max_len = std::max(std::strlen(line_flu_pos), std::strlen(line_flu_quat));
            int text_width = static_cast<int>(max_len) * 6 + 4;
            draw_overlay_background(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x, overlay_y, text_width, 20);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 2, line_flu_pos, 0xFF00FF00);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 11, line_flu_quat, 0xFF00FF00);
        }

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, CAM_WIDTH, CAM_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1,  1);
        glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();

        glfwSwapBuffers(window);
    }

    save_camera_state(cam_cache, state.position[0], state.position[1], state.position[2], g_input.yaw, g_input.pitch);
    const bool pose_recorded = write_camera_trace_json(
        options.record_camera_pose_path,
        resolved_scene_path,
        camera_trace
    );
    if(pose_recorded && !options.record_camera_pose_path.empty()) {
        std::cout << "Wrote camera pose trace JSON to " << options.record_camera_pose_path << std::endl;
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();

    rlt::free(device, camera_staging);
    rlt::free(device, env);
    return pose_recorded ? 0 : 1;
}
