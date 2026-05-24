#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0

#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

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

static constexpr auto OUTPUT_MODE =
    RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        ? rlt::rendering::raytracing::OutputMode::DEPTH
        : (RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
            ? rlt::rendering::raytracing::OutputMode::RGBD
            : rlt::rendering::raytracing::OutputMode::RGB);

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

#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
static void depth_to_rgba(const float* depth, uint32_t* pixels, int count, float max_depth) {
    for (int i = 0; i < count; i++) {
        float normalized = std::fmin(std::fmax(depth[i] / max_depth, 0.0f), 1.0f);
        uint8_t value = static_cast<uint8_t>((1.0f - normalized) * 255.0f);
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

static Quaternion quaternion_from_yaw_pitch(float yaw, float pitch) {
    float half_yaw = yaw * 0.5f;
    float half_pitch = pitch * 0.5f;
    float cy = std::cos(half_yaw), sy = std::sin(half_yaw);
    float cp = std::cos(half_pitch), sp = std::sin(half_pitch);
    // Rotation order: yaw (around Y) then pitch (around Z-local, but here we use right-hand: pitch around X-local after yaw around Y)
    return {
        cy * cp,
        cy * sp,
        sy * cp,
        -sy * sp
    };
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
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
static bool g_show_depth = true;
#endif

struct InteractiveOptions {
    std::string scene_arg;
    std::string record_camera_pose_path;
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
    std::cerr << "Usage: " << argv0 << " [conta:HASH | scene.glb] [--record-camera-pose trace.json]" << std::endl;
    std::cerr << "  Or set CONTA_ROOT to use the default scene" << std::endl;
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
        const Quaternion q = quaternion_from_yaw_pitch(pose.yaw, pose.pitch);
        const float l2f_position[3] = {pose.position[0], pose.position[2], pose.position[1]};
        const Quaternion l2f_q{q.w, q.x, q.z, q.y};

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
        f << "      },\n";
        f << "      \"glb_scene_overlay\": {\n";
        f << "        \"position\": ";
        write_json_vec3(f, pose.position);
        f << ",\n";
        f << "        \"quaternion_wxyz\": ";
        write_json_quaternion(f, q);
        f << "\n";
        f << "      },\n";
        f << "      \"l2f_flu_overlay\": {\n";
        f << "        \"position\": ";
        write_json_vec3(f, l2f_position);
        f << ",\n";
        f << "        \"quaternion_wxyz\": ";
        write_json_quaternion(f, l2f_q);
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

static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        g_show_depth = !g_show_depth;
        return;
    }
#endif
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    if (key == GLFW_KEY_W) g_input.forward = pressed;
    if (key == GLFW_KEY_S) g_input.backward = pressed;
    if (key == GLFW_KEY_A) g_input.left = pressed;
    if (key == GLFW_KEY_D) g_input.right = pressed;
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
    constexpr TI CAM_WIDTH = 640;
    constexpr TI CAM_HEIGHT = 480;
    constexpr TI NUM_ENVS = 1;
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, 64, rlt::rendering::raytracing::HighFidelityShading, false, 1, false, 1, OUTPUT_MODE>;
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
    if (!options.scene_arg.empty()) {
        const char* scene_arg = options.scene_arg.c_str();
        if (std::strncmp(scene_arg, "conta:", 6) == 0) {
            const char* hash_str = scene_arg + 6;
            const char* conta_root = std::getenv("CONTA_ROOT");
            if (!conta_root) {
                std::cerr << "CONTA_ROOT environment variable is not set" << std::endl;
                return 1;
            }
            resolved_scene_path = std::string(conta_root) + "/data/" + hash_str;
        } else {
            resolved_scene_path = scene_arg;
        }
    } else {
        const char* conta_root = std::getenv("CONTA_ROOT");
        if (conta_root) {
            resolved_scene_path = std::string(conta_root) + "/data/" + DEFAULT_CONTA_HASH;
            std::cout << "No scene argument given, using default: conta:" << DEFAULT_CONTA_HASH << std::endl;
        } else {
            print_usage(argv[0]);
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

    constexpr float MOVE_SPEED = 3.0f;
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
            float speed = MOVE_SPEED * dt / move_len;
            state.position[0] += dx * speed;
            state.position[1] += dy * speed;
            state.position[2] += dz * speed;
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
        rlt::set(device, env.renderer->cameras, rlt::make_camera_data(eye, look_at, up, SPEC::RAYTRACING_SPEC::COS_FOVY, aspect), static_cast<TI>(0));

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

        rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        rlt::render_depth_only(device, *env.renderer);
        rlt::read_depth_buffer(device, *env.renderer, env.renderer->depth_buffer);
        {
            const float max_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            depth_to_rgba(rlt::data(env.renderer->depth_buffer), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
        if (g_show_depth) {
            rlt::render_depth_only(device, *env.renderer);
            rlt::read_depth_buffer(device, *env.renderer, env.renderer->depth_buffer);
            const float max_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            depth_to_rgba(rlt::data(env.renderer->depth_buffer), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
        else {
            rlt::render_rgb_only(device, *env.renderer);
            rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);
            const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
            std::memcpy(pixels.data(), fb_data, pixels.size() * sizeof(uint32_t));
        }
#else
        rlt::render_rgb_only(device, *env.renderer);
        rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);
        const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
        std::memcpy(pixels.data(), fb_data, pixels.size() * sizeof(uint32_t));
#endif

        {
            Quaternion q = quaternion_from_yaw_pitch(g_input.yaw, g_input.pitch);
            // GLB/Scene: X=forward, Y=up, Z=left (Y-up)
            char line_glb_pos[128];
            char line_glb_quat[128];
            std::snprintf(line_glb_pos, sizeof(line_glb_pos), "GLB  Pos: (%.2f, %.2f, %.2f)", state.position[0], state.position[1], state.position[2]);
            std::snprintf(line_glb_quat, sizeof(line_glb_quat), "GLB Quat: (%.3f, %.3f, %.3f, %.3f)", q.w, q.x, q.y, q.z);
            // L2F/RLtools: X=forward, Y=left, Z=up (FLU) — Y/Z swap from GLB
            char line_l2f_pos[128];
            char line_l2f_quat[128];
            std::snprintf(line_l2f_pos, sizeof(line_l2f_pos), "L2F  Pos: (%.2f, %.2f, %.2f)", state.position[0], state.position[2], state.position[1]);
            std::snprintf(line_l2f_quat, sizeof(line_l2f_quat), "L2F Quat: (%.3f, %.3f, %.3f, %.3f)", q.w, q.x, q.z, q.y);
            int overlay_x = 4;
            int overlay_y = 4;
            int max_len = std::max({std::strlen(line_glb_pos), std::strlen(line_glb_quat), std::strlen(line_l2f_pos), std::strlen(line_l2f_quat)});
            int text_width = static_cast<int>(max_len) * 6 + 4;
            draw_overlay_background(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x, overlay_y, text_width, 38);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 2, line_glb_pos, 0xFF00FF00);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 11, line_glb_quat, 0xFF00FF00);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 20, line_l2f_pos, 0xFF88CCFF);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 29, line_l2f_quat, 0xFF88CCFF);
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

    rlt::free(device, env);
    return pose_recorded ? 0 : 1;
}
