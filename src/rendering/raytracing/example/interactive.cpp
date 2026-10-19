
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION 3
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW 4

#ifndef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"

#ifdef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_ENABLE_GLFW
#include "interactive/glfw.h"
#endif
#ifdef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_ENABLE_KITTY
#include "interactive/kitty.h"
#endif

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
namespace rro = rlt::rendering::raytracing::outputs;

static constexpr bool OUTPUT_FLOW = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW;
static constexpr bool OUTPUT_RGB = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB || RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD || OUTPUT_FLOW;
static constexpr bool OUTPUT_DEPTH = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD || RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH;
static constexpr bool OUTPUT_SEGMENTATION = RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION;
// flow mode renders at half the window width and shows RGB on the left and the selected flow view
// on the right; one orbiting, spinning cube overlay in front of the start pose adds object flow, a
// disocclusion band and a forward-direction target next to the camera's ego-motion flow
static constexpr unsigned int FLOW_NUM_OVERLAYS = OUTPUT_FLOW ? 1 : 0;
static constexpr unsigned int FLOW_HALVES = OUTPUT_FLOW ? 2 : 1;
static constexpr float FLOW_CUBE_HALF_EXTENT = 0.25f;
static constexpr float FLOW_CUBE_DISTANCE = 2.5f;
static constexpr float FLOW_CUBE_ORBIT_RADIUS = 0.6f;
static constexpr float FLOW_CUBE_ORBIT_RATE = 0.8f; // rad/s
static constexpr float FLOW_CUBE_SPIN_RATE = 1.5f;  // rad/s about z
static constexpr float FLOW_DISPLAY_SCALE_PX = 32.0f; // flow magnitude that saturates the hue display

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
    {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'l', {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}},
    {'w', {0x00,0x00,0x11,0x11,0x15,0x15,0x0A}},
    {'b', {0x10,0x10,0x16,0x19,0x11,0x11,0x1E}},
    {'c', {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E}},
    {'k', {0x10,0x10,0x12,0x14,0x18,0x14,0x12}},
    {'r', {0x00,0x00,0x16,0x19,0x10,0x10,0x10}},
    {'d', {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F}},
    {'f', {0x06,0x09,0x08,0x1C,0x08,0x08,0x08}},
    {'g', {0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E}},
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

#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
static bool g_show_depth = true;
#endif
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
enum class FlowView : int {
    BACKWARD = 0,
    BACKWARD_STATUS,
    FORWARD,
    FORWARD_STATUS,
    COUNT
};
static FlowView g_flow_view = FlowView::BACKWARD;
static const char* flow_view_label(FlowView view) {
    switch(view) {
        case FlowView::BACKWARD: return "Flow: backward";
        case FlowView::BACKWARD_STATUS: return "Flow: backward status";
        case FlowView::FORWARD: return "Flow: forward";
        default: return "Flow: forward status";
    }
}
static rlt::rendering::raytracing::Mesh make_flow_cube() {
    using namespace rlt::rendering::raytracing::constants;
    rlt::rendering::raytracing::Mesh mesh;
    mesh.color[0] = 0.85f;
    mesh.color[1] = 0.3f;
    mesh.color[2] = 0.1f;
    for(int vertex_i = 0; vertex_i < NUM_VERTICES; vertex_i++) {
        for(int dim_i = 0; dim_i < 3; dim_i++) {
            mesh.vertices.push_back(default_vertices[vertex_i][dim_i] * FLOW_CUBE_HALF_EXTENT);
        }
    }
    for(int triangle_i = 0; triangle_i < NUM_INDICES; triangle_i++) {
        for(int corner_i = 0; corner_i < 3; corner_i++) {
            mesh.indices.push_back(default_indices[triangle_i][corner_i]);
        }
    }
    return mesh;
}
static void flow_cube_transform(const float centre[3], float time_s, float transform[12]) {
    const float orbit = FLOW_CUBE_ORBIT_RATE * time_s;
    const float position[3] = {centre[0] + FLOW_CUBE_ORBIT_RADIUS * std::cos(orbit), centre[1] + FLOW_CUBE_ORBIT_RADIUS * std::sin(orbit), centre[2]};
    const float half_spin = FLOW_CUBE_SPIN_RATE * time_s / 2.0f;
    const float quaternion_wxyz[4] = {std::cos(half_spin), 0, 0, std::sin(half_spin)};
    rlt::make_transform(position, quaternion_wxyz, transform);
}
#endif

static constexpr char CAPTURED_POSE_PATH[] = "camera_poses.txt";

enum class Frontend {
    AUTOMATIC,
    WINDOW,
    KITTY
};

struct InteractiveOptions {
    std::string scene_arg;
    std::string record_camera_pose_path;
    float linear_velocity = 3.0f;
    Frontend frontend = Frontend::AUTOMATIC;
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
    std::cerr << "Usage: " << argv0 << " [conta:HASH | scene.glb] [--record-camera-pose trace.json] [--linear-velocity m/s] [--kitty | --window]" << std::endl;
    std::cerr << "  Without arguments the default scene is fetched via conta (downloaded into the cache if required)" << std::endl;
    std::cerr << "  Renders into the terminal (locally or over ssh) if it supports the kitty graphics and keyboard protocols and into a window otherwise; --kitty / --window force either" << std::endl;
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
        if(arg == "--kitty") {
            options.frontend = Frontend::KITTY;
        }
        else if(arg == "--window") {
            options.frontend = Frontend::WINDOW;
        }
        else if(arg == "--record-camera-pose" || arg == "--record-camera-trace") {
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

template <typename FRONTEND>
static int run(const InteractiveOptions& options) {
    using T = float;
    using TI = unsigned int;
    constexpr TI CAM_WIDTH = 1280;
    constexpr TI CAM_HEIGHT = 960;
    constexpr TI RENDER_WIDTH = CAM_WIDTH / FLOW_HALVES; // flow mode: each half of the window shows one full render
    constexpr TI NUM_ENVS = 1;
    using OUTPUTS = typename rro::Concat<
        rro::If<OUTPUT_RGB, rro::RGB<>>, rro::If<OUTPUT_DEPTH, rro::Depth<>>,
        rro::If<OUTPUT_SEGMENTATION, rro::Segmentation<>>,
        rro::If<OUTPUT_FLOW, rro::Flow<>>, rro::If<OUTPUT_FLOW, rro::FlowStatus<>>,
        rro::If<OUTPUT_FLOW, rro::Flow<rro::direction::Forward>>,
        rro::If<OUTPUT_FLOW, rro::FlowStatus<rro::direction::Forward>>, rro::List<rro::Probes>>::type;
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, RENDER_WIDTH, CAM_HEIGHT, 64, rlt::rendering::raytracing::High, false, 1, true, 2, OUTPUTS, FLOW_NUM_OVERLAYS, FLOW_NUM_OVERLAYS, FLOW_NUM_OVERLAYS>;
    // FOV is horizontal: a half-width render keeps the full view's vertical field and pixel density
    // by halving the tangent of the horizontal half-angle
    const T fov = OUTPUT_FLOW ? static_cast<T>(2) * std::atan(std::tan(rlt::rendering::degrees_to_radians(SPEC::FOV) / static_cast<T>(2)) / static_cast<T>(FLOW_HALVES)) * static_cast<T>(180.0 / M_PI) : SPEC::FOV;
    const T aspect = static_cast<T>(RENDER_WIDTH) / static_cast<T>(CAM_HEIGHT);
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;

    DEVICE device;
    rlt::init(device);

    static constexpr char DEFAULT_CONTA_HASH[] = "75d1d67a17ecd16bc3a214baa895240da8060df7";
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
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
    const auto flow_cube_asset = rlt::add(device, env.pool, make_flow_cube());
#endif
    rlt::init(device, env);
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
    std::cout << "RGBD target enabled. Press Tab to toggle RGB/depth display." << std::endl;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    std::cout << "Depth target enabled." << std::endl;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION
    std::cout << "Segmentation target enabled." << std::endl;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
    std::cout << "Flow target enabled: RGB on the left, flow on the right. Press Tab to cycle backward flow / backward status / forward flow / forward status (hue = direction, saturation = magnitude up to " << FLOW_DISPLAY_SCALE_PX << " px; status: white VALID, black MISS, magenta BEHIND, blue OUT_OF_FRAME, orange OCCLUDED)." << std::endl;
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

    FRONTEND frontend;
    if (!rlt::init(device, frontend, CAM_WIDTH, CAM_HEIGHT)) {
        return 1;
    }

    rlt::rendering::raytracing::interactive::InputState input;
    input.yaw = cached_yaw;
    input.pitch = cached_pitch;
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
    const float flow_cube_centre[3] = {state.position[0] + FLOW_CUBE_DISTANCE * std::cos(cached_yaw), state.position[1] + FLOW_CUBE_DISTANCE * std::sin(cached_yaw), state.position[2]};
    rlt::attach(device, *env.renderer, (TI)0, rlt::rendering::raytracing::OverlayIndex{0});
    float flow_cube_pose[12];
    flow_cube_transform(flow_cube_centre, 0.0f, flow_cube_pose);
    const auto flow_cube_placement = rlt::spawn(device, *env.renderer, rlt::rendering::raytracing::OverlayIndex{0}, flow_cube_asset, rlt::make_pose(flow_cube_pose));
    rlt::update(device, *env.renderer);
#endif

    std::vector<uint32_t> pixels(CAM_WIDTH * CAM_HEIGHT);
    std::vector<RecordedCameraPose> camera_trace;
    rlt::Tensor<typename decltype(env.renderer->cameras)::SPEC> camera_staging;
    rlt::malloc(device, camera_staging);
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
    {
        // publish the start pose as the current instant so the first advance_flow has a valid previous camera
        const T eye[3] = {state.position[0], state.position[1], state.position[2]};
        const T look_at[3] = {eye[0] + std::cos(input.yaw) * std::cos(input.pitch), eye[1] + std::sin(input.yaw) * std::cos(input.pitch), eye[2] + std::sin(input.pitch)};
        const T up[3] = {0, 0, 1};
        rlt::set(device, camera_staging, rlt::make_camera_data(eye, look_at, up, fov, aspect), 0);
        rlt::copy(device, env.renderer->device, camera_staging, rlt::cameras(device, *env.renderer));
    }
#endif

    constexpr float LINEAR_VELOCITY_RAMP_S = 0.5f;
    float linear_velocity_ramp_elapsed_s = 0.0f;
    auto last_time = std::chrono::steady_clock::now();
    const auto trace_start_time = last_time;
    size_t frame_index = 0;

    while (true) {
        rlt::poll(device, frontend, input);
        if (input.quit) {
            break;
        }
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (input.cycle_view) {
            input.cycle_view = false;
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
            g_show_depth = !g_show_depth;
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
            g_flow_view = static_cast<FlowView>((static_cast<int>(g_flow_view) + 1) % static_cast<int>(FlowView::COUNT));
            std::cout << flow_view_label(g_flow_view) << std::endl;
#endif
        }

        float cos_yaw = std::cos(input.yaw);
        float sin_yaw = std::sin(input.yaw);
        float dx = 0, dy = 0, dz = 0;
        if (input.forward)  { dx += cos_yaw; dy += sin_yaw; }
        if (input.backward) { dx -= cos_yaw; dy -= sin_yaw; }
        if (input.left)     { dx -= sin_yaw; dy += cos_yaw; }
        if (input.right)    { dx += sin_yaw; dy -= cos_yaw; }
        if (input.up)       { dz += 1; }
        if (input.down)     { dz -= 1; }
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
        state.yaw = input.yaw;

        T eye[3] = {state.position[0], state.position[1], state.position[2]};
        T look_at[3] = {
            eye[0] + std::cos(input.yaw) * std::cos(input.pitch),
            eye[1] + std::sin(input.yaw) * std::cos(input.pitch),
            eye[2] + std::sin(input.pitch)
        };
        T up[3] = {0, 0, 1};
        const auto camera = rlt::make_camera_data(eye, look_at, up, fov, aspect);

        if (input.capture_pose) {
            input.capture_pose = false;
            if (append_captured_pose(CAPTURED_POSE_PATH, eye, look_at, up, input.yaw, input.pitch)) {
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
            pose.yaw = input.yaw;
            pose.pitch = input.pitch;
            camera_trace.push_back(pose);
        }
        frame_index++;

#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
        rlt::advance_flow(device, *env.renderer); // last frame's camera and cube pose become the previous instant
#endif
        rlt::set(device, camera_staging, camera, 0);
        rlt::copy(device, env.renderer->device, camera_staging, rlt::cameras(device, *env.renderer));
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
        flow_cube_transform(flow_cube_centre, std::chrono::duration<float>(now - trace_start_time).count(), flow_cube_pose);
        rlt::set_pose(device, *env.renderer, rlt::rendering::raytracing::OverlayIndex{0}, flow_cube_placement, rlt::make_pose(flow_cube_pose));
        rlt::update(device, *env.renderer);
        rlt::render(device, *env.renderer);
        {
            constexpr size_t RENDER_PIXELS = (size_t)RENDER_WIDTH * CAM_HEIGHT;
            std::vector<uint32_t> rgb_staging(RENDER_PIXELS);
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::RGB<>{}))>::SPEC> frame_alias;
                frame_alias._data = rgb_staging.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::RGB<>{}), frame_alias);
            }
            std::vector<uint32_t> view_staging(RENDER_PIXELS);
            if (g_flow_view == FlowView::BACKWARD || g_flow_view == FlowView::FORWARD) {
                std::vector<float> flow_staging(RENDER_PIXELS * 2);
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::Flow<>{}))>::SPEC> flow_alias;
                flow_alias._data = flow_staging.data();
                if (g_flow_view == FlowView::BACKWARD) {
                    rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::Flow<>{}), flow_alias);
                }
                else {
                    rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::Flow<rro::direction::Forward>{}), flow_alias);
                }
                for (size_t pixel_i = 0; pixel_i < RENDER_PIXELS; pixel_i++) {
                    view_staging[pixel_i] = rlt::flow_color(device, flow_staging[pixel_i * 2 + 0], flow_staging[pixel_i * 2 + 1], FLOW_DISPLAY_SCALE_PX);
                }
            }
            else {
                std::vector<uint32_t> status_staging(RENDER_PIXELS);
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::FlowStatus<>{}))>::SPEC> status_alias;
                status_alias._data = status_staging.data();
                if (g_flow_view == FlowView::BACKWARD_STATUS) {
                    rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::FlowStatus<>{}), status_alias);
                }
                else {
                    rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::FlowStatus<rro::direction::Forward>{}), status_alias);
                }
                for (size_t pixel_i = 0; pixel_i < RENDER_PIXELS; pixel_i++) {
                    view_staging[pixel_i] = rlt::flow_status_color(device, status_staging[pixel_i]);
                }
            }
            for (size_t row = 0; row < CAM_HEIGHT; row++) {
                std::memcpy(pixels.data() + row * CAM_WIDTH, rgb_staging.data() + row * RENDER_WIDTH, RENDER_WIDTH * sizeof(uint32_t));
                std::memcpy(pixels.data() + row * CAM_WIDTH + RENDER_WIDTH, view_staging.data() + row * RENDER_WIDTH, RENDER_WIDTH * sizeof(uint32_t));
            }
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        rlt::render(device, *env.renderer);
        {
            std::vector<float> depth_staging(pixels.size());
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::Depth<>{}))>::SPEC> depth_alias;
                depth_alias._data = depth_staging.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::Depth<>{}), depth_alias);
            }
            const float max_depth = env.renderer->max_ray_length > 0 ? env.renderer->max_ray_length : 1e30f;
            depth_to_rgba(depth_staging.data(), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
        if (g_show_depth) {
            rlt::render(device, *env.renderer);
            std::vector<float> depth_staging(pixels.size());
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::Depth<>{}))>::SPEC> depth_alias;
                depth_alias._data = depth_staging.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::Depth<>{}), depth_alias);
            }
            const float max_depth = env.renderer->max_ray_length > 0 ? env.renderer->max_ray_length : 1e30f;
            depth_to_rgba(depth_staging.data(), pixels.data(), static_cast<int>(pixels.size()), max_depth);
        }
        else {
            rlt::render(device, *env.renderer);
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::RGB<>{}))>::SPEC> frame_alias;
                frame_alias._data = pixels.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::RGB<>{}), frame_alias);
            }
        }
#elif RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_SEGMENTATION
        rlt::render(device, *env.renderer);
        {
            std::vector<uint32_t> segmentation_staging(pixels.size());
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::Segmentation<>{}))>::SPEC> segmentation_alias;
                segmentation_alias._data = segmentation_staging.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::Segmentation<>{}), segmentation_alias);
            }
            for (size_t pixel_i = 0; pixel_i < pixels.size(); pixel_i++) {
                pixels[pixel_i] = rlt::segmentation_color(device, segmentation_staging[pixel_i]);
            }
        }
#else
        rlt::render(device, *env.renderer);
            {
                rlt::Tensor<typename rlt::utils::typing::remove_reference_t<decltype(rlt::output(device, *env.renderer, rro::RGB<>{}))>::SPEC> frame_alias;
                frame_alias._data = pixels.data();
                rlt::copy(env.renderer->device, device, rlt::output(device, *env.renderer, rro::RGB<>{}), frame_alias);
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
            int text_height = 20;
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
            const char* line_flow_view = flow_view_label(g_flow_view);
            text_width = std::max(text_width, static_cast<int>(std::strlen(line_flow_view)) * 6 + 4);
            text_height = 29;
#endif
            draw_overlay_background(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x, overlay_y, text_width, text_height);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 2, line_flu_pos, 0xFF00FF00);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 11, line_flu_quat, 0xFF00FF00);
#if RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_FLOW
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, overlay_x + 2, overlay_y + 20, line_flow_view, 0xFF00FF00);
            // label the right half as well so a screenshot of either half is self-describing
            draw_overlay_background(pixels.data(), CAM_WIDTH, CAM_HEIGHT, RENDER_WIDTH + overlay_x, overlay_y, static_cast<int>(std::strlen(line_flow_view)) * 6 + 4, 11);
            draw_string(pixels.data(), CAM_WIDTH, CAM_HEIGHT, RENDER_WIDTH + overlay_x + 2, overlay_y + 2, line_flow_view, 0xFF00FF00);
#endif
        }

        rlt::present(device, frontend, pixels.data());
    }
    rlt::free(device, frontend);

    save_camera_state(cam_cache, state.position[0], state.position[1], state.position[2], input.yaw, input.pitch);
    const bool pose_recorded = write_camera_trace_json(
        options.record_camera_pose_path,
        resolved_scene_path,
        camera_trace
    );
    if(pose_recorded && !options.record_camera_pose_path.empty()) {
        std::cout << "Wrote camera pose trace JSON to " << options.record_camera_pose_path << std::endl;
    }

    rlt::free(device, camera_staging);
    rlt::free(device, env);
    return pose_recorded ? 0 : 1;
}

static bool kitty_available() {
#ifdef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_ENABLE_KITTY
    return rlt::rendering::raytracing::interactive::kitty::available();
#else
    return false;
#endif
}

int main(int argc, char** argv) {
    InteractiveOptions options;
    if(!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    if(options.help) {
        print_usage(argv[0]);
        return 0;
    }
    if(options.frontend == Frontend::KITTY || (options.frontend == Frontend::AUTOMATIC && kitty_available())) {
#ifdef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_ENABLE_KITTY
        return run<rlt::rendering::raytracing::interactive::Kitty>(options);
#else
        std::cerr << "--kitty is unavailable: this build has no zlib or is not on a POSIX system" << std::endl;
        return 1;
#endif
    }
#ifdef RL_TOOLS_RENDERING_RAYTRACING_INTERACTIVE_ENABLE_GLFW
    return run<rlt::rendering::raytracing::interactive::Glfw>(options);
#else
    std::cerr << "This build has no window (GLFW/OpenGL not found): run it in a terminal that supports the kitty graphics and keyboard protocols" << std::endl;
    return 1;
#endif
}
