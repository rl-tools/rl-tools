#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace rlt = rl_tools;
using json = nlohmann::json;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;

static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;
static constexpr int FRAME_JPEG_QUALITY = 90;
static constexpr TI MIN_VIDEO_DIMENSION = 2048;
static constexpr double DEFAULT_FOV_DEG = 80.0;
static constexpr double DEFAULT_SMOOTH_POSITION_SIGMA_S = 1.0;
static constexpr double DEFAULT_SMOOTH_ORIENTATION_SIGMA_S = 6.0;
static constexpr double DEFAULT_MAX_ORIENTATION_SPEED_RAD_S = 1.6;
static constexpr double DEFAULT_ORIENTATION_JUMP_RAMP_MULTIPLIER = 5.0;

struct RenderResolutionOption {
    const char* name;
    TI width;
    TI height;
    bool enabled_by_default;
    const char* alias = nullptr;
};

enum class AntiAliasingSelection {
    NONE,
    AA2,
    BOTH
};

static constexpr RenderResolutionOption RENDER_RESOLUTIONS[] = {
    {"60", 60, 34, true},
    {"120", 120, 68, true},
    {"240", 240, 135, true},
    {"480", 480, 270, true},
    {"960", 960, 540, true},
    {"1920", 1920, 1080, true},
    {"3840", 3840, 2160, true, "4k"}
};

struct TracePose {
    T eye[3];
    T look_at[3];
    T up[3];
    double yaw = 0.0;
    double pitch = 0.0;
    double timestamp_s = 0.0;
    bool has_timestamp = false;
    bool has_yaw_pitch = false;
};

struct Options {
    std::string trace_path;
    std::string scene_path;
    std::string output_dir = ".";
    std::string ffmpeg = "ffmpeg";
    std::string settings = "very_high";
    int resolution_width = 0;
    int resolution_height = 0;
    int fps = 60;
    int max_frames = 0;
    double fov_deg = DEFAULT_FOV_DEG;
    AntiAliasingSelection aa = AntiAliasingSelection::BOTH;
    double smooth_position_sigma_s = DEFAULT_SMOOTH_POSITION_SIGMA_S;
    double smooth_orientation_sigma_s = DEFAULT_SMOOTH_ORIENTATION_SIGMA_S;
    double max_orientation_speed_rad_s = DEFAULT_MAX_ORIENTATION_SPEED_RAD_S;
    double orientation_jump_ramp_multiplier = DEFAULT_ORIENTATION_JUMP_RAMP_MULTIPLIER;
    bool write_frames = false;
    bool help = false;
};

struct RenderRecord {
    std::string name;
    std::string output;
    std::string profile;
    std::string anti_aliasing;
    std::string path;
    std::string frames_dir;
    std::string frame_pattern;
    std::string frame_format;
    std::string ffmpeg_command;
    TI width = 0;
    TI height = 0;
    TI video_width = 0;
    TI video_height = 0;
    TI video_oversampling_factor = 1;
    TI aa_grid_size = 1;
    TI samples_per_pixel = 1;
    size_t frame_count = 0;
    int frame_jpeg_quality = 0;
    int ffmpeg_status = 0;
    bool ok = false;
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --trace pose_trace.json [options]\n"
        << "Options:\n"
        << "  --trace <path>             Camera pose JSON or trace JSON\n"
        << "  --scene <path>             Scene path override\n"
        << "  --output-dir <dir>         Parent output directory; renders go under <dir>/<trace-name>/ (default: .)\n"
        << "  --fps <n>                  MP4 frame rate; timestamped traces are resampled to this rate (default: 60)\n"
        << "  --fov <deg>                Horizontal FOV in degrees (default: 80)\n"
        << "  --ffmpeg <path>            ffmpeg binary (default: ffmpeg)\n"
        << "  --settings <list>          all, rgb, depth, very_high, or comma list; depth has no rendering profile\n"
        << "                              medium, high, and low are temporarily disabled\n"
        << "                              Legacy aliases: very_high_fidelity, basic, high_fidelity, fast_flat\n"
        << "  --resolution <name>        Render one resolution: 60, 120, 240, 480, 960, 1920, 3840, or 4k\n"
        << "                              4k is an alias for UHD 3840x2160\n"
        << "                              By default all listed resolutions are rendered\n"
        << "                              Encoded videos are upscaled to at least 2048 pixels per axis\n"
        << "  --aa <mode>                Anti-aliasing mode: none, aa2, or both (default: both)\n"
        << "  --frames                   Write per-frame PNG/JPEG image sequences\n"
        << "                              Frames are written under <output-dir>/<trace-name>/frames/<render-name>/\n"
        << "                              60-wide and 120-wide frames are PNG; larger frames are JPEG\n"
        << "  --no-frames                Do not write per-frame image sequences (default)\n"
        << "  --max-frames <n>           Limit trace frames when >0\n"
        << "  --smooth-sigma-s <s>       Gaussian smoothing sigma for position and orientation (use 0 to disable)\n"
        << "  --smooth-position-sigma-s <s>\n"
        << "                              Gaussian smoothing sigma for position only (default: 1)\n"
        << "  --smooth-orientation-sigma-s <s>\n"
        << "                              Gaussian smoothing sigma for orientation only (default: 6)\n"
        << "  --max-orientation-speed-deg-s <deg/s>\n"
        << "                              Detect orientation jumps before smoothing (default: 91.7; use 0 to disable)\n"
        << "  --orientation-jump-ramp-multiplier <x>\n"
        << "                              Lengthen detected jump slerp ramps by this factor (default: 5)\n";
}

static bool parse_int(const std::string& value, int& out) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if(end == value.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static bool parse_resolution(const std::string& value, int& width, int& height) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    for(const RenderResolutionOption& resolution : RENDER_RESOLUTIONS) {
        if(lower == resolution.name || (resolution.alias != nullptr && lower == resolution.alias)) {
            width = static_cast<int>(resolution.width);
            height = static_cast<int>(resolution.height);
            return true;
        }
    }

    int requested_width = 0;
    if(!parse_int(value, requested_width) || requested_width <= 0) {
        return false;
    }
    for(const RenderResolutionOption& resolution : RENDER_RESOLUTIONS) {
        if(static_cast<TI>(requested_width) == resolution.width) {
            width = static_cast<int>(resolution.width);
            height = static_cast<int>(resolution.height);
            return true;
        }
    }
    return false;
}

static bool should_render_resolution(const Options& options, TI width, TI height) {
    if(options.resolution_width != 0 || options.resolution_height != 0) {
        return static_cast<TI>(options.resolution_width) == width && static_cast<TI>(options.resolution_height) == height;
    }
    for(const RenderResolutionOption& resolution : RENDER_RESOLUTIONS) {
        if(width == resolution.width && height == resolution.height) {
            return resolution.enabled_by_default;
        }
    }
    return false;
}

static const char* aa_selection_name(AntiAliasingSelection aa) {
    switch(aa) {
        case AntiAliasingSelection::NONE: return "none";
        case AntiAliasingSelection::AA2: return "aa2";
        case AntiAliasingSelection::BOTH: return "both";
    }
    return "both";
}

static bool parse_aa_selection(const std::string& value, AntiAliasingSelection& out) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if(lower == "none" || lower == "off" || lower == "no") {
        out = AntiAliasingSelection::NONE;
        return true;
    }
    if(lower == "aa2" || lower == "2" || lower == "on") {
        out = AntiAliasingSelection::AA2;
        return true;
    }
    if(lower == "both" || lower == "all") {
        out = AntiAliasingSelection::BOTH;
        return true;
    }
    return false;
}

static bool should_render_aa(const Options& options, bool enabled, TI grid_size) {
    switch(options.aa) {
        case AntiAliasingSelection::NONE:
            return !enabled;
        case AntiAliasingSelection::AA2:
            return enabled && grid_size == static_cast<TI>(2);
        case AntiAliasingSelection::BOTH:
            return true;
    }
    return true;
}

static bool parse_double(const std::string& value, double& out) {
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if(end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

static double degrees_to_radians(double degrees) {
    return degrees * 3.14159265358979323846 / 180.0;
}

static bool option_value(int& i, int argc, char** argv, const std::string& arg, const char* name, std::string& out) {
    const std::string prefix = std::string(name) + "=";
    if(arg.compare(0, prefix.size(), prefix) == 0) {
        out = arg.substr(prefix.size());
        return true;
    }
    if(arg == name) {
        if(i + 1 >= argc) {
            std::cerr << "Missing value for " << name << std::endl;
            return false;
        }
        out = argv[++i];
        return true;
    }
    return false;
}

static bool parse_options(int argc, char** argv, Options& options) {
    for(int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        std::string value;
        if(arg == "-h" || arg == "--help") {
            options.help = true;
            return true;
        }
        else if(option_value(i, argc, argv, arg, "--trace", value)) {
            options.trace_path = value;
        }
        else if(option_value(i, argc, argv, arg, "--scene", value)) {
            options.scene_path = value;
        }
        else if(option_value(i, argc, argv, arg, "--output-dir", value)) {
            options.output_dir = value;
        }
        else if(option_value(i, argc, argv, arg, "--ffmpeg", value)) {
            options.ffmpeg = value;
        }
        else if(option_value(i, argc, argv, arg, "--settings", value)) {
            options.settings = value;
        }
        else if(option_value(i, argc, argv, arg, "--resolution", value)) {
            if(!parse_resolution(value, options.resolution_width, options.resolution_height)) {
                std::cerr << "Invalid --resolution: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--aa", value)) {
            if(!parse_aa_selection(value, options.aa)) {
                std::cerr << "Invalid --aa: " << value << " (expected none, aa2, or both)" << std::endl;
                return false;
            }
        }
        else if(arg == "--no-frames") {
            options.write_frames = false;
        }
        else if(arg == "--frames") {
            options.write_frames = true;
        }
        else if(option_value(i, argc, argv, arg, "--fps", value)) {
            if(!parse_int(value, options.fps) || options.fps <= 0) {
                std::cerr << "Invalid --fps: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--fov", value) || option_value(i, argc, argv, arg, "--fov-deg", value)) {
            if(!parse_double(value, options.fov_deg) || options.fov_deg <= 0.0 || options.fov_deg >= 180.0) {
                std::cerr << "Invalid --fov: " << value << " (expected degrees in (0, 180))" << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--max-frames", value)) {
            if(!parse_int(value, options.max_frames) || options.max_frames < 0) {
                std::cerr << "Invalid --max-frames: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--smooth-sigma-s", value)) {
            double sigma_s = 0.0;
            if(!parse_double(value, sigma_s) || sigma_s < 0.0) {
                std::cerr << "Invalid --smooth-sigma-s: " << value << std::endl;
                return false;
            }
            options.smooth_position_sigma_s = sigma_s;
            options.smooth_orientation_sigma_s = sigma_s;
        }
        else if(option_value(i, argc, argv, arg, "--smooth-position-sigma-s", value)) {
            if(!parse_double(value, options.smooth_position_sigma_s) || options.smooth_position_sigma_s < 0.0) {
                std::cerr << "Invalid --smooth-position-sigma-s: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--smooth-orientation-sigma-s", value)) {
            if(!parse_double(value, options.smooth_orientation_sigma_s) || options.smooth_orientation_sigma_s < 0.0) {
                std::cerr << "Invalid --smooth-orientation-sigma-s: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--max-orientation-speed-deg-s", value)) {
            double max_speed_deg_s = 0.0;
            if(!parse_double(value, max_speed_deg_s) || max_speed_deg_s < 0.0) {
                std::cerr << "Invalid --max-orientation-speed-deg-s: " << value << std::endl;
                return false;
            }
            options.max_orientation_speed_rad_s = max_speed_deg_s * 3.14159265358979323846 / 180.0;
        }
        else if(option_value(i, argc, argv, arg, "--orientation-jump-ramp-multiplier", value)) {
            if(!parse_double(value, options.orientation_jump_ramp_multiplier) || options.orientation_jump_ramp_multiplier < 1.0) {
                std::cerr << "Invalid --orientation-jump-ramp-multiplier: " << value << std::endl;
                return false;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    if(options.trace_path.empty()) {
        std::cerr << "--trace is required" << std::endl;
        return false;
    }
    return true;
}

static bool mkdir_p(const std::string& path) {
    if(path.empty()) {
        return true;
    }
    std::string current;
    size_t i = 0;
    if(path[0] == '/') {
        current = "/";
        i = 1;
    }
    while(i <= path.size()) {
        const size_t next = path.find('/', i);
        std::string part = path.substr(i, next == std::string::npos ? std::string::npos : next - i);
        if(!part.empty()) {
            if(!current.empty() && current.back() != '/') {
                current += "/";
            }
            current += part;
            if(mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                std::cerr << "Failed to create directory " << current << ": " << std::strerror(errno) << std::endl;
                return false;
            }
        }
        if(next == std::string::npos) {
            break;
        }
        i = next + 1;
    }
    return true;
}

static std::string join_path(const std::string& a, const std::string& b) {
    if(a.empty() || a == ".") {
        return b;
    }
    if(a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

static std::string path_basename(const std::string& path) {
    const size_t end = path.find_last_not_of("/\\");
    if(end == std::string::npos) {
        return std::string();
    }
    const size_t begin = path.find_last_of("/\\", end);
    if(begin == std::string::npos) {
        return path.substr(0, end + 1);
    }
    return path.substr(begin + 1, end - begin);
}

static std::string strip_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if(dot == std::string::npos || dot == 0) {
        return path;
    }
    return path.substr(0, dot);
}

static std::string sanitize_path_component(const std::string& value) {
    std::string out;
    bool wrote_separator = false;
    for(unsigned char c : value) {
        if(std::isalnum(c) || c == '-' || c == '_' || c == '.') {
            out += static_cast<char>(c);
            wrote_separator = false;
        }
        else if(!wrote_separator) {
            out += '_';
            wrote_separator = true;
        }
    }
    while(!out.empty() && out[0] == '.') {
        out.erase(out.begin());
    }
    while(!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out.empty() ? std::string("trace") : out;
}

static std::string trace_output_name(const std::string& trace_path) {
    const std::string basename = path_basename(trace_path);
    const std::string stem = strip_extension(basename);
    return sanitize_path_component(stem.empty() ? basename : stem);
}

static std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for(char c : value) {
        if(c == '\'') {
            quoted += "'\\''";
        }
        else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static std::vector<std::string> split_settings(const std::string& settings) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(settings);
    while(std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c){ return std::isspace(c); }), token.end());
        if(!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

static bool contains_token(const std::vector<std::string>& tokens, const std::string& value) {
    return std::find(tokens.begin(), tokens.end(), value) != tokens.end();
}

static bool contains_profile_token(const std::vector<std::string>& tokens, const std::string& profile) {
    return contains_token(tokens, profile)
        || (profile == "very_high" &&
            (contains_token(tokens, "very_high_fidelity") ||
             contains_token(tokens, "veryhigh") ||
             contains_token(tokens, "veryhigh_fidelity")))
        || (profile == "medium" && contains_token(tokens, "basic"))
        || (profile == "high" && contains_token(tokens, "high_fidelity"))
        || (profile == "low" && contains_token(tokens, "fast_flat"));
}

static bool has_profile_filter(const std::vector<std::string>& tokens) {
    return contains_token(tokens, "very_high") || contains_token(tokens, "very_high_fidelity")
        || contains_token(tokens, "veryhigh") || contains_token(tokens, "veryhigh_fidelity")
        || contains_token(tokens, "medium") || contains_token(tokens, "basic")
        || contains_token(tokens, "high") || contains_token(tokens, "high_fidelity")
        || contains_token(tokens, "low") || contains_token(tokens, "fast_flat");
}

static bool should_render_setting(const Options& options, const char* output, const char* profile) {
    const std::vector<std::string> tokens = split_settings(options.settings);
    if(tokens.empty() || contains_token(tokens, "all")) {
        return true;
    }
    const bool has_output_filter = contains_token(tokens, "rgb") || contains_token(tokens, "depth");
    const bool has_profile_filter_value = has_profile_filter(tokens);
    const bool output_ok = !has_output_filter || contains_token(tokens, output);
    const bool profile_ok = !has_profile_filter_value || contains_profile_token(tokens, profile);
    return output_ok && profile_ok;
}

static bool should_render_depth(const Options& options) {
    const std::vector<std::string> tokens = split_settings(options.settings);
    return tokens.empty() || contains_token(tokens, "all") || contains_token(tokens, "depth");
}

static bool read_json_file(const std::string& path, json& out) {
    std::ifstream f(path);
    if(!f) {
        std::cerr << "Failed to open trace JSON: " << path << std::endl;
        return false;
    }
    try {
        f >> out;
    }
    catch(const std::exception& e) {
        std::cerr << "Failed to parse trace JSON: " << e.what() << std::endl;
        return false;
    }
    return true;
}

static bool read_vec3(const json& value, T out[3]) {
    if(!value.is_array() || value.size() != 3) {
        return false;
    }
    for(int i = 0; i < 3; i++) {
        if(!value[i].is_number()) {
            return false;
        }
        out[i] = value[i].get<T>();
    }
    return true;
}

static void normalize(T v[3]) {
    const T norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if(norm < static_cast<T>(1e-6)) {
        v[0] = static_cast<T>(1);
        v[1] = static_cast<T>(0);
        v[2] = static_cast<T>(0);
        return;
    }
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
}

static double wrap_angle_delta(double delta) {
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    delta = std::fmod(delta + 3.14159265358979323846, two_pi);
    if(delta < 0.0) {
        delta += two_pi;
    }
    return delta - 3.14159265358979323846;
}

static void set_pose_yaw_pitch(TracePose& pose, double yaw, double pitch) {
    pose.yaw = yaw;
    pose.pitch = pitch;
    pose.has_yaw_pitch = true;
    pose.up[0] = static_cast<T>(0);
    pose.up[1] = static_cast<T>(0);
    pose.up[2] = static_cast<T>(1);
    pose.look_at[0] = pose.eye[0] + static_cast<T>(std::cos(yaw) * std::cos(pitch));
    pose.look_at[1] = pose.eye[1] + static_cast<T>(std::sin(yaw) * std::cos(pitch));
    pose.look_at[2] = pose.eye[2] + static_cast<T>(std::sin(pitch));
}

static bool parse_pose(const json& pose_json, TracePose& pose) {
    const json* source = &pose_json;
    pose.timestamp_s = 0.0;
    pose.has_timestamp = false;
    pose.has_yaw_pitch = false;
    if(pose_json.contains("timestamp_s") && pose_json["timestamp_s"].is_number()) {
        pose.timestamp_s = pose_json["timestamp_s"].get<double>();
        pose.has_timestamp = true;
    }
    if(pose_json.contains("renderer") && pose_json["renderer"].is_object()) {
        source = &pose_json["renderer"];
    }
    if(!pose.has_timestamp && source->contains("timestamp_s") && (*source)["timestamp_s"].is_number()) {
        pose.timestamp_s = (*source)["timestamp_s"].get<double>();
        pose.has_timestamp = true;
    }
    if(!source->contains("position") || !read_vec3((*source)["position"], pose.eye)) {
        return false;
    }
    if(source->contains("yaw") && source->contains("pitch") && (*source)["yaw"].is_number() && (*source)["pitch"].is_number()) {
        pose.yaw = (*source)["yaw"].get<double>();
        pose.pitch = (*source)["pitch"].get<double>();
        pose.has_yaw_pitch = true;
    }
    if(source->contains("up")) {
        if(!read_vec3((*source)["up"], pose.up)) {
            return false;
        }
    }
    else {
        pose.up[0] = static_cast<T>(0);
        pose.up[1] = static_cast<T>(0);
        pose.up[2] = static_cast<T>(1);
    }
    if(source->contains("look_at")) {
        return read_vec3((*source)["look_at"], pose.look_at);
    }
    if(source->contains("forward")) {
        T forward[3];
        if(!read_vec3((*source)["forward"], forward)) {
            return false;
        }
        normalize(forward);
        pose.look_at[0] = pose.eye[0] + forward[0];
        pose.look_at[1] = pose.eye[1] + forward[1];
        pose.look_at[2] = pose.eye[2] + forward[2];
        return true;
    }
    if(source->contains("yaw") && source->contains("pitch") && (*source)["yaw"].is_number() && (*source)["pitch"].is_number()) {
        set_pose_yaw_pitch(pose, pose.yaw, pose.pitch);
        return true;
    }
    return false;
}

static bool load_trace(const Options& options, std::string& scene_path, std::vector<TracePose>& poses) {
    json trace_json;
    if(!read_json_file(options.trace_path, trace_json)) {
        return false;
    }
    scene_path = options.scene_path;
    if(scene_path.empty() && trace_json.contains("scene_path") && trace_json["scene_path"].is_string()) {
        scene_path = trace_json["scene_path"].get<std::string>();
    }
    if(scene_path.empty()) {
        std::cerr << "Trace JSON has no scene_path; pass --scene <path>" << std::endl;
        return false;
    }

    if(trace_json.contains("poses") && trace_json["poses"].is_array()) {
        for(const json& item : trace_json["poses"]) {
            TracePose pose;
            if(!parse_pose(item, pose)) {
                std::cerr << "Invalid pose entry in trace JSON" << std::endl;
                return false;
            }
            poses.push_back(pose);
            if(options.max_frames > 0 && static_cast<int>(poses.size()) >= options.max_frames) {
                break;
            }
        }
    }
    else {
        TracePose pose;
        if(!parse_pose(trace_json, pose)) {
            std::cerr << "Trace JSON is neither a valid single pose nor a trace with poses[]" << std::endl;
            return false;
        }
        poses.push_back(pose);
    }
    if(poses.empty()) {
        std::cerr << "Trace contains no poses" << std::endl;
        return false;
    }
    return true;
}

static T lerp_scalar(T a, T b, double t) {
    return static_cast<T>(static_cast<double>(a) + (static_cast<double>(b) - static_cast<double>(a)) * t);
}

static void lerp_vec3(const T a[3], const T b[3], double t, T out[3]) {
    out[0] = lerp_scalar(a[0], b[0], t);
    out[1] = lerp_scalar(a[1], b[1], t);
    out[2] = lerp_scalar(a[2], b[2], t);
}

static void pose_forward(const TracePose& pose, T out[3]) {
    out[0] = pose.look_at[0] - pose.eye[0];
    out[1] = pose.look_at[1] - pose.eye[1];
    out[2] = pose.look_at[2] - pose.eye[2];
    normalize(out);
}

static double dot_vec3(const T a[3], const T b[3]) {
    return static_cast<double>(a[0]) * static_cast<double>(b[0])
         + static_cast<double>(a[1]) * static_cast<double>(b[1])
         + static_cast<double>(a[2]) * static_cast<double>(b[2]);
}

static bool normalize_vec3_checked(T v[3]) {
    const double norm = std::sqrt(dot_vec3(v, v));
    if(norm < 1e-9) {
        return false;
    }
    v[0] = static_cast<T>(static_cast<double>(v[0]) / norm);
    v[1] = static_cast<T>(static_cast<double>(v[1]) / norm);
    v[2] = static_cast<T>(static_cast<double>(v[2]) / norm);
    return true;
}

static void slerp_unit_vec3(const T a_in[3], const T b_in[3], double t, T out[3]) {
    T a[3] = {a_in[0], a_in[1], a_in[2]};
    T b[3] = {b_in[0], b_in[1], b_in[2]};
    if(!normalize_vec3_checked(a)) {
        a[0] = static_cast<T>(1);
        a[1] = static_cast<T>(0);
        a[2] = static_cast<T>(0);
    }
    if(!normalize_vec3_checked(b)) {
        b[0] = a[0];
        b[1] = a[1];
        b[2] = a[2];
    }

    double d = dot_vec3(a, b);
    d = std::min(std::max(d, -1.0), 1.0);
    if(d > 0.9995) {
        lerp_vec3(a, b, t, out);
        normalize(out);
        return;
    }
    if(d < -0.9995) {
        T ortho[3] = {-a[1], a[0], static_cast<T>(0)};
        if(!normalize_vec3_checked(ortho)) {
            ortho[0] = -a[2];
            ortho[1] = static_cast<T>(0);
            ortho[2] = a[0];
            normalize_vec3_checked(ortho);
        }
        const double angle = 3.14159265358979323846 * t;
        const double ca = std::cos(angle);
        const double sa = std::sin(angle);
        for(int i = 0; i < 3; i++) {
            out[i] = static_cast<T>(ca * static_cast<double>(a[i]) + sa * static_cast<double>(ortho[i]));
        }
        normalize(out);
        return;
    }

    const double theta = std::acos(d);
    const double sin_theta = std::sin(theta);
    const double wa = std::sin((1.0 - t) * theta) / sin_theta;
    const double wb = std::sin(t * theta) / sin_theta;
    for(int i = 0; i < 3; i++) {
        out[i] = static_cast<T>(wa * static_cast<double>(a[i]) + wb * static_cast<double>(b[i]));
    }
    normalize(out);
}

static void orthonormalize_up_for_forward(const T forward[3], const T preferred_up[3], T up[3]) {
    up[0] = preferred_up[0];
    up[1] = preferred_up[1];
    up[2] = preferred_up[2];
    const double projection = dot_vec3(up, forward);
    for(int i = 0; i < 3; i++) {
        up[i] = static_cast<T>(static_cast<double>(up[i]) - projection * static_cast<double>(forward[i]));
    }
    if(normalize_vec3_checked(up)) {
        return;
    }
    T fallback[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
    if(std::fabs(dot_vec3(fallback, forward)) > 0.95) {
        fallback[0] = static_cast<T>(0);
        fallback[1] = static_cast<T>(1);
        fallback[2] = static_cast<T>(0);
    }
    const double fallback_projection = dot_vec3(fallback, forward);
    for(int i = 0; i < 3; i++) {
        up[i] = static_cast<T>(static_cast<double>(fallback[i]) - fallback_projection * static_cast<double>(forward[i]));
    }
    normalize(up);
}

static TracePose interpolate_pose(const TracePose& a, const TracePose& b, double timestamp_s) {
    const double duration = b.timestamp_s - a.timestamp_s;
    const double t = duration > 0.0 ? std::min(std::max((timestamp_s - a.timestamp_s) / duration, 0.0), 1.0) : 0.0;

    TracePose out;
    lerp_vec3(a.eye, b.eye, t, out.eye);

    if(a.has_yaw_pitch && b.has_yaw_pitch) {
        const double yaw = a.yaw + wrap_angle_delta(b.yaw - a.yaw) * t;
        const double pitch = static_cast<double>(a.pitch) + (static_cast<double>(b.pitch) - static_cast<double>(a.pitch)) * t;
        set_pose_yaw_pitch(out, yaw, pitch);
        out.timestamp_s = timestamp_s;
        out.has_timestamp = true;
        return out;
    }

    T forward_a[3];
    T forward_b[3];
    T forward[3];
    pose_forward(a, forward_a);
    pose_forward(b, forward_b);
    slerp_unit_vec3(forward_a, forward_b, t, forward);

    T up_preferred[3];
    slerp_unit_vec3(a.up, b.up, t, up_preferred);
    orthonormalize_up_for_forward(forward, up_preferred, out.up);

    out.look_at[0] = out.eye[0] + forward[0];
    out.look_at[1] = out.eye[1] + forward[1];
    out.look_at[2] = out.eye[2] + forward[2];
    out.timestamp_s = timestamp_s;
    out.has_timestamp = true;
    return out;
}

static bool trace_timestamp_stats(const std::vector<TracePose>& poses, double& duration_s, double& implied_fps) {
    duration_s = 0.0;
    implied_fps = 0.0;
    if(poses.size() < 2) {
        return false;
    }
    for(size_t i = 0; i < poses.size(); i++) {
        if(!poses[i].has_timestamp || !std::isfinite(poses[i].timestamp_s)) {
            return false;
        }
        if(i > 0 && poses[i].timestamp_s <= poses[i - 1].timestamp_s) {
            return false;
        }
    }
    duration_s = poses.back().timestamp_s - poses.front().timestamp_s;
    if(duration_s <= 0.0) {
        return false;
    }
    implied_fps = static_cast<double>(poses.size() - 1) / duration_s;
    return true;
}

static std::vector<TracePose> resample_trace(const std::vector<TracePose>& source, int fps, double& duration_s, double& implied_fps, bool& timestamp_resampled) {
    timestamp_resampled = false;
    if(!trace_timestamp_stats(source, duration_s, implied_fps)) {
        return source;
    }

    const size_t output_frames = std::max<size_t>(1, static_cast<size_t>(std::ceil(duration_s * static_cast<double>(fps))));
    std::vector<TracePose> resampled;
    resampled.reserve(output_frames);

    size_t segment = 0;
    const double start_s = source.front().timestamp_s;
    for(size_t frame_i = 0; frame_i < output_frames; frame_i++) {
        const double target_s = std::min(start_s + static_cast<double>(frame_i) / static_cast<double>(fps), source.back().timestamp_s);
        while(segment + 1 < source.size() && source[segment + 1].timestamp_s < target_s) {
            segment++;
        }
        if(segment + 1 >= source.size()) {
            resampled.push_back(source.back());
        }
        else {
            resampled.push_back(interpolate_pose(source[segment], source[segment + 1], target_s));
        }
    }
    timestamp_resampled = true;
    return resampled;
}

struct Quaternion {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

static double dot_quaternion(const Quaternion& a, const Quaternion& b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

static Quaternion negate_quaternion(const Quaternion& q) {
    Quaternion out;
    out.w = -q.w;
    out.x = -q.x;
    out.y = -q.y;
    out.z = -q.z;
    return out;
}

static Quaternion normalize_quaternion(const Quaternion& q) {
    const double norm = std::sqrt(dot_quaternion(q, q));
    if(norm < 1e-12) {
        return Quaternion{};
    }
    Quaternion out;
    out.w = q.w / norm;
    out.x = q.x / norm;
    out.y = q.y / norm;
    out.z = q.z / norm;
    return out;
}

static double quaternion_angle(const Quaternion& a, const Quaternion& b) {
    const Quaternion an = normalize_quaternion(a);
    const Quaternion bn = normalize_quaternion(b);
    double d = std::fabs(dot_quaternion(an, bn));
    d = std::min(std::max(d, -1.0), 1.0);
    return 2.0 * std::acos(d);
}

static Quaternion slerp_quaternion(const Quaternion& a_in, const Quaternion& b_in, double t) {
    Quaternion a = normalize_quaternion(a_in);
    Quaternion b = normalize_quaternion(b_in);
    double d = dot_quaternion(a, b);
    if(d < 0.0) {
        b = negate_quaternion(b);
        d = -d;
    }
    d = std::min(std::max(d, -1.0), 1.0);
    if(d > 0.9995) {
        Quaternion out;
        out.w = a.w + (b.w - a.w) * t;
        out.x = a.x + (b.x - a.x) * t;
        out.y = a.y + (b.y - a.y) * t;
        out.z = a.z + (b.z - a.z) * t;
        return normalize_quaternion(out);
    }
    const double theta = std::acos(d);
    const double sin_theta = std::sin(theta);
    const double wa = std::sin((1.0 - t) * theta) / sin_theta;
    const double wb = std::sin(t * theta) / sin_theta;
    Quaternion out;
    out.w = wa * a.w + wb * b.w;
    out.x = wa * a.x + wb * b.x;
    out.y = wa * a.y + wb * b.y;
    out.z = wa * a.z + wb * b.z;
    return normalize_quaternion(out);
}

static void cross3(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static double norm3(const T v[3]) {
    return std::sqrt(static_cast<double>(v[0]) * static_cast<double>(v[0]) + static_cast<double>(v[1]) * static_cast<double>(v[1]) + static_cast<double>(v[2]) * static_cast<double>(v[2]));
}

static bool normalize3(T v[3]) {
    const double norm = norm3(v);
    if(norm < 1e-9) {
        return false;
    }
    v[0] = static_cast<T>(static_cast<double>(v[0]) / norm);
    v[1] = static_cast<T>(static_cast<double>(v[1]) / norm);
    v[2] = static_cast<T>(static_cast<double>(v[2]) / norm);
    return true;
}

static void pose_basis(const TracePose& pose, T forward[3], T left[3], T up[3]) {
    pose_forward(pose, forward);
    up[0] = pose.up[0];
    up[1] = pose.up[1];
    up[2] = pose.up[2];
    if(!normalize3(up)) {
        up[0] = static_cast<T>(0);
        up[1] = static_cast<T>(0);
        up[2] = static_cast<T>(1);
    }
    cross3(up, forward, left);
    if(!normalize3(left)) {
        T reference_up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
        cross3(reference_up, forward, left);
        if(!normalize3(left)) {
            reference_up[0] = static_cast<T>(0);
            reference_up[1] = static_cast<T>(1);
            reference_up[2] = static_cast<T>(0);
            cross3(reference_up, forward, left);
            normalize3(left);
        }
    }
    cross3(forward, left, up);
    normalize3(up);
}

static Quaternion quaternion_from_basis(const T forward[3], const T left[3], const T up[3]) {
    const double m00 = forward[0];
    const double m01 = left[0];
    const double m02 = up[0];
    const double m10 = forward[1];
    const double m11 = left[1];
    const double m12 = up[1];
    const double m20 = forward[2];
    const double m21 = left[2];
    const double m22 = up[2];
    Quaternion q;
    const double trace = m00 + m11 + m22;
    if(trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    }
    else if(m00 > m11 && m00 > m22) {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q.w = (m21 - m12) / s;
        q.x = 0.25 * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    }
    else if(m11 > m22) {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25 * s;
        q.z = (m12 + m21) / s;
    }
    else {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25 * s;
    }
    return normalize_quaternion(q);
}

static Quaternion pose_quaternion(const TracePose& pose) {
    T forward[3];
    T left[3];
    T up[3];
    pose_basis(pose, forward, left, up);
    return quaternion_from_basis(forward, left, up);
}

static void quaternion_to_forward_up(const Quaternion& q_in, T forward[3], T up[3]) {
    const Quaternion q = normalize_quaternion(q_in);
    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double wx = q.w * q.x;
    const double wy = q.w * q.y;
    const double wz = q.w * q.z;

    forward[0] = static_cast<T>(1.0 - 2.0 * (yy + zz));
    forward[1] = static_cast<T>(2.0 * (xy + wz));
    forward[2] = static_cast<T>(2.0 * (xz - wy));

    up[0] = static_cast<T>(2.0 * (xz + wy));
    up[1] = static_cast<T>(2.0 * (yz - wx));
    up[2] = static_cast<T>(1.0 - 2.0 * (xx + yy));
    normalize(forward);
    normalize(up);
}

static std::vector<double> smoothing_times(const std::vector<TracePose>& poses, int fps) {
    double duration_s = 0.0;
    double implied_fps = 0.0;
    const bool use_trace_timestamps = trace_timestamp_stats(poses, duration_s, implied_fps);
    std::vector<double> times;
    times.reserve(poses.size());
    for(size_t i = 0; i < poses.size(); i++) {
        times.push_back(use_trace_timestamps ? poses[i].timestamp_s : static_cast<double>(i) / static_cast<double>(fps));
    }
    return times;
}

static double gaussian_weight(double dt, double sigma_s) {
    const double x = dt / sigma_s;
    return std::exp(-0.5 * x * x);
}

static bool trace_has_yaw_pitch(const std::vector<TracePose>& poses) {
    if(poses.empty()) {
        return false;
    }
    for(const TracePose& pose : poses) {
        if(!pose.has_yaw_pitch || !std::isfinite(pose.yaw) || !std::isfinite(pose.pitch)) {
            return false;
        }
    }
    return true;
}

static std::vector<double> gaussian_smooth_scalar(const std::vector<double>& values, const std::vector<double>& times, double sigma_s) {
    if(sigma_s <= 0.0 || values.size() < 2) {
        return values;
    }
    std::vector<double> output(values.size());
    const double radius_s = 3.0 * sigma_s;
    for(size_t i = 0; i < values.size(); i++) {
        const auto begin_it = std::lower_bound(times.begin(), times.end(), times[i] - radius_s);
        const auto end_it = std::upper_bound(times.begin(), times.end(), times[i] + radius_s);
        const size_t begin = static_cast<size_t>(begin_it - times.begin());
        const size_t end = static_cast<size_t>(end_it - times.begin());
        double sum_w = 0.0;
        double sum = 0.0;
        for(size_t j = begin; j < end; j++) {
            const double w = gaussian_weight(times[j] - times[i], sigma_s);
            sum_w += w;
            sum += w * values[j];
        }
        output[i] = sum_w > 0.0 ? sum / sum_w : values[i];
    }
    return output;
}

static void limit_yaw_pitch_speed(const std::vector<double>& input_yaw, const std::vector<double>& input_pitch, const std::vector<double>& times, double max_speed_rad_s, double ramp_multiplier, std::vector<double>& yaw, std::vector<double>& pitch) {
    yaw = input_yaw;
    pitch = input_pitch;
    if(max_speed_rad_s <= 0.0 || yaw.size() < 2) {
        return;
    }
    const double ramp_scale = std::max(ramp_multiplier, 1.0);
    bool ramping = false;
    for(size_t i = 1; i < yaw.size(); i++) {
        double dt = times[i] - times[i - 1];
        if(dt <= 1e-6 || !std::isfinite(dt)) {
            dt = 1.0 / 30.0;
        }
        const double trigger_angle = max_speed_rad_s * dt;
        const double ramp_angle = trigger_angle / ramp_scale;
        const double input_delta_yaw = input_yaw[i] - input_yaw[i - 1];
        const double input_delta_pitch = input_pitch[i] - input_pitch[i - 1];
        const double input_angle = std::sqrt(input_delta_yaw * input_delta_yaw + input_delta_pitch * input_delta_pitch);
        const double target_delta_yaw = input_yaw[i] - yaw[i - 1];
        const double target_delta_pitch = input_pitch[i] - pitch[i - 1];
        const double target_angle = std::sqrt(target_delta_yaw * target_delta_yaw + target_delta_pitch * target_delta_pitch);
        if(ramping || input_angle > trigger_angle) {
            if(ramp_angle > 0.0 && target_angle > ramp_angle) {
                const double alpha = ramp_angle / target_angle;
                yaw[i] = yaw[i - 1] + target_delta_yaw * alpha;
                pitch[i] = pitch[i - 1] + target_delta_pitch * alpha;
                ramping = true;
            }
            else {
                yaw[i] = input_yaw[i];
                pitch[i] = input_pitch[i];
                ramping = false;
            }
        }
    }
}

static std::vector<TracePose> filter_yaw_pitch_trace(const std::vector<TracePose>& input, const Options& options, bool& yaw_pitch_filter_applied) {
    const bool filter_requested = options.max_orientation_speed_rad_s > 0.0;
    yaw_pitch_filter_applied = false;
    if(!filter_requested || input.size() < 2 || !trace_has_yaw_pitch(input)) {
        return input;
    }

    const std::vector<double> times = smoothing_times(input, options.fps);
    std::vector<double> input_yaw(input.size());
    std::vector<double> input_pitch(input.size());
    input_yaw[0] = input[0].yaw;
    input_pitch[0] = input[0].pitch;
    for(size_t i = 1; i < input.size(); i++) {
        input_yaw[i] = input_yaw[i - 1] + wrap_angle_delta(input[i].yaw - input[i - 1].yaw);
        input_pitch[i] = input[i].pitch;
    }

    std::vector<double> yaw;
    std::vector<double> pitch;
    limit_yaw_pitch_speed(
        input_yaw,
        input_pitch,
        times,
        options.max_orientation_speed_rad_s,
        options.orientation_jump_ramp_multiplier,
        yaw,
        pitch
    );

    std::vector<TracePose> output = input;
    for(size_t i = 0; i < output.size(); i++) {
        set_pose_yaw_pitch(output[i], yaw[i], pitch[i]);
    }
    yaw_pitch_filter_applied = true;
    return output;
}

static std::vector<Quaternion> pose_quaternions_continuous(const std::vector<TracePose>& poses) {
    std::vector<Quaternion> quaternions;
    quaternions.reserve(poses.size());
    for(const TracePose& pose : poses) {
        Quaternion q = pose_quaternion(pose);
        if(!quaternions.empty() && dot_quaternion(q, quaternions.back()) < 0.0) {
            q = negate_quaternion(q);
        }
        quaternions.push_back(q);
    }
    return quaternions;
}

static void smooth_position_at(const std::vector<TracePose>& input, const std::vector<double>& times, size_t i, double sigma_s, T eye[3]) {
    const double radius_s = 3.0 * sigma_s;
    const auto begin_it = std::lower_bound(times.begin(), times.end(), times[i] - radius_s);
    const auto end_it = std::upper_bound(times.begin(), times.end(), times[i] + radius_s);
    const size_t begin = static_cast<size_t>(begin_it - times.begin());
    const size_t end = static_cast<size_t>(end_it - times.begin());
    double sum_w = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    for(size_t j = begin; j < end; j++) {
        const double w = gaussian_weight(times[j] - times[i], sigma_s);
        sum_w += w;
        sum[0] += w * static_cast<double>(input[j].eye[0]);
        sum[1] += w * static_cast<double>(input[j].eye[1]);
        sum[2] += w * static_cast<double>(input[j].eye[2]);
    }
    if(sum_w <= 0.0) {
        eye[0] = input[i].eye[0];
        eye[1] = input[i].eye[1];
        eye[2] = input[i].eye[2];
        return;
    }
    eye[0] = static_cast<T>(sum[0] / sum_w);
    eye[1] = static_cast<T>(sum[1] / sum_w);
    eye[2] = static_cast<T>(sum[2] / sum_w);
}

static Quaternion smooth_orientation_at(const std::vector<Quaternion>& quaternions, const std::vector<double>& times, size_t i, double sigma_s) {
    const double radius_s = 3.0 * sigma_s;
    const auto begin_it = std::lower_bound(times.begin(), times.end(), times[i] - radius_s);
    const auto end_it = std::upper_bound(times.begin(), times.end(), times[i] + radius_s);
    const size_t begin = static_cast<size_t>(begin_it - times.begin());
    const size_t end = static_cast<size_t>(end_it - times.begin());
    Quaternion sum;
    sum.w = 0.0;
    sum.x = 0.0;
    sum.y = 0.0;
    sum.z = 0.0;
    for(size_t j = begin; j < end; j++) {
        double w = gaussian_weight(times[j] - times[i], sigma_s);
        Quaternion q = quaternions[j];
        if(dot_quaternion(q, quaternions[i]) < 0.0) {
            q = negate_quaternion(q);
        }
        sum.w += w * q.w;
        sum.x += w * q.x;
        sum.y += w * q.y;
        sum.z += w * q.z;
    }
    return normalize_quaternion(sum);
}

static std::vector<Quaternion> limit_orientation_speed(const std::vector<Quaternion>& input, const std::vector<double>& times, double max_speed_rad_s, double ramp_multiplier) {
    if(max_speed_rad_s <= 0.0 || input.size() < 2) {
        return input;
    }
    const double ramp_scale = std::max(ramp_multiplier, 1.0);
    std::vector<Quaternion> output = input;
    bool ramping = false;
    for(size_t i = 1; i < output.size(); i++) {
        double dt = times[i] - times[i - 1];
        if(dt <= 1e-6 || !std::isfinite(dt)) {
            dt = 1.0 / 30.0;
        }
        const double trigger_angle = max_speed_rad_s * dt;
        const double ramp_angle = trigger_angle / ramp_scale;
        const double input_angle = quaternion_angle(input[i - 1], input[i]);
        const double target_angle = quaternion_angle(output[i - 1], input[i]);
        if(ramping || input_angle > trigger_angle) {
            if(ramp_angle > 0.0 && target_angle > ramp_angle) {
                output[i] = slerp_quaternion(output[i - 1], input[i], ramp_angle / target_angle);
                ramping = true;
            }
            else {
                output[i] = input[i];
                ramping = false;
            }
        }
    }
    return output;
}

static void set_pose_orientation(TracePose& pose, const Quaternion& q) {
    T forward[3];
    T up[3];
    quaternion_to_forward_up(q, forward, up);
    pose.up[0] = up[0];
    pose.up[1] = up[1];
    pose.up[2] = up[2];
    pose.look_at[0] = pose.eye[0] + forward[0];
    pose.look_at[1] = pose.eye[1] + forward[1];
    pose.look_at[2] = pose.eye[2] + forward[2];
}

static std::vector<TracePose> smooth_trace(const std::vector<TracePose>& input, const Options& options, bool& smoothing_applied, bool source_yaw_pitch_filter_applied) {
    const bool smoothing_requested = options.smooth_position_sigma_s > 0.0 ||
        options.smooth_orientation_sigma_s > 0.0 ||
        (!source_yaw_pitch_filter_applied && options.max_orientation_speed_rad_s > 0.0);
    smoothing_applied = smoothing_requested && input.size() >= 2;
    if(!smoothing_applied) {
        return input;
    }

    const std::vector<double> times = smoothing_times(input, options.fps);
    std::vector<TracePose> output = input;

    if(options.smooth_position_sigma_s > 0.0) {
        for(size_t i = 0; i < input.size(); i++) {
            T forward[3];
            pose_forward(input[i], forward);
            smooth_position_at(input, times, i, options.smooth_position_sigma_s, output[i].eye);
            output[i].look_at[0] = output[i].eye[0] + forward[0];
            output[i].look_at[1] = output[i].eye[1] + forward[1];
            output[i].look_at[2] = output[i].eye[2] + forward[2];
        }
    }

    if(options.smooth_orientation_sigma_s > 0.0 || (!source_yaw_pitch_filter_applied && options.max_orientation_speed_rad_s > 0.0)) {
        if(trace_has_yaw_pitch(input)) {
            std::vector<double> input_yaw(input.size());
            std::vector<double> input_pitch(input.size());
            input_yaw[0] = input[0].yaw;
            input_pitch[0] = input[0].pitch;
            for(size_t i = 1; i < input.size(); i++) {
                input_yaw[i] = input_yaw[i - 1] + wrap_angle_delta(input[i].yaw - input[i - 1].yaw);
                input_pitch[i] = input[i].pitch;
            }
            std::vector<double> yaw = input_yaw;
            std::vector<double> pitch = input_pitch;
            if(!source_yaw_pitch_filter_applied) {
                limit_yaw_pitch_speed(
                    input_yaw,
                    input_pitch,
                    times,
                    options.max_orientation_speed_rad_s,
                    options.orientation_jump_ramp_multiplier,
                    yaw,
                    pitch
                );
            }
            yaw = gaussian_smooth_scalar(yaw, times, options.smooth_orientation_sigma_s);
            pitch = gaussian_smooth_scalar(pitch, times, options.smooth_orientation_sigma_s);
            for(size_t i = 0; i < input.size(); i++) {
                set_pose_yaw_pitch(output[i], yaw[i], pitch[i]);
            }
        }
        else {
            std::vector<Quaternion> quaternions = pose_quaternions_continuous(input);
            quaternions = limit_orientation_speed(
                quaternions,
                times,
                options.max_orientation_speed_rad_s,
                options.orientation_jump_ramp_multiplier
            );
            for(size_t i = 0; i < input.size(); i++) {
                const Quaternion q = options.smooth_orientation_sigma_s > 0.0
                    ? smooth_orientation_at(quaternions, times, i, options.smooth_orientation_sigma_s)
                    : quaternions[i];
                set_pose_orientation(output[i], q);
            }
        }
    }

    return output;
}

static std::string resolution_suffix(TI width, TI height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

static std::string aa_suffix(bool enabled, TI grid_size) {
    return enabled ? std::string("_aa") + std::to_string(grid_size) : std::string();
}

static std::string aa_name(bool enabled, TI grid_size) {
    return enabled ? std::string("aa") + std::to_string(grid_size) : std::string("none");
}

static TI video_oversampling_factor(TI width, TI height) {
    const TI width_factor = width < MIN_VIDEO_DIMENSION ? (MIN_VIDEO_DIMENSION + width - 1) / width : static_cast<TI>(1);
    const TI height_factor = height < MIN_VIDEO_DIMENSION ? (MIN_VIDEO_DIMENSION + height - 1) / height : static_cast<TI>(1);
    return std::max(width_factor, height_factor);
}

static std::string ffmpeg_command(const Options& options, TI width, TI height, TI video_width, TI video_height, const std::string& output_path) {
    std::ostringstream cmd;
    cmd << shell_quote(options.ffmpeg)
        << " -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgba"
        << " -s " << width << "x" << height
        << " -r " << options.fps
        << " -i -";
    if(video_width != width || video_height != height) {
        cmd << " -vf scale=" << video_width << ":" << video_height << ":flags=neighbor";
    }
    cmd << " -an -c:v libx264 -pix_fmt yuv420p " << shell_quote(output_path);
    return cmd.str();
}

static bool write_frame(FILE* pipe, const std::vector<uint32_t>& frame, int frame_i) {
    const size_t bytes = frame.size() * sizeof(uint32_t);
    const size_t written = std::fwrite(frame.data(), 1, bytes, pipe);
    if(written != bytes) {
        std::cerr << "Failed writing frame " << frame_i << " to ffmpeg" << std::endl;
        return false;
    }
    return true;
}

static bool use_png_frames(TI width, TI height) {
    return width <= 120 && height <= 68;
}

static const char* frame_extension(bool png) {
    return png ? "png" : "jpg";
}

static const char* frame_format_name(bool png) {
    return png ? "png_rgb" : "jpeg_rgb";
}

static std::string frame_filename(const std::string& frames_dir, size_t frame_i, bool png) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "frame_%06zu.%s", frame_i, frame_extension(png));
    return join_path(frames_dir, filename);
}

static bool write_rgb_image_frame(const std::string& path, const std::vector<uint32_t>& frame, std::vector<uint8_t>& rgb_frame, TI width, TI height, bool png) {
    const size_t pixels = frame.size();
    if(rgb_frame.size() != pixels * 3) {
        rgb_frame.resize(pixels * 3);
    }
    for(size_t i = 0; i < pixels; i++) {
        const uint32_t rgba = frame[i];
        rgb_frame[i * 3 + 0] = static_cast<uint8_t>((rgba >> 0) & 0xFF);
        rgb_frame[i * 3 + 1] = static_cast<uint8_t>((rgba >> 8) & 0xFF);
        rgb_frame[i * 3 + 2] = static_cast<uint8_t>((rgba >> 16) & 0xFF);
    }
    const int write_ok = png
        ? stbi_write_png(path.c_str(), width, height, 3, rgb_frame.data(), width * 3)
        : stbi_write_jpg(path.c_str(), width, height, 3, rgb_frame.data(), FRAME_JPEG_QUALITY);
    if(write_ok == 0) {
        std::cerr << "Failed writing RGB " << (png ? "PNG" : "JPEG") << " frame: " << path << std::endl;
        return false;
    }
    return true;
}

static void depth_range(const float* depth, size_t count, float max_depth, float& min_depth, float& max_valid_depth) {
    const float sentinel = max_depth * 0.999f;
    for(size_t i = 0; i < count; i++) {
        const float value = depth[i];
        if(std::isfinite(value) && value > 0.f && value < sentinel) {
            min_depth = std::min(min_depth, value);
            max_valid_depth = std::max(max_valid_depth, value);
        }
    }
}

static void depth_to_rgba(const float* depth, std::vector<uint32_t>& frame, float max_depth, float min_valid_depth, float max_valid_depth) {
    const float sentinel = max_depth * 0.999f;
    const bool has_valid_depth = min_valid_depth <= max_valid_depth;
    const float range = has_valid_depth ? max_valid_depth - min_valid_depth : 0.f;
    for(size_t i = 0; i < frame.size(); i++) {
        uint8_t value = 0;
        const float d = depth[i];
        if(has_valid_depth && std::isfinite(d) && d > 0.f && d < sentinel) {
            const float normalized = std::fmin(std::fmax((d - min_valid_depth) / (range + 1e-6f), 0.f), 1.f);
            value = static_cast<uint8_t>((1.f - normalized) * 255.f);
        }
        frame[i] = (0xFFu << 24) | (uint32_t(value) << 16) | (uint32_t(value) << 8) | uint32_t(value);
    }
}

template <typename SPEC>
static bool render_trace_for_setting(rlt::devices::DEVICE_FACTORY<>& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, const char* output_name, const char* profile_name, RenderRecord& record) {
    constexpr TI WIDTH = SPEC::CAM_WIDTH;
    constexpr TI HEIGHT = SPEC::CAM_HEIGHT;
    constexpr bool ENABLE_AA = SPEC::ENABLE_ANTI_ALIASING;
    constexpr TI AA_GRID_SIZE = ENABLE_AA ? SPEC::ANTI_ALIASING_GRID_SIZE : 1;
    record.output = output_name;
    record.profile = profile_name[0] == '\0' ? "none" : profile_name;
    record.anti_aliasing = aa_name(ENABLE_AA, AA_GRID_SIZE);
    record.width = WIDTH;
    record.height = HEIGHT;
    record.video_oversampling_factor = video_oversampling_factor(WIDTH, HEIGHT);
    record.video_width = WIDTH * record.video_oversampling_factor;
    record.video_height = HEIGHT * record.video_oversampling_factor;
    record.aa_grid_size = AA_GRID_SIZE;
    record.samples_per_pixel = AA_GRID_SIZE * AA_GRID_SIZE;
    const std::string base_name = profile_name[0] == '\0' ? std::string(output_name) : std::string(output_name) + "_" + profile_name;
    record.name = base_name + "_" + resolution_suffix(WIDTH, HEIGHT) + aa_suffix(ENABLE_AA, AA_GRID_SIZE);
    record.path = join_path(options.output_dir, std::string("trace_") + record.name + ".mp4");
    const bool png_frames = use_png_frames(WIDTH, HEIGHT);
    if(options.write_frames) {
        record.frames_dir = join_path(join_path(options.output_dir, "frames"), record.name);
        record.frame_pattern = join_path(record.frames_dir, std::string("frame_%06d.") + frame_extension(png_frames));
        record.frame_format = frame_format_name(png_frames);
        record.frame_jpeg_quality = png_frames ? 0 : FRAME_JPEG_QUALITY;
    }
    record.ffmpeg_command = ffmpeg_command(options, WIDTH, HEIGHT, record.video_width, record.video_height, record.path);
    record.frame_count = poses.size();

    if(options.write_frames && !mkdir_p(record.frames_dir)) {
        return false;
    }

    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = scene_path.c_str();
    rlt::malloc(device, env);
    rlt::init(device, env);

    std::vector<uint32_t> frame(static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT));
    FILE* pipe = popen(record.ffmpeg_command.c_str(), "w");
    if(pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg for " << record.path << std::endl;
        rlt::free(device, env);
        return false;
    }

    bool ok = true;
    std::vector<uint8_t> rgb_frame;
    const T fov = static_cast<T>(degrees_to_radians(options.fov_deg));
    for(size_t frame_i = 0; frame_i < poses.size(); frame_i++) {
        const TracePose& pose = poses[frame_i];
        rlt::set(device, env.renderer->cameras, rlt::make_camera_data(pose.eye, pose.look_at, pose.up, fov, static_cast<T>(WIDTH) / static_cast<T>(HEIGHT)), static_cast<TI>(0));
        rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
        if constexpr (SPEC::HAS_DEPTH) {
            rlt::render_depth_only(device, *env.renderer);
            rlt::read_depth_buffer(device, *env.renderer, env.renderer->depth_buffer);
            const float miss_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            float min_depth = std::numeric_limits<float>::max();
            float max_depth_value = std::numeric_limits<float>::lowest();
            depth_range(rlt::data(env.renderer->depth_buffer), frame.size(), miss_depth, min_depth, max_depth_value);
            depth_to_rgba(rlt::data(env.renderer->depth_buffer), frame, miss_depth, min_depth, max_depth_value);
        }
        else {
            rlt::render_rgb_only(device, *env.renderer);
            rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);
            const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
            std::memcpy(frame.data(), fb_data, frame.size() * sizeof(uint32_t));
        }
        if(options.write_frames && !write_rgb_image_frame(frame_filename(record.frames_dir, frame_i, png_frames), frame, rgb_frame, WIDTH, HEIGHT, png_frames)) {
            ok = false;
            break;
        }
        if(!write_frame(pipe, frame, static_cast<int>(frame_i))) {
            ok = false;
            break;
        }
    }

    record.ffmpeg_status = pclose(pipe);
    record.ok = ok && record.ffmpeg_status == 0;
    if(!record.ok) {
        std::cerr << "ffmpeg failed for " << record.path << " with status " << record.ffmpeg_status << std::endl;
    }
    else {
        std::cout << "Wrote " << record.path << std::endl;
        if(options.write_frames) {
            std::cout << "Wrote RGB frames to " << record.frames_dir << std::endl;
        }
    }

    rlt::free(device, env);
    return record.ok;
}

static bool write_manifest(const Options& options, const std::string& scene_path, const std::vector<TracePose>& source_poses, const std::vector<TracePose>& render_poses, double trace_duration_s, double source_implied_fps, bool timestamp_resampled, bool smoothing_applied, bool yaw_pitch_filter_applied, const std::vector<RenderRecord>& records) {
    json manifest;
    manifest["trace_path"] = options.trace_path;
    manifest["scene_path"] = scene_path;
    manifest["output_dir"] = options.output_dir;
    manifest["fps"] = options.fps;
    manifest["fov_deg"] = options.fov_deg;
    manifest["fov_rad"] = degrees_to_radians(options.fov_deg);
    manifest["resolutions"] = json::array();
    for(const RenderResolutionOption& resolution : RENDER_RESOLUTIONS) {
        if(should_render_resolution(options, resolution.width, resolution.height)) {
            if(resolution.width == resolution.height) {
                manifest["resolutions"].push_back(resolution.width);
            }
            else {
                manifest["resolutions"].push_back(resolution.name);
            }
        }
    }
    manifest["write_frames"] = options.write_frames;
    manifest["aa"] = aa_selection_name(options.aa);
    manifest["anti_aliasing"] = json::array();
    if(should_render_aa(options, false, 1)) {
        manifest["anti_aliasing"].push_back({
            {"name", "none"},
            {"enabled", false},
            {"grid_size", 1},
            {"samples_per_pixel", 1}
        });
    }
    if(should_render_aa(options, true, 2)) {
        manifest["anti_aliasing"].push_back({
            {"name", "aa2"},
            {"enabled", true},
            {"grid_size", 2},
            {"samples_per_pixel", 4}
        });
    }
    manifest["frames"] = render_poses.size();
    manifest["source_frames"] = source_poses.size();
    manifest["rendered_frames"] = render_poses.size();
    manifest["trace_duration_s"] = trace_duration_s;
    manifest["source_implied_fps"] = source_implied_fps;
    manifest["timestamp_resampled"] = timestamp_resampled;
    manifest["depth_visualization"] = {
        {"mapping", "per_frame_min_max"},
        {"near_value", 255},
        {"far_value", 0}
    };
    manifest["smoothing"] = {
        {"applied", smoothing_applied},
        {"algorithm", yaw_pitch_filter_applied ? "source_yaw_pitch_speed_limit_then_resample_then_yaw_pitch_gaussian" : "yaw_pitch_or_quaternion_speed_limit_then_gaussian"},
        {"source_yaw_pitch_filter_applied", yaw_pitch_filter_applied},
        {"position_sigma_s", options.smooth_position_sigma_s},
        {"orientation_sigma_s", options.smooth_orientation_sigma_s},
        {"max_orientation_speed_rad_s", options.max_orientation_speed_rad_s},
        {"max_orientation_speed_deg_s", options.max_orientation_speed_rad_s * 180.0 / 3.14159265358979323846},
        {"orientation_jump_ramp_multiplier", options.orientation_jump_ramp_multiplier}
    };
    manifest["renders"] = json::array();
    for(const RenderRecord& record : records) {
        json item;
        item["name"] = record.name;
        item["output"] = record.output;
        item["profile"] = record.profile;
        item["fidelity"] = record.profile;
        item["anti_aliasing"] = record.anti_aliasing;
        item["aa_grid_size"] = record.aa_grid_size;
        item["samples_per_pixel"] = record.samples_per_pixel;
        item["width"] = record.width;
        item["height"] = record.height;
        item["video_width"] = record.video_width;
        item["video_height"] = record.video_height;
        item["video_oversampling_factor"] = record.video_oversampling_factor;
        item["path"] = record.path;
        item["write_frames"] = options.write_frames;
        if(options.write_frames) {
            item["frames_dir"] = record.frames_dir;
            item["frame_pattern"] = record.frame_pattern;
            item["frame_format"] = record.frame_format;
            if(record.frame_jpeg_quality > 0) {
                item["frame_jpeg_quality"] = record.frame_jpeg_quality;
            }
        }
        item["frame_count"] = record.frame_count;
        item["ffmpeg_command"] = record.ffmpeg_command;
        item["ffmpeg_status"] = record.ffmpeg_status;
        item["ok"] = record.ok;
        manifest["renders"].push_back(item);
    }

    const std::string manifest_path = join_path(options.output_dir, "trace_render_manifest.json");
    std::ofstream f(manifest_path);
    if(!f) {
        std::cerr << "Failed to write manifest: " << manifest_path << std::endl;
        return false;
    }
    f << manifest.dump(2) << "\n";
    std::cout << "Wrote " << manifest_path << std::endl;
    return true;
}

template <TI WIDTH, TI HEIGHT, bool ENABLE_AA, TI AA_GRID_SIZE>
static bool render_selected_settings_for_resolution(rlt::devices::DEVICE_FACTORY<>& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, std::vector<RenderRecord>& records) {
    if(!should_render_resolution(options, WIDTH, HEIGHT)) {
        return true;
    }
    if(!should_render_aa(options, ENABLE_AA, AA_GRID_SIZE)) {
        return true;
    }
    bool ok = true;
    if(should_render_setting(options, "rgb", "very_high")) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, WIDTH, HEIGHT, NUM_PROBES, rlt::rendering::raytracing::VeryHigh, false, 1, ENABLE_AA, AA_GRID_SIZE, rlt::rendering::raytracing::OutputMode::RGB>;
        records.emplace_back();
        ok = render_trace_for_setting<SPEC>(device, options, scene_path, poses, "rgb", "very_high", records.back()) && ok;
    }
    if(should_render_depth(options)) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, WIDTH, HEIGHT, NUM_PROBES, rlt::rendering::raytracing::Medium, false, 1, ENABLE_AA, AA_GRID_SIZE, rlt::rendering::raytracing::OutputMode::DEPTH>;
        records.emplace_back();
        ok = render_trace_for_setting<SPEC>(device, options, scene_path, poses, "depth", "", records.back()) && ok;
    }
    return ok;
}

int main(int argc, char** argv) {
    Options options;
    if(!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    if(options.help) {
        print_usage(argv[0]);
        return 0;
    }
    options.output_dir = join_path(options.output_dir, trace_output_name(options.trace_path));
    if(!mkdir_p(options.output_dir)) {
        return 1;
    }

    std::string scene_path;
    std::vector<TracePose> source_poses;
    if(!load_trace(options, scene_path, source_poses)) {
        return 1;
    }
    bool yaw_pitch_filter_applied = false;
    std::vector<TracePose> filtered_source_poses = filter_yaw_pitch_trace(source_poses, options, yaw_pitch_filter_applied);
    if(yaw_pitch_filter_applied) {
        std::cout << "Applied source yaw/pitch jump filter before resampling with max orientation speed "
                  << (options.max_orientation_speed_rad_s * 180.0 / 3.14159265358979323846)
                  << " deg/s with " << options.orientation_jump_ramp_multiplier
                  << "x jump ramps" << std::endl;
    }
    double trace_duration_s = 0.0;
    double source_implied_fps = 0.0;
    bool timestamp_resampled = false;
    std::vector<TracePose> poses = resample_trace(filtered_source_poses, options.fps, trace_duration_s, source_implied_fps, timestamp_resampled);
    if(timestamp_resampled) {
        std::cout << "Resampled " << source_poses.size() << " timestamped poses over " << trace_duration_s << "s to " << poses.size() << " frames at " << options.fps << " fps" << std::endl;
    }
    else {
        std::cout << "Rendering " << poses.size() << " poses without timestamp resampling" << std::endl;
    }
    bool post_resample_smoothing_applied = false;
    poses = smooth_trace(poses, options, post_resample_smoothing_applied, yaw_pitch_filter_applied);
    const bool smoothing_applied = yaw_pitch_filter_applied || post_resample_smoothing_applied;
    if(post_resample_smoothing_applied) {
        std::cout << "Applied post-resample trace smoothing with position sigma " << options.smooth_position_sigma_s << "s";
        if(!yaw_pitch_filter_applied) {
            std::cout << ", orientation sigma " << options.smooth_orientation_sigma_s
                      << "s, and max orientation speed " << (options.max_orientation_speed_rad_s * 180.0 / 3.14159265358979323846)
                      << " deg/s with " << options.orientation_jump_ramp_multiplier
                      << "x jump ramps";
        }
        std::cout << std::endl;
    }

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    DEVICE device;
    rlt::init(device);

    std::vector<RenderRecord> records;
    bool ok = true;

    ok = render_selected_settings_for_resolution<60, 34, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<60, 34, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<120, 68, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<120, 68, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<240, 135, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<240, 135, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<480, 270, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<480, 270, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<960, 540, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<960, 540, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<1920, 1080, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<1920, 1080, true, 2>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<3840, 2160, false, 1>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<3840, 2160, true, 2>(device, options, scene_path, poses, records) && ok;

    if(records.empty()) {
        std::cerr << "No render settings selected by --settings=" << options.settings << std::endl;
        return 1;
    }
    ok = write_manifest(options, scene_path, source_poses, poses, trace_duration_s, source_implied_fps, timestamp_resampled, smoothing_applied, yaw_pitch_filter_applied, records) && ok;
    return ok ? 0 : 1;
}
