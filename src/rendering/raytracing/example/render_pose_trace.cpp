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
static constexpr TI RENDER_RESOLUTIONS[] = {64, 128, 256, 512, 1024, 2048};

struct TracePose {
    T eye[3];
    T look_at[3];
    T up[3];
    double timestamp_s = 0.0;
    bool has_timestamp = false;
};

struct Options {
    std::string trace_path;
    std::string scene_path;
    std::string output_dir = ".";
    std::string ffmpeg = "ffmpeg";
    std::string settings = "all";
    int fps = 30;
    int max_frames = 0;
    double smooth_position_sigma_s = 0.0;
    double smooth_orientation_sigma_s = 0.0;
    bool help = false;
};

struct RenderRecord {
    std::string name;
    std::string output;
    std::string fidelity;
    std::string path;
    std::string ffmpeg_command;
    TI width = 0;
    TI height = 0;
    int ffmpeg_status = 0;
    bool ok = false;
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --trace pose_trace.json [options]\n"
        << "Options:\n"
        << "  --trace <path>             Camera pose JSON or trace JSON\n"
        << "  --scene <path>             Scene path override\n"
        << "  --output-dir <dir>         Output directory (default: .)\n"
        << "  --fps <n>                  MP4 frame rate; timestamped traces are resampled to this rate (default: 30)\n"
        << "  --ffmpeg <path>            ffmpeg binary (default: ffmpeg)\n"
        << "  --settings <list>          all, rgb, depth, basic, high_fidelity, fast_flat, or comma list; depth has no fidelity profile\n"
        << "                              Always renders square resolutions 64, 128, 256, 512, 1024, and 2048\n"
        << "  --max-frames <n>           Limit trace frames when >0\n"
        << "  --smooth-sigma-s <s>       Gaussian smoothing sigma for position and orientation (default: 0)\n"
        << "  --smooth-position-sigma-s <s>\n"
        << "                              Gaussian smoothing sigma for position only (default: 0)\n"
        << "  --smooth-orientation-sigma-s <s>\n"
        << "                              Gaussian smoothing sigma for orientation only (default: 0)\n";
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

static bool parse_double(const std::string& value, double& out) {
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if(end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
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
        else if(option_value(i, argc, argv, arg, "--fps", value)) {
            if(!parse_int(value, options.fps) || options.fps <= 0) {
                std::cerr << "Invalid --fps: " << value << std::endl;
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

static bool should_render_setting(const Options& options, const char* output, const char* fidelity) {
    const std::vector<std::string> tokens = split_settings(options.settings);
    if(tokens.empty() || contains_token(tokens, "all")) {
        return true;
    }
    const bool has_output_filter = contains_token(tokens, "rgb") || contains_token(tokens, "depth");
    const bool has_fidelity_filter = contains_token(tokens, "basic") || contains_token(tokens, "high_fidelity") || contains_token(tokens, "fast_flat");
    const bool output_ok = !has_output_filter || contains_token(tokens, output);
    const bool fidelity_ok = !has_fidelity_filter || contains_token(tokens, fidelity);
    return output_ok && fidelity_ok;
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

static bool parse_pose(const json& pose_json, TracePose& pose) {
    const json* source = &pose_json;
    pose.timestamp_s = 0.0;
    pose.has_timestamp = false;
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
        const T yaw = (*source)["yaw"].get<T>();
        const T pitch = (*source)["pitch"].get<T>();
        pose.look_at[0] = pose.eye[0] + std::cos(yaw) * std::cos(pitch);
        pose.look_at[1] = pose.eye[1] + std::sin(yaw) * std::cos(pitch);
        pose.look_at[2] = pose.eye[2] + std::sin(pitch);
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

static TracePose interpolate_pose(const TracePose& a, const TracePose& b, double timestamp_s) {
    const double duration = b.timestamp_s - a.timestamp_s;
    const double t = duration > 0.0 ? std::min(std::max((timestamp_s - a.timestamp_s) / duration, 0.0), 1.0) : 0.0;

    TracePose out;
    lerp_vec3(a.eye, b.eye, t, out.eye);

    T forward_a[3];
    T forward_b[3];
    T forward[3];
    pose_forward(a, forward_a);
    pose_forward(b, forward_b);
    lerp_vec3(forward_a, forward_b, t, forward);
    normalize(forward);

    lerp_vec3(a.up, b.up, t, out.up);
    normalize(out.up);

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

static std::vector<TracePose> smooth_trace(const std::vector<TracePose>& input, const Options& options, bool& smoothing_applied) {
    const bool smoothing_requested = options.smooth_position_sigma_s > 0.0 || options.smooth_orientation_sigma_s > 0.0;
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

    if(options.smooth_orientation_sigma_s > 0.0) {
        const std::vector<Quaternion> quaternions = pose_quaternions_continuous(input);
        for(size_t i = 0; i < input.size(); i++) {
            const Quaternion q = smooth_orientation_at(quaternions, times, i, options.smooth_orientation_sigma_s);
            T forward[3];
            T up[3];
            quaternion_to_forward_up(q, forward, up);
            output[i].up[0] = up[0];
            output[i].up[1] = up[1];
            output[i].up[2] = up[2];
            output[i].look_at[0] = output[i].eye[0] + forward[0];
            output[i].look_at[1] = output[i].eye[1] + forward[1];
            output[i].look_at[2] = output[i].eye[2] + forward[2];
        }
    }

    return output;
}

static std::string resolution_suffix(TI width, TI height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

static std::string ffmpeg_command(const Options& options, TI width, TI height, const std::string& output_path) {
    std::ostringstream cmd;
    cmd << shell_quote(options.ffmpeg)
        << " -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgba"
        << " -s " << width << "x" << height
        << " -r " << options.fps
        << " -i - -an -c:v libx264 -pix_fmt yuv420p "
        << shell_quote(output_path);
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
static bool render_trace_for_setting(rlt::devices::DEVICE_FACTORY<>& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, const char* output_name, const char* fidelity_name, RenderRecord& record) {
    constexpr TI WIDTH = SPEC::CAM_WIDTH;
    constexpr TI HEIGHT = SPEC::CAM_HEIGHT;
    record.output = output_name;
    record.fidelity = fidelity_name[0] == '\0' ? "none" : fidelity_name;
    record.width = WIDTH;
    record.height = HEIGHT;
    const std::string base_name = fidelity_name[0] == '\0' ? std::string(output_name) : std::string(output_name) + "_" + fidelity_name;
    record.name = base_name + "_" + resolution_suffix(WIDTH, HEIGHT);
    record.path = join_path(options.output_dir, std::string("trace_") + record.name + ".mp4");
    record.ffmpeg_command = ffmpeg_command(options, WIDTH, HEIGHT, record.path);

    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = scene_path.c_str();
    rlt::malloc(device, env);
    rlt::init(device, env);

    std::vector<uint32_t> frame(static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT));
    float min_depth = std::numeric_limits<float>::max();
    float max_depth_value = std::numeric_limits<float>::lowest();

    if constexpr (SPEC::HAS_DEPTH) {
        const float miss_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
        for(const TracePose& pose : poses) {
            rlt::set(device, env.renderer->cameras, rlt::make_camera_data(pose.eye, pose.look_at, pose.up, SPEC::RAYTRACING_SPEC::COS_FOVY, static_cast<T>(WIDTH) / static_cast<T>(HEIGHT)), static_cast<TI>(0));
            rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
            rlt::render_depth_only(device, *env.renderer);
            rlt::read_depth_buffer(device, *env.renderer, env.renderer->depth_buffer);
            depth_range(rlt::data(env.renderer->depth_buffer), frame.size(), miss_depth, min_depth, max_depth_value);
        }
    }

    FILE* pipe = popen(record.ffmpeg_command.c_str(), "w");
    if(pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg for " << record.path << std::endl;
        rlt::free(device, env);
        return false;
    }

    bool ok = true;
    for(size_t frame_i = 0; frame_i < poses.size(); frame_i++) {
        const TracePose& pose = poses[frame_i];
        rlt::set(device, env.renderer->cameras, rlt::make_camera_data(pose.eye, pose.look_at, pose.up, SPEC::RAYTRACING_SPEC::COS_FOVY, static_cast<T>(WIDTH) / static_cast<T>(HEIGHT)), static_cast<TI>(0));
        rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
        if constexpr (SPEC::HAS_DEPTH) {
            rlt::render_depth_only(device, *env.renderer);
            rlt::read_depth_buffer(device, *env.renderer, env.renderer->depth_buffer);
            const float miss_depth = env.renderer->camera_radius > 0 ? env.renderer->camera_radius * 2.0f : 1e30f;
            depth_to_rgba(rlt::data(env.renderer->depth_buffer), frame, miss_depth, min_depth, max_depth_value);
        }
        else {
            rlt::render_rgb_only(device, *env.renderer);
            rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);
            const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
            std::memcpy(frame.data(), fb_data, frame.size() * sizeof(uint32_t));
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
    }

    rlt::free(device, env);
    return record.ok;
}

static bool write_manifest(const Options& options, const std::string& scene_path, const std::vector<TracePose>& source_poses, const std::vector<TracePose>& render_poses, double trace_duration_s, double source_implied_fps, bool timestamp_resampled, bool smoothing_applied, const std::vector<RenderRecord>& records) {
    json manifest;
    manifest["trace_path"] = options.trace_path;
    manifest["scene_path"] = scene_path;
    manifest["output_dir"] = options.output_dir;
    manifest["fps"] = options.fps;
    manifest["resolutions"] = json::array();
    for(TI resolution : RENDER_RESOLUTIONS) {
        manifest["resolutions"].push_back(resolution);
    }
    manifest["frames"] = render_poses.size();
    manifest["source_frames"] = source_poses.size();
    manifest["rendered_frames"] = render_poses.size();
    manifest["trace_duration_s"] = trace_duration_s;
    manifest["source_implied_fps"] = source_implied_fps;
    manifest["timestamp_resampled"] = timestamp_resampled;
    manifest["smoothing"] = {
        {"applied", smoothing_applied},
        {"algorithm", "offline_zero_phase_gaussian_quaternion_nlerp"},
        {"position_sigma_s", options.smooth_position_sigma_s},
        {"orientation_sigma_s", options.smooth_orientation_sigma_s}
    };
    manifest["renders"] = json::array();
    for(const RenderRecord& record : records) {
        json item;
        item["name"] = record.name;
        item["output"] = record.output;
        item["fidelity"] = record.fidelity;
        item["width"] = record.width;
        item["height"] = record.height;
        item["path"] = record.path;
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

template <TI RESOLUTION>
static bool render_selected_settings_for_resolution(rlt::devices::DEVICE_FACTORY<>& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, std::vector<RenderRecord>& records) {
    bool ok = true;
    if(should_render_setting(options, "rgb", "basic")) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, RESOLUTION, RESOLUTION, NUM_PROBES, rlt::rendering::raytracing::BasicShading, false, 1, false, 1, rlt::rendering::raytracing::OutputMode::RGB>;
        records.emplace_back();
        ok = render_trace_for_setting<SPEC>(device, options, scene_path, poses, "rgb", "basic", records.back()) && ok;
    }
    if(should_render_setting(options, "rgb", "high_fidelity")) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, RESOLUTION, RESOLUTION, NUM_PROBES, rlt::rendering::raytracing::HighFidelityShading, false, 1, false, 1, rlt::rendering::raytracing::OutputMode::RGB>;
        records.emplace_back();
        ok = render_trace_for_setting<SPEC>(device, options, scene_path, poses, "rgb", "high_fidelity", records.back()) && ok;
    }
    if(should_render_setting(options, "rgb", "fast_flat")) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, RESOLUTION, RESOLUTION, NUM_PROBES, rlt::rendering::raytracing::FastFlatShading, false, 1, false, 1, rlt::rendering::raytracing::OutputMode::RGB>;
        records.emplace_back();
        ok = render_trace_for_setting<SPEC>(device, options, scene_path, poses, "rgb", "fast_flat", records.back()) && ok;
    }
    if(should_render_depth(options)) {
        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_CAMERAS, RESOLUTION, RESOLUTION, NUM_PROBES, rlt::rendering::raytracing::BasicShading, false, 1, false, 1, rlt::rendering::raytracing::OutputMode::DEPTH>;
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
    if(!mkdir_p(options.output_dir)) {
        return 1;
    }

    std::string scene_path;
    std::vector<TracePose> source_poses;
    if(!load_trace(options, scene_path, source_poses)) {
        return 1;
    }
    double trace_duration_s = 0.0;
    double source_implied_fps = 0.0;
    bool timestamp_resampled = false;
    std::vector<TracePose> poses = resample_trace(source_poses, options.fps, trace_duration_s, source_implied_fps, timestamp_resampled);
    if(timestamp_resampled) {
        std::cout << "Resampled " << source_poses.size() << " timestamped poses over " << trace_duration_s << "s to " << poses.size() << " frames at " << options.fps << " fps" << std::endl;
    }
    else {
        std::cout << "Rendering " << poses.size() << " poses without timestamp resampling" << std::endl;
    }
    bool smoothing_applied = false;
    poses = smooth_trace(poses, options, smoothing_applied);
    if(smoothing_applied) {
        std::cout << "Applied Gaussian smoothing with position sigma " << options.smooth_position_sigma_s << "s and orientation sigma " << options.smooth_orientation_sigma_s << "s" << std::endl;
    }

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    DEVICE device;
    rlt::init(device);

    std::vector<RenderRecord> records;
    bool ok = true;

    ok = render_selected_settings_for_resolution<64>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<128>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<256>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<512>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<1024>(device, options, scene_path, poses, records) && ok;
    ok = render_selected_settings_for_resolution<2048>(device, options, scene_path, poses, records) && ok;

    if(records.empty()) {
        std::cerr << "No render settings selected by --settings=" << options.settings << std::endl;
        return 1;
    }
    ok = write_manifest(options, scene_path, source_poses, poses, trace_duration_s, source_implied_fps, timestamp_resampled, smoothing_applied, records) && ok;
    return ok ? 0 : 1;
}
