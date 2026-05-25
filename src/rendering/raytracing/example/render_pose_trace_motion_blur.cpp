#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
using json = nlohmann::json;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI DEFAULT_CAM_WIDTH = 2048;
static constexpr TI DEFAULT_CAM_HEIGHT = 2048;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;

template <TI SAMPLES, TI WIDTH, TI HEIGHT>
using RendererSpec = rlt::rendering::raytracing::Specification<T, TI, WIDTH, HEIGHT, NUM_CAMERAS, NUM_PROBES, rlt::rendering::raytracing::HighFidelityShading, true, SAMPLES, false, 1, rlt::rendering::raytracing::OutputMode::RGB>;

template <typename SPEC>
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

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
    std::string output_path = "raytracing_pose_trace_motion_blur.mp4";
    std::string ffmpeg = "ffmpeg";
    int width = DEFAULT_CAM_WIDTH;
    int height = DEFAULT_CAM_HEIGHT;
    int fps = 30;
    int max_frames = 0;
    int motion_blur_samples = 16;
    double shutter_time_s = -1.0;
    bool help = false;
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --trace pose_trace.json [options]\n"
        << "Options:\n"
        << "  --trace <path>                  Camera pose trace JSON from the interactive viewer\n"
        << "  --scene <path>                  Scene path override; otherwise read from trace JSON\n"
        << "  --output <path>                 Output MP4 path (default: raytracing_pose_trace_motion_blur.mp4)\n"
        << "  --resolution <n>                Square output resolution; supported: 2048 (default: 2048)\n"
        << "  --fps <n>                       Output frame rate (default: 30)\n"
        << "  --ffmpeg <path>                 ffmpeg binary (default: ffmpeg)\n"
        << "  --shutter-time-s <seconds>      Exposure interval ending at each frame time (default: 1/fps)\n"
        << "  --motion-blur-samples <n>       2, 4, 8, 16, or 32 temporal samples (default: 16)\n"
        << "  --max-frames <n>                Limit rendered frames when >0\n";
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

static bool valid_motion_blur_samples(int samples) {
    return samples == 2 || samples == 4 || samples == 8 || samples == 16 || samples == 32;
}

static bool valid_resolution(int width, int height) {
    return width == DEFAULT_CAM_WIDTH && height == DEFAULT_CAM_HEIGHT;
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
        else if(option_value(i, argc, argv, arg, "--output", value) || option_value(i, argc, argv, arg, "-o", value)) {
            options.output_path = value;
        }
        else if(option_value(i, argc, argv, arg, "--ffmpeg", value)) {
            options.ffmpeg = value;
        }
        else if(option_value(i, argc, argv, arg, "--resolution", value)) {
            int resolution = 0;
            if(!parse_int(value, resolution) || resolution <= 0) {
                std::cerr << "Invalid --resolution: " << value << std::endl;
                return false;
            }
            options.width = resolution;
            options.height = resolution;
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
        else if(option_value(i, argc, argv, arg, "--motion-blur-samples", value) || option_value(i, argc, argv, arg, "--samples", value)) {
            if(!parse_int(value, options.motion_blur_samples) || !valid_motion_blur_samples(options.motion_blur_samples)) {
                std::cerr << "Invalid --motion-blur-samples: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--shutter-time-s", value) || option_value(i, argc, argv, arg, "--shutter-time", value)) {
            if(!parse_double(value, options.shutter_time_s) || options.shutter_time_s < 0.0) {
                std::cerr << "Invalid --shutter-time-s: " << value << std::endl;
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
    if(options.shutter_time_s < 0.0) {
        options.shutter_time_s = 1.0 / static_cast<double>(options.fps);
    }
    if(!valid_resolution(options.width, options.height)) {
        std::cerr << "Unsupported resolution: " << options.width << "x" << options.height << std::endl;
        return false;
    }
    return true;
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

static bool normalize3(T v[3]) {
    const T norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if(norm < static_cast<T>(1e-6)) {
        return false;
    }
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    return true;
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
        if(!read_vec3((*source)["up"], pose.up) || !normalize3(pose.up)) {
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
        if(!read_vec3((*source)["forward"], forward) || !normalize3(forward)) {
            return false;
        }
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

static bool timestamped_trace(const std::vector<TracePose>& poses) {
    for(size_t i = 0; i < poses.size(); i++) {
        if(!poses[i].has_timestamp || !std::isfinite(poses[i].timestamp_s)) {
            return false;
        }
        if(i > 0 && poses[i].timestamp_s <= poses[i - 1].timestamp_s) {
            return false;
        }
    }
    return true;
}

static std::vector<double> trace_times(const std::vector<TracePose>& poses, int fps, bool& from_timestamps) {
    from_timestamps = timestamped_trace(poses);
    std::vector<double> times;
    times.reserve(poses.size());
    for(size_t i = 0; i < poses.size(); i++) {
        times.push_back(from_timestamps ? poses[i].timestamp_s : static_cast<double>(i) / static_cast<double>(fps));
    }
    return times;
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
    if(!normalize3(out)) {
        out[0] = static_cast<T>(1);
        out[1] = static_cast<T>(0);
        out[2] = static_cast<T>(0);
    }
}

static TracePose interpolate_pose(const TracePose& a, const TracePose& b, double a_time_s, double b_time_s, double target_time_s) {
    const double duration = b_time_s - a_time_s;
    const double t = duration > 0.0 ? std::min(std::max((target_time_s - a_time_s) / duration, 0.0), 1.0) : 0.0;

    TracePose out;
    lerp_vec3(a.eye, b.eye, t, out.eye);

    T forward_a[3];
    T forward_b[3];
    T forward[3];
    pose_forward(a, forward_a);
    pose_forward(b, forward_b);
    lerp_vec3(forward_a, forward_b, t, forward);
    if(!normalize3(forward)) {
        forward[0] = forward_a[0];
        forward[1] = forward_a[1];
        forward[2] = forward_a[2];
    }

    lerp_vec3(a.up, b.up, t, out.up);
    if(!normalize3(out.up)) {
        out.up[0] = static_cast<T>(0);
        out.up[1] = static_cast<T>(0);
        out.up[2] = static_cast<T>(1);
    }

    out.look_at[0] = out.eye[0] + forward[0];
    out.look_at[1] = out.eye[1] + forward[1];
    out.look_at[2] = out.eye[2] + forward[2];
    out.timestamp_s = target_time_s;
    out.has_timestamp = true;
    return out;
}

static TracePose pose_at(const std::vector<TracePose>& poses, const std::vector<double>& times, double target_time_s) {
    if(target_time_s <= times.front() || poses.size() == 1) {
        TracePose out = poses.front();
        out.timestamp_s = target_time_s;
        out.has_timestamp = true;
        return out;
    }
    if(target_time_s >= times.back()) {
        TracePose out = poses.back();
        out.timestamp_s = target_time_s;
        out.has_timestamp = true;
        return out;
    }
    const auto upper = std::upper_bound(times.begin(), times.end(), target_time_s);
    const size_t right = static_cast<size_t>(upper - times.begin());
    const size_t left = right - 1;
    return interpolate_pose(poses[left], poses[right], times[left], times[right], target_time_s);
}

static size_t output_frame_count(const std::vector<TracePose>& poses, const std::vector<double>& times, bool from_timestamps, int fps, int max_frames) {
    size_t count = poses.size();
    if(from_timestamps && times.size() >= 2) {
        const double duration_s = times.back() - times.front();
        count = static_cast<size_t>(std::floor(duration_s * static_cast<double>(fps))) + 1;
        if(count == 0) {
            count = 1;
        }
    }
    if(max_frames > 0) {
        count = std::min(count, static_cast<size_t>(max_frames));
    }
    return std::max<size_t>(count, 1);
}

template <typename SPEC>
static bool setup_renderer(DEVICE& device, Renderer<SPEC>& renderer, const std::string& scene_path) {
    rlt::malloc(device, renderer);
    if(!rlt::load_model(device, renderer, scene_path)) {
        return false;
    }
    rlt::upload_geometry(device, renderer);
    const T up[3] = {0, 0, 1};
    rlt::generate_cameras(device, renderer, renderer.scene_center, renderer.camera_radius, up, SPEC::COS_FOVY);
    rlt::build_pipeline(device, renderer);
    return true;
}

template <TI WIDTH, TI HEIGHT>
static std::string ffmpeg_command(const Options& options) {
    std::ostringstream cmd;
    cmd << shell_quote(options.ffmpeg)
        << " -hide_banner -loglevel error -y -f rawvideo -pixel_format rgba"
        << " -video_size " << WIDTH << "x" << HEIGHT
        << " -framerate " << options.fps
        << " -i - -an -c:v libx264 -pix_fmt yuv420p "
        << shell_quote(options.output_path);
    return cmd.str();
}

static bool write_frame(FILE* pipe, const std::vector<uint32_t>& frame, size_t frame_i) {
    const size_t bytes = frame.size() * sizeof(uint32_t);
    const size_t written = std::fwrite(frame.data(), 1, bytes, pipe);
    if(written != bytes) {
        std::cerr << "Failed writing frame " << frame_i << " to ffmpeg" << std::endl;
        return false;
    }
    return true;
}

static bool write_manifest(const Options& options, const std::string& scene_path, TI width, TI height, size_t source_frames, size_t rendered_frames, bool from_timestamps, double trace_start_s, double trace_end_s, const std::string& command, int ffmpeg_status, bool ok) {
    json manifest;
    manifest["trace_path"] = options.trace_path;
    manifest["scene_path"] = scene_path;
    manifest["output_path"] = options.output_path;
    manifest["fps"] = options.fps;
    manifest["width"] = width;
    manifest["height"] = height;
    manifest["shutter_time_s"] = options.shutter_time_s;
    manifest["motion_blur_samples"] = options.motion_blur_samples;
    manifest["source_frames"] = source_frames;
    manifest["rendered_frames"] = rendered_frames;
    manifest["trace_has_timestamps"] = from_timestamps;
    manifest["trace_start_s"] = trace_start_s;
    manifest["trace_end_s"] = trace_end_s;
    manifest["ffmpeg_command"] = command;
    manifest["ffmpeg_status"] = ffmpeg_status;
    manifest["ok"] = ok;

    const std::string manifest_path = options.output_path + ".manifest.json";
    std::ofstream f(manifest_path);
    if(!f) {
        std::cerr << "Failed to write manifest: " << manifest_path << std::endl;
        return false;
    }
    f << manifest.dump(2) << "\n";
    std::cout << "Wrote " << manifest_path << std::endl;
    return true;
}

template <TI SAMPLES, TI WIDTH, TI HEIGHT>
static bool render_trace(DEVICE& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, const std::vector<double>& times, bool from_timestamps) {
    using SPEC = RendererSpec<SAMPLES, WIDTH, HEIGHT>;

    Renderer<SPEC> renderer;
    if(!setup_renderer(device, renderer, scene_path)) {
        std::cerr << "Failed to initialize renderer for scene: " << scene_path << std::endl;
        return false;
    }

    const size_t frame_count = output_frame_count(poses, times, from_timestamps, options.fps, options.max_frames);
    const std::string command = ffmpeg_command<WIDTH, HEIGHT>(options);
    FILE* pipe = popen(command.c_str(), "w");
    if(pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg" << std::endl;
        rlt::free(device, renderer);
        return false;
    }

    std::cout << "Writing " << options.output_path << std::endl;
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT
              << ", fps: " << options.fps
              << ", shutter: " << options.shutter_time_s << "s"
              << ", motion blur samples: " << SAMPLES << std::endl;

    std::vector<uint32_t> frame(static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT));
    bool ok = true;
    const double start_s = times.front();
    const double end_s = times.back();
    const T aspect = static_cast<T>(WIDTH) / static_cast<T>(HEIGHT);
    for(size_t frame_i = 0; frame_i < frame_count; frame_i++) {
        const double close_t = std::min(start_s + static_cast<double>(frame_i) / static_cast<double>(options.fps), end_s);
        const double open_t = std::max(start_s, close_t - options.shutter_time_s);
        const TracePose open_pose = pose_at(poses, times, open_t);
        const TracePose close_pose = pose_at(poses, times, close_t);

        rlt::set(device, renderer.cameras_open, rlt::make_camera_data(open_pose.eye, open_pose.look_at, open_pose.up, SPEC::COS_FOVY, aspect), static_cast<TI>(0));
        rlt::set(device, renderer.cameras, rlt::make_camera_data(close_pose.eye, close_pose.look_at, close_pose.up, SPEC::COS_FOVY, aspect), static_cast<TI>(0));
        rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
        rlt::render_rgb_only(device, renderer);
        rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);

        const uint32_t* fb_data = rlt::data(renderer.frame_buffer);
        std::memcpy(frame.data(), fb_data, frame.size() * sizeof(uint32_t));
        if(!write_frame(pipe, frame, frame_i)) {
            ok = false;
            break;
        }
    }

    const int ffmpeg_status = pclose(pipe);
    ok = ok && ffmpeg_status == 0;
    if(!ok) {
        std::cerr << "ffmpeg failed with status " << ffmpeg_status << std::endl;
    }
    else {
        std::cout << "video written: " << options.output_path << std::endl;
    }

    ok = write_manifest(options, scene_path, WIDTH, HEIGHT, poses.size(), frame_count, from_timestamps, start_s, end_s, command, ffmpeg_status, ok) && ok;
    rlt::free(device, renderer);
    return ok;
}

template <TI WIDTH, TI HEIGHT>
static bool dispatch_samples(DEVICE& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, const std::vector<double>& times, bool from_timestamps) {
    switch(options.motion_blur_samples) {
        case 2:
            return render_trace<2, WIDTH, HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
        case 4:
            return render_trace<4, WIDTH, HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
        case 8:
            return render_trace<8, WIDTH, HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
        case 16:
            return render_trace<16, WIDTH, HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
        case 32:
            return render_trace<32, WIDTH, HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
        default:
            std::cerr << "Unsupported motion blur sample count: " << options.motion_blur_samples << std::endl;
            return false;
    }
}

static bool dispatch_render(DEVICE& device, const Options& options, const std::string& scene_path, const std::vector<TracePose>& poses, const std::vector<double>& times, bool from_timestamps) {
    return dispatch_samples<DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT>(device, options, scene_path, poses, times, from_timestamps);
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

    std::string scene_path;
    std::vector<TracePose> poses;
    if(!load_trace(options, scene_path, poses)) {
        return 1;
    }

    bool from_timestamps = false;
    const std::vector<double> times = trace_times(poses, options.fps, from_timestamps);
    if(from_timestamps) {
        std::cout << "Using timestamped trace with " << poses.size() << " poses over " << (times.back() - times.front()) << "s" << std::endl;
    }
    else {
        std::cout << "Trace has no strictly increasing timestamps; treating poses as " << options.fps << " fps samples" << std::endl;
    }

    DEVICE device;
    rlt::init(device);
    return dispatch_render(device, options, scene_path, poses, times, from_timestamps) ? 0 : 1;
}
