#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_ALL -1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_ALL
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS 4096
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_WIDTH
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_WIDTH 64
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_HEIGHT
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_HEIGHT 64
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_FRAME_TOOL
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_FRAME_TOOL 0
#endif

#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_BASIC 0
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_HIGH_FIDELITY 1
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_FAST_FLAT 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE
#define RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_BASIC
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace rlt = rl_tools;

using T = float;
using TI = int;

static constexpr const char* OBJECTS20_LAYOUT_NAME = "canonical_staggered_v1";
static constexpr TI NUM_ENVS = RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS;
static constexpr TI CAM_WIDTH = RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_WIDTH;
static constexpr TI CAM_HEIGHT = RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_HEIGHT;
template <int T_PROFILE>
struct BenchmarkShadingProfile;
template <>
struct BenchmarkShadingProfile<RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_BASIC> {
    using type = rlt::rendering::raytracing::BasicShading;
};
template <>
struct BenchmarkShadingProfile<RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_HIGH_FIDELITY> {
    using type = rlt::rendering::raytracing::HighFidelityShading;
};
template <>
struct BenchmarkShadingProfile<RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_FAST_FLAT> {
    using type = rlt::rendering::raytracing::FastFlatShading;
};
using ShadingProfile = typename BenchmarkShadingProfile<RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE>::type;
template <rlt::rendering::raytracing::OutputMode T_OUTPUT_MODE>
using BenchmarkSpec = rlt::rendering::raytracing::Specification<T, TI, CAM_WIDTH, CAM_HEIGHT, NUM_ENVS, 1, ShadingProfile, false, 1, false, 1, T_OUTPUT_MODE>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

enum class SceneAxis { OBJECTS_20, PROCTHOR };
enum class StepAxis { RENDER_ONLY, RENDER_PHYSICS };
enum class OutputAxis { RGB, DEPTH };
enum class OrientationMode { LOOK_AT_SCENE_JITTER, RANDOM_YAW_PITCH, UNIFORM_SO3 };

struct CameraOffset {
    T x;
    T y;
    T z;
};

static CameraOffset camera_offset(SceneAxis scene) {
    return scene == SceneAxis::OBJECTS_20
        ? CameraOffset{static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}
        : CameraOffset{static_cast<T>(-3.92), static_cast<T>(-5.67), static_cast<T>(1.0)};
}

struct Options {
    std::string scene = "all";
    std::string step_mode = "all";
    std::string output = "all";
    std::string orientation_mode = "uniform_so3";
    std::string gpu_label;
    std::string output_dir = ".";
    std::string output_png = "hyperdrone_procthor_frame.png";
    std::string procthor_path = "tests/data/ProcTHOR-Train-1.glb";
    double seconds = 10.0;
    double warmup_seconds = 2.0;
    int iterations = 0;
    int warmup_iterations = 10;
    int sync_interval = 10;
    int num_envs = NUM_ENVS;
    int resolution = CAM_WIDTH;
    uint32_t seed = 0;
    bool has_position = false;
    bool has_forward = false;
    bool has_look_at = false;
    bool has_up = false;
    bool has_orientation = false;
    T position[3] = {-3.92f, -5.67f, 1.0f};
    T forward[3] = {1.0f, 0.0f, 0.0f};
    T look_at[3] = {0.0f, 0.0f, 0.0f};
    T up[3] = {0.0f, 0.0f, 1.0f};
    T orientation_wxyz[4] = {1.0f, 0.0f, 0.0f, 0.0f};
};

struct CameraPose {
    T direction[3];
    T up[3];
};

struct BenchmarkResult {
    int iterations;
    double elapsed_s;
    double frames_per_s;
    double pixels_per_s;
    double mrays_per_s;
};

struct FrameStats {
    bool plausible;
    int bad_frames;
    double min_value;
    double max_value;
    double mean_value;
};

static constexpr double RAD_TO_DEG = 57.29577951308232;

static bool has_prefix(const std::string& value, const char* prefix) {
    return value.compare(0, std::strlen(prefix), prefix) == 0;
}

static std::string value_after_prefix(const std::string& value, const char* prefix) {
    return value.substr(std::strlen(prefix));
}

static void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --scene <all|20_objects|procthor>\n"
        << "  --step-mode <all|render_only|render_physics>\n"
        << "  --output <all|rgb|depth>\n"
        << "  --orientation-mode <uniform_so3|random_yaw_pitch|look_at_scene_jitter>\n"
        << "  --gpu-label <label>\n"
        << "  --seconds <seconds>              Timed duration per combination (default: 10)\n"
        << "  --iterations <count>             Fixed timed iterations; overrides --seconds when >0\n"
        << "  --warmup-seconds <seconds>       Untimed warmup duration before timing (default: 2)\n"
        << "  --warmup-iterations <count>      Legacy warmup count used when --warmup-seconds=0 (default: 10)\n"
        << "  --sync-interval <count>          Render-only async sync interval (default: 10)\n"
        << "  --seed <count>                   Deterministic camera-orientation seed (default: 0)\n"
        << "  --num-envs <count>               Must match compile-time NUM_ENVS=" << NUM_ENVS << "\n"
        << "  --resolution <pixels>            Must match compile-time resolution "
        << CAM_WIDTH << "x" << CAM_HEIGHT << "\n"
        << "  --output-dir <dir>\n"
        << "  --procthor-path <file>\n";
#if RL_TOOLS_RENDERING_RAYTRACING_SIM_FRAME_TOOL
    std::cout
        << "Frame target options:\n"
        << "  --output-png <file>              Single-frame PNG path\n"
        << "  --position-flu X Y Z             Camera position in ProcTHOR FLU coordinates\n"
        << "  --forward-flu X Y Z              Camera forward direction in FLU coordinates\n"
        << "  --look-at-flu X Y Z              Look-at target in FLU coordinates\n"
        << "  --up-flu X Y Z                   Camera up vector in FLU coordinates\n"
        << "  --orientation-flu-wxyz W X Y Z   Camera body quaternion in FLU coordinates\n";
#endif
}

static bool parse_int(const std::string& value, int& out) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if(end == value.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static bool parse_double(const std::string& value, double& out) {
    char* end = nullptr;
    double parsed = std::strtod(value.c_str(), &end);
    if(end == value.c_str() || *end != '\0') {
        return false;
    }
    out = parsed;
    return true;
}

static bool parse_vec3_csv(const std::string& value, T out[3]) {
    size_t start = 0;
    for(int i = 0; i < 3; i++) {
        const size_t end = value.find(',', start);
        const std::string part = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        double parsed = 0.0;
        if(!parse_double(part, parsed)) {
            return false;
        }
        out[i] = static_cast<T>(parsed);
        if(i < 2) {
            if(end == std::string::npos) {
                return false;
            }
            start = end + 1;
        }
        else if(end != std::string::npos) {
            return false;
        }
    }
    return true;
}

static bool parse_vec3_args(int& i, int argc, char** argv, const char* name, T out[3]) {
    if(i + 3 >= argc) {
        std::cerr << "Missing values for " << name << std::endl;
        return false;
    }
    for(int component = 0; component < 3; component++) {
        double parsed = 0.0;
        if(!parse_double(argv[++i], parsed)) {
            std::cerr << "Invalid " << name << " component: " << argv[i] << std::endl;
            return false;
        }
        out[component] = static_cast<T>(parsed);
    }
    return true;
}

static bool get_vec3_option(int& i, int argc, char** argv, const std::string& arg, const char* name, T out[3], bool& seen) {
    const std::string eq_prefix = std::string(name) + "=";
    if(has_prefix(arg, eq_prefix.c_str())) {
        const std::string value = value_after_prefix(arg, eq_prefix.c_str());
        if(!parse_vec3_csv(value, out)) {
            std::cerr << "Invalid " << name << ": " << value << std::endl;
            return false;
        }
        seen = true;
        return true;
    }
    if(arg == name) {
        if(!parse_vec3_args(i, argc, argv, name, out)) {
            return false;
        }
        seen = true;
        return true;
    }
    return false;
}

static bool parse_vec4_csv(const std::string& value, T out[4]) {
    size_t start = 0;
    for(int i = 0; i < 4; i++) {
        const size_t end = value.find(',', start);
        const std::string part = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        double parsed = 0.0;
        if(!parse_double(part, parsed)) {
            return false;
        }
        out[i] = static_cast<T>(parsed);
        if(i < 3) {
            if(end == std::string::npos) {
                return false;
            }
            start = end + 1;
        }
        else if(end != std::string::npos) {
            return false;
        }
    }
    return true;
}

static bool parse_vec4_args(int& i, int argc, char** argv, const char* name, T out[4]) {
    if(i + 4 >= argc) {
        std::cerr << "Missing values for " << name << std::endl;
        return false;
    }
    for(int component = 0; component < 4; component++) {
        double parsed = 0.0;
        if(!parse_double(argv[++i], parsed)) {
            std::cerr << "Invalid " << name << " component: " << argv[i] << std::endl;
            return false;
        }
        out[component] = static_cast<T>(parsed);
    }
    return true;
}

static bool get_vec4_option(int& i, int argc, char** argv, const std::string& arg, const char* name, T out[4], bool& seen) {
    const std::string eq_prefix = std::string(name) + "=";
    if(has_prefix(arg, eq_prefix.c_str())) {
        const std::string value = value_after_prefix(arg, eq_prefix.c_str());
        if(!parse_vec4_csv(value, out)) {
            std::cerr << "Invalid " << name << ": " << value << std::endl;
            return false;
        }
        seen = true;
        return true;
    }
    if(arg == name) {
        if(!parse_vec4_args(i, argc, argv, name, out)) {
            return false;
        }
        seen = true;
        return true;
    }
    return false;
}

static bool get_option_value(int& i, int argc, char** argv, const std::string& arg, const char* name, std::string& out) {
    std::string eq_prefix = std::string(name) + "=";
    if(has_prefix(arg, eq_prefix.c_str())) {
        out = value_after_prefix(arg, eq_prefix.c_str());
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
        std::string arg = argv[i];
        std::string value;
        if(arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        }
        else if(get_option_value(i, argc, argv, arg, "--scene", value)) {
            options.scene = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--step-mode", value)) {
            options.step_mode = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output", value)) {
            options.output = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--orientation-mode", value)) {
            options.orientation_mode = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--gpu-label", value)) {
            options.gpu_label = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output-dir", value)) {
            options.output_dir = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output-png", value)) {
            options.output_png = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--procthor-path", value)) {
            options.procthor_path = value;
        }
        else if(get_vec3_option(i, argc, argv, arg, "--position-flu", options.position, options.has_position)) {
        }
        else if(get_vec3_option(i, argc, argv, arg, "--forward-flu", options.forward, options.has_forward)) {
        }
        else if(get_vec3_option(i, argc, argv, arg, "--look-at-flu", options.look_at, options.has_look_at)) {
        }
        else if(get_vec3_option(i, argc, argv, arg, "--up-flu", options.up, options.has_up)) {
        }
        else if(get_vec4_option(i, argc, argv, arg, "--orientation-flu-wxyz", options.orientation_wxyz, options.has_orientation)) {
        }
        else if(get_option_value(i, argc, argv, arg, "--seconds", value)) {
            if(!parse_double(value, options.seconds)) {
                std::cerr << "Invalid --seconds: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--iterations", value)) {
            if(!parse_int(value, options.iterations)) {
                std::cerr << "Invalid --iterations: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--warmup-seconds", value)) {
            if(!parse_double(value, options.warmup_seconds)) {
                std::cerr << "Invalid --warmup-seconds: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--warmup-iterations", value)) {
            if(!parse_int(value, options.warmup_iterations)) {
                std::cerr << "Invalid --warmup-iterations: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--sync-interval", value)) {
            if(!parse_int(value, options.sync_interval)) {
                std::cerr << "Invalid --sync-interval: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--num-envs", value)) {
            if(!parse_int(value, options.num_envs)) {
                std::cerr << "Invalid --num-envs: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--resolution", value)) {
            if(!parse_int(value, options.resolution)) {
                std::cerr << "Invalid --resolution: " << value << std::endl;
                return false;
            }
        }
        else if(get_option_value(i, argc, argv, arg, "--seed", value)) {
            int parsed_seed = 0;
            if(!parse_int(value, parsed_seed) || parsed_seed < 0) {
                std::cerr << "Invalid --seed: " << value << std::endl;
                return false;
            }
            options.seed = static_cast<uint32_t>(parsed_seed);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if(options.num_envs != NUM_ENVS) {
        std::cerr << "--num-envs is compile-time for this renderer. Requested " << options.num_envs
                  << ", but this target was built with " << NUM_ENVS << "." << std::endl;
        return false;
    }
    if(options.resolution != CAM_WIDTH || CAM_WIDTH != CAM_HEIGHT) {
        std::cerr << "This target was built for " << CAM_WIDTH << "x" << CAM_HEIGHT
                  << ". Requested square resolution " << options.resolution << "." << std::endl;
        return false;
    }
    if(options.has_forward && options.has_look_at) {
        std::cerr << "Use only one of --forward-flu or --look-at-flu." << std::endl;
        return false;
    }
    if(options.has_orientation && (options.has_forward || options.has_look_at)) {
        std::cerr << "Use --orientation-flu-wxyz without --forward-flu or --look-at-flu." << std::endl;
        return false;
    }
    if(options.seconds <= 0 && options.iterations <= 0) {
        std::cerr << "Either --seconds must be > 0 or --iterations must be > 0." << std::endl;
        return false;
    }
    if(options.warmup_iterations < 0) {
        std::cerr << "--warmup-iterations must be >= 0." << std::endl;
        return false;
    }
    if(options.warmup_seconds < 0) {
        std::cerr << "--warmup-seconds must be >= 0." << std::endl;
        return false;
    }
    if(options.sync_interval <= 0) {
        std::cerr << "--sync-interval must be > 0." << std::endl;
        return false;
    }
    if(options.orientation_mode != "random_yaw_pitch" &&
       options.orientation_mode != "look_at_scene_jitter" &&
       options.orientation_mode != "uniform_so3" &&
       options.orientation_mode != "uniform_so3_once_per_camera") {
        std::cerr << "Unsupported --orientation-mode: " << options.orientation_mode << std::endl;
        return false;
    }
    return true;
}

static const char* scene_name(SceneAxis scene) {
    return scene == SceneAxis::OBJECTS_20 ? "20_objects" : "procthor";
}

static const char* step_name(StepAxis step) {
    return step == StepAxis::RENDER_ONLY ? "render_only" : "render_physics";
}

static OrientationMode orientation_mode(const Options& options) {
    if(options.orientation_mode == "look_at_scene_jitter") {
        return OrientationMode::LOOK_AT_SCENE_JITTER;
    }
    if(options.orientation_mode == "uniform_so3" || options.orientation_mode == "uniform_so3_once_per_camera") {
        return OrientationMode::UNIFORM_SO3;
    }
    return OrientationMode::RANDOM_YAW_PITCH;
}

static const char* orientation_mode_name(OrientationMode mode) {
    return mode == OrientationMode::LOOK_AT_SCENE_JITTER ? "look_at_scene_jitter"
        : (mode == OrientationMode::UNIFORM_SO3 ? "uniform_so3_once_per_camera" : "random_yaw_pitch");
}

static constexpr const char* output_name() {
    return RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH ? "depth"
        : (RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB ? "rgb" : "all");
}

static constexpr const char* shading_profile_name() {
    return RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE == RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_FAST_FLAT ? "fast_flat"
        : (RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE == RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_HIGH_FIDELITY ? "high_fidelity" : "basic");
}

template <typename SPEC>
static constexpr const char* output_name_for_spec() {
    return SPEC::HAS_DEPTH ? "depth" : "rgb";
}

static bool select_scenes(const std::string& requested, std::vector<SceneAxis>& scenes) {
    if(requested == "all") {
        scenes = {SceneAxis::OBJECTS_20, SceneAxis::PROCTHOR};
    }
    else if(requested == "20_objects" || requested == "20-objects" || requested == "objects20") {
        scenes = {SceneAxis::OBJECTS_20};
    }
    else if(requested == "procthor") {
        scenes = {SceneAxis::PROCTHOR};
    }
    else {
        std::cerr << "Unsupported --scene: " << requested << std::endl;
        return false;
    }
    return true;
}

static bool select_steps(const std::string& requested, std::vector<StepAxis>& steps) {
    if(requested == "all") {
        steps = {StepAxis::RENDER_ONLY, StepAxis::RENDER_PHYSICS};
    }
    else if(requested == "render_only" || requested == "render-only") {
        steps = {StepAxis::RENDER_ONLY};
    }
    else if(requested == "render_physics" || requested == "render+physics" || requested == "render-physics") {
        steps = {StepAxis::RENDER_PHYSICS};
    }
    else {
        std::cerr << "Unsupported --step-mode: " << requested << std::endl;
        return false;
    }
    return true;
}

static bool select_outputs(const std::string& requested, std::vector<OutputAxis>& outputs) {
    if(requested == "all") {
        outputs = {OutputAxis::RGB, OutputAxis::DEPTH};
    }
    else if(requested == "rgb") {
        outputs = {OutputAxis::RGB};
    }
    else if(requested == "depth") {
        outputs = {OutputAxis::DEPTH};
    }
    else {
        std::cerr << "Unsupported --output: " << requested << std::endl;
        return false;
    }
    return true;
}

static bool ensure_output_dir(const std::string& dir) {
    if(dir.empty() || dir == ".") {
        return true;
    }
    if(mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
    std::cerr << "Failed to create output directory '" << dir << "': " << std::strerror(errno) << std::endl;
    return false;
}

static std::string sanitize_label(const std::string& input) {
    std::string out;
    for(char ch : input) {
        if((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out.push_back(ch);
        }
        else if(ch == '-' || ch == '_') {
            out.push_back(ch);
        }
        else if(ch == ' ' || ch == '/' || ch == '+') {
            out.push_back('-');
        }
    }
    return out.empty() ? "gpu" : out;
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if(dir.empty() || dir == ".") {
        return file;
    }
    if(dir[dir.size() - 1] == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

static void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path fs_path(path);
    const std::filesystem::path parent = fs_path.parent_path();
    if(!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

static std::string csv_quote(const std::string& input) {
    std::string out = "\"";
    for(char ch : input) {
        if(ch == '"') {
            out += "\"\"";
        }
        else {
            out.push_back(ch);
        }
    }
    out += '"';
    return out;
}

static std::string cuda_device_name() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.name;
    }
    return "unknown";
}

template <typename SPEC>
static void add_box(rlt::rendering::raytracing::Renderer<SPEC>& renderer, T cx, T cy, T cz, T sx, T sy, T sz, T r, T g, T b) {
    rlt::rendering::raytracing::MeshData<SPEC> md;
    md.color[0] = r;
    md.color[1] = g;
    md.color[2] = b;
    const T x0 = cx - sx * static_cast<T>(0.5);
    const T x1 = cx + sx * static_cast<T>(0.5);
    const T y0 = cy - sy * static_cast<T>(0.5);
    const T y1 = cy + sy * static_cast<T>(0.5);
    const T z0 = cz - sz * static_cast<T>(0.5);
    const T z1 = cz + sz * static_cast<T>(0.5);
    const T vertices[] = {
        x0, y0, z0,  x1, y0, z0,  x1, y1, z0,  x0, y1, z0,
        x0, y0, z1,  x1, y0, z1,  x1, y1, z1,  x0, y1, z1
    };
    const int indices[] = {
        0, 1, 2,  0, 2, 3,
        4, 6, 5,  4, 7, 6,
        0, 4, 5,  0, 5, 1,
        1, 5, 6,  1, 6, 2,
        2, 6, 7,  2, 7, 3,
        3, 7, 4,  3, 4, 0
    };
    md.vertices.assign(vertices, vertices + sizeof(vertices) / sizeof(vertices[0]));
    md.indices.assign(indices, indices + sizeof(indices) / sizeof(indices[0]));
    renderer.meshes.push_back(std::move(md));
}

template <typename SPEC>
static void add_sphere(rlt::rendering::raytracing::Renderer<SPEC>& renderer, T cx, T cy, T cz, T radius, T r, T g, T b) {
    static constexpr int SEGMENTS = 16;
    static constexpr int RINGS = 8;
    static constexpr T PI = static_cast<T>(3.14159265358979323846);
    rlt::rendering::raytracing::MeshData<SPEC> md;
    md.color[0] = r;
    md.color[1] = g;
    md.color[2] = b;
    md.vertices.reserve(static_cast<size_t>((RINGS + 1) * SEGMENTS * 3));
    md.normals.reserve(static_cast<size_t>((RINGS + 1) * SEGMENTS * 3));
    md.indices.reserve(static_cast<size_t>(RINGS * SEGMENTS * 6));

    for(int ring = 0; ring <= RINGS; ring++) {
        const T theta = PI * static_cast<T>(ring) / static_cast<T>(RINGS);
        const T sin_theta = std::sin(theta);
        const T cos_theta = std::cos(theta);
        for(int segment = 0; segment < SEGMENTS; segment++) {
            const T phi = static_cast<T>(2) * PI * static_cast<T>(segment) / static_cast<T>(SEGMENTS);
            const T nx = sin_theta * std::cos(phi);
            const T ny = sin_theta * std::sin(phi);
            const T nz = cos_theta;
            md.vertices.push_back(cx + radius * nx);
            md.vertices.push_back(cy + radius * ny);
            md.vertices.push_back(cz + radius * nz);
            md.normals.push_back(nx);
            md.normals.push_back(ny);
            md.normals.push_back(nz);
        }
    }

    for(int ring = 0; ring < RINGS; ring++) {
        for(int segment = 0; segment < SEGMENTS; segment++) {
            const int next_segment = (segment + 1) % SEGMENTS;
            const int a = ring * SEGMENTS + segment;
            const int b = ring * SEGMENTS + next_segment;
            const int c = (ring + 1) * SEGMENTS + next_segment;
            const int d = (ring + 1) * SEGMENTS + segment;
            md.indices.push_back(a);
            md.indices.push_back(d);
            md.indices.push_back(c);
            md.indices.push_back(a);
            md.indices.push_back(c);
            md.indices.push_back(b);
        }
    }

    renderer.meshes.push_back(std::move(md));
}

struct Objects20Position {
    T x;
    T y;
    T z;
};

static constexpr Objects20Position OBJECTS20_POSITIONS[] = {
    {-1.5f, -2.0f, 0.35f},
    {-0.4f, -2.5f, 0.55f},
    {0.9f, -2.2f, 0.45f},
    {2.2f, -1.8f, 0.70f},
    {3.2f, -0.8f, 0.35f},
    {-2.2f, -0.9f, 0.60f},
    {-0.9f, -0.8f, 0.45f},
    {0.4f, -0.6f, 0.80f},
    {1.8f, -0.3f, 0.50f},
    {3.0f, 0.4f, 0.65f},
    {-2.9f, 0.6f, 0.45f},
    {-1.4f, 0.8f, 0.75f},
    {0.1f, 1.0f, 0.50f},
    {1.4f, 1.2f, 0.60f},
    {2.8f, 1.5f, 0.40f},
    {-2.4f, 2.2f, 0.70f},
    {-0.8f, 2.4f, 0.35f},
    {0.8f, 2.5f, 0.75f},
    {2.2f, 2.4f, 0.45f},
    {3.4f, 2.0f, 0.65f},
};

static void objects20_scale(int i, T& sx, T& sy, T& sz) {
    if((i % 2) == 0) {
        sx = static_cast<T>(0.35 + 0.05 * (i % 3));
        sy = static_cast<T>(0.45 + 0.04 * ((i + 1) % 4));
        sz = static_cast<T>(0.55 + 0.05 * ((i + 2) % 5));
    }
    else {
        const T radius_scale = static_cast<T>(0.45 + 0.04 * (i % 4));
        sx = radius_scale;
        sy = radius_scale;
        sz = radius_scale;
    }
}

static void objects20_color(int i, T& r, T& g, T& b) {
    static constexpr T palette[][3] = {
        {0.85f, 0.20f, 0.18f},
        {0.15f, 0.48f, 0.88f},
        {0.95f, 0.72f, 0.20f},
        {0.18f, 0.64f, 0.39f},
        {0.70f, 0.34f, 0.82f},
    };
    const int index = i % 5;
    r = palette[index][0];
    g = palette[index][1];
    b = palette[index][2];
}

template <typename SPEC>
static void make_20_object_scene(rlt::rendering::raytracing::Renderer<SPEC>& renderer) {
    renderer.meshes.clear();
    for(int i = 0; i < 20; i++) {
        T sx, sy, sz, r, g, b;
        objects20_scale(i, sx, sy, sz);
        objects20_color(i, r, g, b);
        const auto& position = OBJECTS20_POSITIONS[i];
        if((i % 2) == 0) {
            add_box(renderer, position.x, position.y, position.z, sx, sy, sz, r, g, b);
        }
        else {
            add_sphere(renderer, position.x, position.y, position.z, sx * static_cast<T>(0.5), r, g, b);
        }
    }
    renderer.scene_center[0] = static_cast<T>(0.25);
    renderer.scene_center[1] = 0;
    renderer.scene_center[2] = static_cast<T>(0.60);
    renderer.scene_half_extent[0] = static_cast<T>(3.65);
    renderer.scene_half_extent[1] = static_cast<T>(2.75);
    renderer.scene_half_extent[2] = static_cast<T>(0.95);
    renderer.camera_radius = static_cast<T>(6.0);
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

static void normalize_quaternion(T q[4]) {
    const T norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if(norm < static_cast<T>(1e-6)) {
        q[0] = static_cast<T>(1);
        q[1] = static_cast<T>(0);
        q[2] = static_cast<T>(0);
        q[3] = static_cast<T>(0);
        return;
    }
    q[0] /= norm;
    q[1] /= norm;
    q[2] /= norm;
    q[3] /= norm;
}

static void cross(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void rotate_by_quaternion(T w, T x, T y, T z, const T v[3], T out[3]) {
    const T qv[3] = {x, y, z};
    T t[3];
    cross(qv, v, t);
    t[0] *= static_cast<T>(2);
    t[1] *= static_cast<T>(2);
    t[2] *= static_cast<T>(2);
    T q_cross_t[3];
    cross(qv, t, q_cross_t);
    out[0] = v[0] + w * t[0] + q_cross_t[0];
    out[1] = v[1] + w * t[1] + q_cross_t[1];
    out[2] = v[2] + w * t[2] + q_cross_t[2];
}

template <typename SPEC>
static std::vector<CameraPose> make_camera_poses(const rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, const Options& options) {
    static constexpr T PI = static_cast<T>(3.14159265358979323846);
    const OrientationMode mode = orientation_mode(options);
    std::vector<CameraPose> poses(SPEC::NUM_CAMERAS);
    std::mt19937 rng(options.seed);
    std::uniform_real_distribution<T> unit_dist(static_cast<T>(0), static_cast<T>(1));
    std::uniform_real_distribution<T> look_at_yaw_dist(static_cast<T>(-0.08), static_cast<T>(0.08));
    std::uniform_real_distribution<T> look_at_pitch_dist(static_cast<T>(-0.06), static_cast<T>(0.06));
    std::uniform_real_distribution<T> yaw_dist(-PI, PI);
    std::uniform_real_distribution<T> pitch_dist(static_cast<T>(-0.35), static_cast<T>(0.35));

    const CameraOffset offset = camera_offset(scene);
    const T eye[3] = {offset.x, offset.y, offset.z};
    const T target[3] = {renderer.scene_center[0], renderer.scene_center[1], renderer.scene_center[2]};
    const T world_up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};

    T look_at_forward[3] = {
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2]
    };
    normalize(look_at_forward);
    T look_at_right[3];
    cross(look_at_forward, world_up, look_at_right);
    normalize(look_at_right);

    for(int i = 0; i < static_cast<int>(SPEC::NUM_CAMERAS); i++) {
        CameraPose& pose = poses[i];
        if(mode == OrientationMode::LOOK_AT_SCENE_JITTER) {
            const T yaw_offset = look_at_yaw_dist(rng);
            const T pitch_offset = look_at_pitch_dist(rng);
            pose.direction[0] = look_at_forward[0] + yaw_offset * look_at_right[0] + pitch_offset * world_up[0];
            pose.direction[1] = look_at_forward[1] + yaw_offset * look_at_right[1] + pitch_offset * world_up[1];
            pose.direction[2] = look_at_forward[2] + yaw_offset * look_at_right[2] + pitch_offset * world_up[2];
            normalize(pose.direction);
            pose.up[0] = world_up[0];
            pose.up[1] = world_up[1];
            pose.up[2] = world_up[2];
        }
        else if(mode == OrientationMode::UNIFORM_SO3) {
            const T u1 = unit_dist(rng);
            const T u2 = unit_dist(rng);
            const T u3 = unit_dist(rng);
            const T qx = std::sqrt(static_cast<T>(1) - u1) * std::sin(static_cast<T>(2) * PI * u2);
            const T qy = std::sqrt(static_cast<T>(1) - u1) * std::cos(static_cast<T>(2) * PI * u2);
            const T qz = std::sqrt(u1) * std::sin(static_cast<T>(2) * PI * u3);
            const T qw = std::sqrt(u1) * std::cos(static_cast<T>(2) * PI * u3);
            const T local_forward[3] = {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)};
            const T local_up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
            rotate_by_quaternion(qw, qx, qy, qz, local_forward, pose.direction);
            rotate_by_quaternion(qw, qx, qy, qz, local_up, pose.up);
            normalize(pose.direction);
            normalize(pose.up);
        }
        else {
            const T yaw = yaw_dist(rng);
            const T pitch = pitch_dist(rng);
            const T cp = std::cos(pitch);
            const T sp = std::sin(pitch);
            const T cy = std::cos(yaw);
            const T sy = std::sin(yaw);
            pose.direction[0] = cp * cy;
            pose.direction[1] = cp * sy;
            pose.direction[2] = sp;
            pose.up[0] = -sp * cy;
            pose.up[1] = -sp * sy;
            pose.up[2] = cp;
        }
    }
    return poses;
}

template <typename DEVICE, typename SPEC>
static void write_cameras(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, const std::vector<CameraPose>& poses) {
    const CameraOffset offset = camera_offset(scene);
    const T eye[3] = {offset.x, offset.y, offset.z};
    const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    for(int i = 0; i < static_cast<int>(SPEC::NUM_CAMERAS); i++) {
        const CameraPose& pose = poses[i];
        const T look_at[3] = {
            eye[0] + pose.direction[0],
            eye[1] + pose.direction[1],
            eye[2] + pose.direction[2]
        };
        rlt::set(device, renderer.cameras, rlt::make_camera_data(eye, look_at, pose.up, SPEC::COS_FOVY, aspect), static_cast<TI>(i));
    }
}

template <typename DEVICE, typename SPEC>
static void write_single_camera(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, const T eye[3], const CameraPose& pose) {
    static_assert(SPEC::NUM_CAMERAS == 1, "single-frame target must be compiled with one camera");
    const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    const T look_at[3] = {
        eye[0] + pose.direction[0],
        eye[1] + pose.direction[1],
        eye[2] + pose.direction[2]
    };
    rlt::set(device, renderer.cameras, rlt::make_camera_data(eye, look_at, pose.up, SPEC::COS_FOVY, aspect), static_cast<TI>(0));
}

template <typename SPEC>
static CameraPose make_single_frame_pose(const rlt::rendering::raytracing::Renderer<SPEC>& renderer, const Options& options) {
    CameraPose pose{};
    const T eye[3] = {options.position[0], options.position[1], options.position[2]};
    if(options.has_orientation) {
        T q[4] = {
            options.orientation_wxyz[0],
            options.orientation_wxyz[1],
            options.orientation_wxyz[2],
            options.orientation_wxyz[3]
        };
        normalize_quaternion(q);
        const T local_forward[3] = {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)};
        const T local_up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
        rotate_by_quaternion(q[0], q[1], q[2], q[3], local_forward, pose.direction);
        rotate_by_quaternion(q[0], q[1], q[2], q[3], local_up, pose.up);
    }
    else if(options.has_forward) {
        pose.direction[0] = options.forward[0];
        pose.direction[1] = options.forward[1];
        pose.direction[2] = options.forward[2];
    }
    else {
        const T* target = options.has_look_at ? options.look_at : renderer.scene_center;
        pose.direction[0] = target[0] - eye[0];
        pose.direction[1] = target[1] - eye[1];
        pose.direction[2] = target[2] - eye[2];
    }
    normalize(pose.direction);
    if(!options.has_orientation) {
        pose.up[0] = options.up[0];
        pose.up[1] = options.up[1];
        pose.up[2] = options.up[2];
    }
    normalize(pose.up);
    return pose;
}

template <typename DEVICE, typename SPEC>
static void render_output_launch(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer) {
    if constexpr (SPEC::HAS_DEPTH) {
        rlt::render_depth_only_launch(device, renderer);
    }
    else {
        rlt::render_rgb_only_launch(device, renderer);
    }
}

template <typename DEVICE, typename SPEC>
static void render_output_sync(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer) {
    if constexpr (SPEC::HAS_DEPTH) {
        rlt::render_depth_only_sync(device, renderer);
    }
    else {
        rlt::render_rgb_only_sync(device, renderer);
    }
}

template <typename DEVICE, typename SPEC>
static void render_output(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer) {
    render_output_launch(device, renderer);
    render_output_sync(device, renderer);
}

template <typename DEVICE, typename SPEC>
static FrameStats validate_current_frame(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer) {
    constexpr int cam_pixels = SPEC::CAM_PIXELS;
    constexpr int num_cameras = SPEC::NUM_CAMERAS;
    FrameStats stats{true, 0, std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0};
    double sum = 0.0;
    long long count = 0;

    if constexpr (SPEC::HAS_DEPTH) {
        rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
        const float* depth = rlt::data(renderer.depth_buffer);
        const float max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        for(int camera_i = 0; camera_i < num_cameras; camera_i++) {
            const float* camera_depth = depth + static_cast<size_t>(camera_i) * cam_pixels;
            float min_depth = std::numeric_limits<float>::max();
            float max_hit_depth = std::numeric_limits<float>::lowest();
            int hit_count = 0;
            for(int pixel_i = 0; pixel_i < cam_pixels; pixel_i++) {
                const float value = camera_depth[pixel_i];
                if(std::isfinite(value)) {
                    stats.min_value = std::min(stats.min_value, static_cast<double>(value));
                    stats.max_value = std::max(stats.max_value, static_cast<double>(value));
                    sum += static_cast<double>(value);
                    count++;
                    if(value < max_depth * 0.999f) {
                        min_depth = std::min(min_depth, value);
                        max_hit_depth = std::max(max_hit_depth, value);
                        hit_count++;
                    }
                }
            }
            if(hit_count == 0 || max_hit_depth - min_depth < 1e-3f) {
                stats.bad_frames++;
            }
        }
    }
    else {
        rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
        const uint32_t* pixels = rlt::data(renderer.frame_buffer);
        for(int camera_i = 0; camera_i < num_cameras; camera_i++) {
            const uint32_t* camera_pixels = pixels + static_cast<size_t>(camera_i) * cam_pixels;
            const uint32_t first_pixel = camera_pixels[0];
            bool nonblack = false;
            bool varied = false;
            for(int pixel_i = 0; pixel_i < cam_pixels; pixel_i++) {
                const uint32_t rgba = camera_pixels[pixel_i];
                const int r = static_cast<int>((rgba >> 0) & 0xFF);
                const int g = static_cast<int>((rgba >> 8) & 0xFF);
                const int b = static_cast<int>((rgba >> 16) & 0xFF);
                const double luminance = static_cast<double>(r + g + b) / 3.0;
                stats.min_value = std::min(stats.min_value, luminance);
                stats.max_value = std::max(stats.max_value, luminance);
                sum += luminance;
                count++;
                if(luminance > 5.0) {
                    nonblack = true;
                }
                if(rgba != first_pixel) {
                    varied = true;
                }
            }
            if(!nonblack || !varied) {
                stats.bad_frames++;
            }
        }
    }

    if(count > 0) {
        stats.mean_value = sum / static_cast<double>(count);
    }
    else {
        stats.min_value = 0.0;
        stats.max_value = 0.0;
        stats.mean_value = 0.0;
        stats.bad_frames = num_cameras;
    }
    stats.plausible = stats.bad_frames == 0;
    return stats;
}

template <typename DEVICE, typename SPEC>
static BenchmarkResult run_benchmark(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, const std::vector<CameraPose>& camera_poses, StepAxis step, const Options& options) {
    const bool with_physics = step == StepAxis::RENDER_PHYSICS;

    if(options.warmup_seconds > 0) {
        auto warmup_start = std::chrono::high_resolution_clock::now();
        int warmup_iterations = 0;
        for(;;) {
            if(with_physics) {
                write_cameras(device, renderer, scene, camera_poses);
                rlt::set_cameras_async(device, renderer, renderer.cameras);
            }
            render_output(device, renderer);
            warmup_iterations++;
            auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = std::chrono::duration<double>(now - warmup_start).count();
            if(elapsed >= options.warmup_seconds && warmup_iterations > 0) {
                break;
            }
        }
    }
    else {
        for(int i = 0; i < options.warmup_iterations; i++) {
            if(with_physics) {
                write_cameras(device, renderer, scene, camera_poses);
                rlt::set_cameras_async(device, renderer, renderer.cameras);
            }
            render_output(device, renderer);
        }
    }

    cudaDeviceSynchronize();
    auto wall_start = std::chrono::high_resolution_clock::now();
    int iterations = 0;

    if(options.iterations > 0) {
        for(; iterations < options.iterations; iterations++) {
            if(with_physics) {
                write_cameras(device, renderer, scene, camera_poses);
                rlt::set_cameras_async(device, renderer, renderer.cameras);
                render_output(device, renderer);
            }
            else {
                render_output_launch(device, renderer);
                if((iterations + 1) % options.sync_interval == 0) {
                    render_output_sync(device, renderer);
                }
            }
        }
        if(!with_physics && iterations % options.sync_interval != 0) {
            render_output_sync(device, renderer);
        }
    }
    else {
        for(;;) {
            if(with_physics) {
                write_cameras(device, renderer, scene, camera_poses);
                rlt::set_cameras_async(device, renderer, renderer.cameras);
                render_output(device, renderer);
                iterations++;
                auto now = std::chrono::high_resolution_clock::now();
                const double elapsed = std::chrono::duration<double>(now - wall_start).count();
                if(elapsed >= options.seconds) {
                    break;
                }
            }
            else {
                render_output_launch(device, renderer);
                iterations++;
                if(iterations % options.sync_interval == 0) {
                    render_output_sync(device, renderer);
                    auto now = std::chrono::high_resolution_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - wall_start).count();
                    if(elapsed >= options.seconds) {
                        break;
                    }
                }
            }
        }
        if(!with_physics && iterations % options.sync_interval != 0) {
            render_output_sync(device, renderer);
        }
    }

    cudaDeviceSynchronize();
    auto wall_end = std::chrono::high_resolution_clock::now();
    const double elapsed_s = std::chrono::duration<double>(wall_end - wall_start).count();
    const double frames = static_cast<double>(iterations) * static_cast<double>(SPEC::NUM_CAMERAS);
    const double pixels = frames * static_cast<double>(SPEC::CAM_PIXELS);
    const double rays = pixels * static_cast<double>(SPEC::HAS_DEPTH ? SPEC::DEPTH_SAMPLES : SPEC::RGB_SAMPLES);
    return {
        iterations,
        elapsed_s,
        frames / elapsed_s,
        pixels / elapsed_s,
        (rays / 1.0e6) / elapsed_s
    };
}

template <typename DEVICE, typename SPEC>
static bool setup_scene(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, const Options& options) {
    if(scene == SceneAxis::OBJECTS_20) {
        make_20_object_scene(renderer);
        return true;
    }
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Loading ProcTHOR scene: " << options.procthor_path);
    return rlt::load_model(device, renderer, options.procthor_path);
}

template <typename DEVICE, typename SPEC>
static FrameStats save_verification_image(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, const std::string& filename) {
    FrameStats stats = validate_current_frame(device, renderer);
    if constexpr (SPEC::HAS_DEPTH) {
        rlt::save_depth_image(device, renderer, filename.c_str());
    }
    else {
        rlt::save_image(device, renderer, filename.c_str());
    }
    return stats;
}

template <typename DEVICE, typename SPEC>
static bool run_procthor_frame(DEVICE& device, const Options& options, const std::string& cuda_name) {
    static_assert(SPEC::NUM_CAMERAS == 1, "ProcTHOR frame target must use one camera");
    rlt::rendering::raytracing::Renderer<SPEC> renderer;
    rlt::malloc(device, renderer);

    if(!setup_scene(device, renderer, SceneAxis::PROCTHOR, options)) {
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to set up ProcTHOR scene.");
        rlt::free(device, renderer);
        return false;
    }

    rlt::upload_geometry(device, renderer);
    const CameraPose pose = make_single_frame_pose(renderer, options);
    write_single_camera(device, renderer, options.position, pose);
    rlt::set_cameras(device, renderer, renderer.cameras);
    rlt::build_pipeline(device, renderer);
    render_output(device, renderer);
    cudaDeviceSynchronize();

    ensure_parent_dir(options.output_png);
    FrameStats frame_stats = save_verification_image(device, renderer, options.output_png);
    std::cout
        << "{\"library\":\"hyperdrone\""
        << ",\"scene\":\"procthor\""
        << ",\"output\":\"" << output_name_for_spec<SPEC>() << "\""
        << ",\"cuda_device\":\"" << cuda_name << "\""
        << ",\"width\":" << SPEC::CAM_WIDTH
        << ",\"height\":" << SPEC::CAM_HEIGHT
        << ",\"camera_position_flu\":["
        << options.position[0] << "," << options.position[1] << "," << options.position[2] << "]"
        << ",\"camera_forward_flu\":["
        << pose.direction[0] << "," << pose.direction[1] << "," << pose.direction[2] << "]"
        << ",\"camera_up_flu\":["
        << pose.up[0] << "," << pose.up[1] << "," << pose.up[2] << "]"
        << ",\"camera_orientation_mode\":\""
        << (options.has_orientation ? "orientation_flu_wxyz" : (options.has_forward ? "forward_up" : (options.has_look_at ? "look_at_up" : "scene_center_up"))) << "\""
        << ",\"camera_orientation_flu_wxyz\":"
        << (options.has_orientation ? "[" : "null");
    if(options.has_orientation) {
        std::cout
            << options.orientation_wxyz[0] << "," << options.orientation_wxyz[1] << ","
            << options.orientation_wxyz[2] << "," << options.orientation_wxyz[3] << "]";
    }
    std::cout
        << ",\"procthor_path\":\"" << options.procthor_path << "\""
        << ",\"output_png\":\"" << options.output_png << "\""
        << ",\"plausible\":" << (frame_stats.plausible ? "true" : "false")
        << ",\"bad_frames\":" << frame_stats.bad_frames
        << ",\"frame_min\":" << frame_stats.min_value
        << ",\"frame_max\":" << frame_stats.max_value
        << ",\"frame_mean\":" << frame_stats.mean_value
        << "}\n";

    rlt::free(device, renderer);
    return frame_stats.plausible;
}

template <typename DEVICE, typename SPEC>
static bool run_combination(DEVICE& device, SceneAxis scene, StepAxis step, const Options& options, const std::string& cuda_name, const std::string& gpu_label) {
    rlt::rendering::raytracing::Renderer<SPEC> renderer;
    rlt::malloc(device, renderer);

    if(!setup_scene(device, renderer, scene, options)) {
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to set up scene: " << scene_name(scene));
        rlt::free(device, renderer);
        return false;
    }

    rlt::upload_geometry(device, renderer);
    const OrientationMode orientation = orientation_mode(options);
    std::vector<CameraPose> camera_poses = make_camera_poses(renderer, scene, options);
    write_cameras(device, renderer, scene, camera_poses);
    rlt::set_cameras(device, renderer, renderer.cameras);
    rlt::build_pipeline(device, renderer);

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const CameraOffset offset = camera_offset(scene);

    RL_TOOLS_RENDERING_RAYTRACING_LOG("Benchmark combination: scene=" << scene_name(scene)
        << ", output=" << output_name_for_spec<SPEC>()
        << ", step_mode=" << step_name(step)
        << ", shading_profile=" << shading_profile_name()
        << ", envs=" << SPEC::NUM_CAMERAS
        << ", resolution=" << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT
        << ", fov_deg=" << static_cast<double>(SPEC::COS_FOVY) * RAD_TO_DEG
        << ", camera_offset_flu=[" << offset.x << "," << offset.y << "," << offset.z << "]"
        << ", camera_orientation_sampling=" << orientation_mode_name(orientation)
        << ", seed=" << options.seed);

    BenchmarkResult result = run_benchmark(device, renderer, scene, camera_poses, step, options);

    const std::string verification_name = std::string("verify_hyperdrone_")
        + scene_name(scene) + "_"
        + output_name_for_spec<SPEC>() + "_"
        + step_name(step) + "_"
        + sanitize_label(gpu_label) + "_"
        + std::to_string(SPEC::NUM_CAMERAS) + "views_"
        + std::to_string(SPEC::CAM_WIDTH) + "x" + std::to_string(SPEC::CAM_HEIGHT) + ".png";
    const std::string verification_path = join_path(options.output_dir, verification_name);
    FrameStats frame_stats = save_verification_image(device, renderer, verification_path);
    if(!frame_stats.plausible) {
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Frame plausibility check failed: " << frame_stats.bad_frames
            << "/" << SPEC::NUM_CAMERAS << " camera frames look black or degenerate.");
    }

    std::cout << "csv_header,scene,objects20_layout,output,step_mode,gpu_label,cuda_device,num_envs,width,height,camera_offset_x,camera_offset_y,camera_offset_z,camera_orientation_sampling,iterations,elapsed_s,frames_per_s,pixels_per_s,mrays_per_s,cuda_free_mb_after_setup,cuda_total_mb,verification_png,plausible,bad_frames,frame_min,frame_max,frame_mean\n";
    std::cout << "csv_result,"
        << csv_quote(scene_name(scene)) << ","
        << csv_quote(scene == SceneAxis::OBJECTS_20 ? OBJECTS20_LAYOUT_NAME : "") << ","
        << csv_quote(output_name_for_spec<SPEC>()) << ","
        << csv_quote(step_name(step)) << ","
        << csv_quote(gpu_label) << ","
        << csv_quote(cuda_name) << ","
        << SPEC::NUM_CAMERAS << ","
        << SPEC::CAM_WIDTH << ","
        << SPEC::CAM_HEIGHT << ","
        << offset.x << ","
        << offset.y << ","
        << offset.z << ","
        << csv_quote(orientation_mode_name(orientation)) << ","
        << result.iterations << ","
        << result.elapsed_s << ","
        << result.frames_per_s << ","
        << result.pixels_per_s << ","
        << result.mrays_per_s << ","
        << static_cast<double>(free_mem) / (1024.0 * 1024.0) << ","
        << static_cast<double>(total_mem) / (1024.0 * 1024.0) << ","
        << csv_quote(verification_path) << ","
        << (frame_stats.plausible ? 1 : 0) << ","
        << frame_stats.bad_frames << ","
        << frame_stats.min_value << ","
        << frame_stats.max_value << ","
        << frame_stats.mean_value << "\n";

    rlt::free(device, renderer);
    return frame_stats.plausible;
}

template <rlt::rendering::raytracing::OutputMode T_OUTPUT_MODE>
static bool run_output_combinations(DEVICE& device, const std::vector<SceneAxis>& scenes, const std::vector<StepAxis>& steps, const Options& options, const std::string& cuda_name, const std::string& gpu_label) {
    using SPEC = BenchmarkSpec<T_OUTPUT_MODE>;
    bool ok = true;
    for(SceneAxis scene : scenes) {
        for(StepAxis step : steps) {
            ok = run_combination<DEVICE, SPEC>(device, scene, step, options, cuda_name, gpu_label) && ok;
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    Options options;
    if(!parse_options(argc, argv, options)) {
        print_help(argv[0]);
        return 1;
    }
    if(!ensure_output_dir(options.output_dir)) {
        return 1;
    }

    std::vector<SceneAxis> scenes;
    std::vector<StepAxis> steps;
    std::vector<OutputAxis> outputs;
    if(!select_scenes(options.scene, scenes) || !select_steps(options.step_mode, steps) || !select_outputs(options.output, outputs)) {
        return 1;
    }

    DEVICE device;
    rlt::init(device);

    const std::string cuda_name = cuda_device_name();
    const std::string gpu_label = options.gpu_label.empty() ? cuda_name : options.gpu_label;

    RL_TOOLS_RENDERING_RAYTRACING_LOG("simulator matrix benchmark output=" << output_name()
        << ", shading_profile=" << shading_profile_name()
        << ", compiled_envs=" << NUM_ENVS
        << ", cuda_device=" << cuda_name
        << ", gpu_label=" << gpu_label);

    bool ok = true;
    (void)outputs;
#if RL_TOOLS_RENDERING_RAYTRACING_SIM_FRAME_TOOL
    if(options.scene != "procthor" && options.scene != "all") {
        std::cerr << "The single-frame target only supports --scene procthor." << std::endl;
        return 1;
    }
    if constexpr (RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB) {
        if(options.output != "rgb" && options.output != "all") {
            std::cerr << "The single-frame target is compiled for RGB output." << std::endl;
            return 1;
        }
        ok = run_procthor_frame<DEVICE, BenchmarkSpec<rlt::rendering::raytracing::OutputMode::RGB>>(device, options, cuda_name);
    }
    else if constexpr (RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH) {
        if(options.output != "depth" && options.output != "all") {
            std::cerr << "The single-frame target is compiled for depth output." << std::endl;
            return 1;
        }
        ok = run_procthor_frame<DEVICE, BenchmarkSpec<rlt::rendering::raytracing::OutputMode::DEPTH>>(device, options, cuda_name);
    }
    else {
        std::cerr << "The single-frame target must be compiled for RGB or depth output." << std::endl;
        return 1;
    }
#else
#if RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
    if(options.output == "depth") {
        std::cerr << "This target was compiled for RGB only; use rendering_raytracing_sim_benchmark or rendering_raytracing_sim_benchmark_depth for depth." << std::endl;
        return 1;
    }
    ok = run_output_combinations<rlt::rendering::raytracing::OutputMode::RGB>(device, scenes, steps, options, cuda_name, gpu_label) && ok;
#elif RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    if(options.output == "rgb") {
        std::cerr << "This target was compiled for depth only; use rendering_raytracing_sim_benchmark or rendering_raytracing_sim_benchmark_rgb for RGB." << std::endl;
        return 1;
    }
    ok = run_output_combinations<rlt::rendering::raytracing::OutputMode::DEPTH>(device, scenes, steps, options, cuda_name, gpu_label) && ok;
#else
    for(OutputAxis output : outputs) {
        if(output == OutputAxis::RGB) {
            ok = run_output_combinations<rlt::rendering::raytracing::OutputMode::RGB>(device, scenes, steps, options, cuda_name, gpu_label) && ok;
        }
        else {
            ok = run_output_combinations<rlt::rendering::raytracing::OutputMode::DEPTH>(device, scenes, steps, options, cuda_name, gpu_label) && ok;
        }
    }
#endif
#endif
    return ok ? 0 : 1;
}
