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
using BenchmarkSpec = rlt::rendering::raytracing::Specification<T, TI, 64, 64, NUM_ENVS, 1, ShadingProfile, false, 1, false, 1, T_OUTPUT_MODE>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

enum class SceneAxis { OBJECTS_20, PROCTHOR };
enum class StepAxis { RENDER_ONLY, RENDER_PHYSICS };
enum class OutputAxis { RGB, DEPTH };

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
    std::string gpu_label;
    std::string output_dir = ".";
    std::string procthor_path = "tests/data/ProcTHOR-Train-1.glb";
    double seconds = 10.0;
    double warmup_seconds = 2.0;
    int iterations = 0;
    int warmup_iterations = 10;
    int sync_interval = 10;
    int num_envs = NUM_ENVS;
    int resolution = 64;
    uint32_t seed = 0;
};

struct CameraMotion {
    T yaw_offset;
    T pitch_offset;
    T yaw_velocity;
    T pitch_velocity;
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
        << "  --gpu-label <label>\n"
        << "  --seconds <seconds>              Timed duration per combination (default: 10)\n"
        << "  --iterations <count>             Fixed timed iterations; overrides --seconds when >0\n"
        << "  --warmup-seconds <seconds>       Untimed warmup duration before timing (default: 2)\n"
        << "  --warmup-iterations <count>      Legacy warmup count used when --warmup-seconds=0 (default: 10)\n"
        << "  --sync-interval <count>          Render-only async sync interval (default: 10)\n"
        << "  --seed <count>                   Deterministic camera-orientation seed (default: 0)\n"
        << "  --num-envs <count>               Must match compile-time NUM_ENVS=" << NUM_ENVS << "\n"
        << "  --resolution <pixels>            Must be 64 for this benchmark target\n"
        << "  --output-dir <dir>\n"
        << "  --procthor-path <file>\n";
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
        else if(get_option_value(i, argc, argv, arg, "--gpu-label", value)) {
            options.gpu_label = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output-dir", value)) {
            options.output_dir = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--procthor-path", value)) {
            options.procthor_path = value;
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
    if(options.resolution != 64) {
        std::cerr << "This benchmark is standardized on 64x64. Requested " << options.resolution << "." << std::endl;
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
    return true;
}

static const char* scene_name(SceneAxis scene) {
    return scene == SceneAxis::OBJECTS_20 ? "20_objects" : "procthor";
}

static const char* step_name(StepAxis step) {
    return step == StepAxis::RENDER_ONLY ? "render_only" : "render_physics";
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

template <typename SPEC>
static std::vector<CameraMotion> make_camera_states(const rlt::rendering::raytracing::Renderer<SPEC>& renderer, const Options& options) {
    (void)renderer;
    std::vector<CameraMotion> states(SPEC::NUM_CAMERAS);
    std::mt19937 rng(options.seed);
    std::uniform_real_distribution<T> yaw_dist(static_cast<T>(-0.08), static_cast<T>(0.08));
    std::uniform_real_distribution<T> pitch_dist(static_cast<T>(-0.06), static_cast<T>(0.06));
    std::uniform_real_distribution<T> velocity_dist(static_cast<T>(0.75), static_cast<T>(1.25));
    std::bernoulli_distribution sign_dist(0.5);
    for(int i = 0; i < static_cast<int>(SPEC::NUM_CAMERAS); i++) {
        const T yaw_sign = sign_dist(rng) ? static_cast<T>(1) : static_cast<T>(-1);
        const T pitch_sign = sign_dist(rng) ? static_cast<T>(1) : static_cast<T>(-1);
        states[i].yaw_offset = yaw_dist(rng);
        states[i].pitch_offset = pitch_dist(rng);
        states[i].yaw_velocity = yaw_sign * static_cast<T>(0.015) * velocity_dist(rng);
        states[i].pitch_velocity = pitch_sign * static_cast<T>(0.010) * velocity_dist(rng);
    }
    return states;
}

template <typename DEVICE, typename SPEC>
static void write_cameras(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, std::vector<CameraMotion>& states, bool advance, T dt) {
    const T target[3] = {renderer.scene_center[0], renderer.scene_center[1], renderer.scene_center[2]};
    const CameraOffset offset = camera_offset(scene);
    const T eye[3] = {offset.x, offset.y, offset.z};
    const T up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
    const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    T forward[3] = {
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2]
    };
    T forward_norm = std::sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    if(forward_norm < static_cast<T>(1e-6)) {
        forward[0] = static_cast<T>(1);
        forward[1] = static_cast<T>(0);
        forward[2] = static_cast<T>(0);
        forward_norm = static_cast<T>(1);
    }
    forward[0] /= forward_norm;
    forward[1] /= forward_norm;
    forward[2] /= forward_norm;
    T right[3] = {
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0]
    };
    T right_norm = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    if(right_norm < static_cast<T>(1e-6)) {
        right[0] = static_cast<T>(0);
        right[1] = static_cast<T>(1);
        right[2] = static_cast<T>(0);
        right_norm = static_cast<T>(1);
    }
    right[0] /= right_norm;
    right[1] /= right_norm;
    right[2] /= right_norm;
    for(int i = 0; i < static_cast<int>(SPEC::NUM_CAMERAS); i++) {
        CameraMotion& state = states[i];
        if(advance) {
            state.yaw_offset += state.yaw_velocity * dt;
            state.pitch_offset += state.pitch_velocity * dt;
            if(state.yaw_offset > static_cast<T>(0.08) || state.yaw_offset < static_cast<T>(-0.08)) {
                state.yaw_velocity = -state.yaw_velocity;
            }
            if(state.pitch_offset > static_cast<T>(0.06) || state.pitch_offset < static_cast<T>(-0.06)) {
                state.pitch_velocity = -state.pitch_velocity;
            }
        }
        T direction[3] = {
            forward[0] + state.yaw_offset * right[0] + state.pitch_offset * up[0],
            forward[1] + state.yaw_offset * right[1] + state.pitch_offset * up[1],
            forward[2] + state.yaw_offset * right[2] + state.pitch_offset * up[2]
        };
        const T direction_norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        direction[0] /= direction_norm;
        direction[1] /= direction_norm;
        direction[2] /= direction_norm;
        const T look_at[3] = {
            eye[0] + direction[0],
            eye[1] + direction[1],
            eye[2] + direction[2]
        };
        rlt::set(device, renderer.cameras, rlt::make_camera_data(eye, look_at, up, SPEC::COS_FOVY, aspect), static_cast<TI>(i));
    }
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
static BenchmarkResult run_benchmark(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, SceneAxis scene, std::vector<CameraMotion>& camera_states, StepAxis step, const Options& options) {
    const bool with_physics = step == StepAxis::RENDER_PHYSICS;
    const T dt = static_cast<T>(1.0 / 60.0);

    if(options.warmup_seconds > 0) {
        auto warmup_start = std::chrono::high_resolution_clock::now();
        int warmup_iterations = 0;
        for(;;) {
            if(with_physics) {
                write_cameras(device, renderer, scene, camera_states, true, dt);
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
                write_cameras(device, renderer, scene, camera_states, true, dt);
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
                write_cameras(device, renderer, scene, camera_states, true, dt);
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
                write_cameras(device, renderer, scene, camera_states, true, dt);
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
static bool run_combination(DEVICE& device, SceneAxis scene, StepAxis step, const Options& options, const std::string& cuda_name, const std::string& gpu_label) {
    rlt::rendering::raytracing::Renderer<SPEC> renderer;
    rlt::malloc(device, renderer);

    if(!setup_scene(device, renderer, scene, options)) {
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to set up scene: " << scene_name(scene));
        rlt::free(device, renderer);
        return false;
    }

    rlt::upload_geometry(device, renderer);
    std::vector<CameraMotion> camera_states = make_camera_states(renderer, options);
    write_cameras(device, renderer, scene, camera_states, false, static_cast<T>(0));
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
        << ", camera_orientation_sampling=random_look_at_jitter"
        << ", seed=" << options.seed);

    BenchmarkResult result = run_benchmark(device, renderer, scene, camera_states, step, options);

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

    std::cout << "csv_header,scene,objects20_layout,output,step_mode,gpu_label,cuda_device,num_envs,width,height,camera_offset_x,camera_offset_y,camera_offset_z,iterations,elapsed_s,frames_per_s,pixels_per_s,mrays_per_s,cuda_free_mb_after_setup,cuda_total_mb,verification_png,plausible,bad_frames,frame_min,frame_max,frame_mean\n";
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
    return ok ? 0 : 1;
}
