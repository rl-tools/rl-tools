
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/raytracing/save_cpu.h>
#include <metra/metra.h>
#include <rl_tools/utils/extrack/extrack.h>
#include <nlohmann/json.hpp>

#include "simulator_matrix_config.h"
#include "simulator_matrix.h"
#include "simulator_matrix_physics.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;
using namespace rl_tools;
using CONFIG = rlt::rendering::raytracing::benchmark::Config;
using BenchmarkMode = rlt::rendering::raytracing::benchmark::Mode;
using T = float;
using TI = int;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RUNTIME = rlt::rendering::raytracing::Runtime<>;
using ShadingProfile = CONFIG::SHADING;
static constexpr T FOV = 80;
static constexpr const char* BACKEND_NAME = rlt::rendering::raytracing::backends::Name<rlt::rendering::raytracing::backends::Default>::VALUE;
static constexpr TI NUM_ENVS = CONFIG::NUM_CAMERAS;
static constexpr TI CAM_WIDTH = CONFIG::RESOLUTION;
static constexpr TI CAM_HEIGHT = CONFIG::RESOLUTION;
static constexpr const char* OBJECTS20_LAYOUT_NAME = "canonical_staggered_v1";
static constexpr const char* PAPER_SCENE_SHA1 = "7f1c9129532798e0b63bc41edb6b4c09251cf8a0";
namespace paper = rlt::rendering::raytracing::benchmark;
static constexpr const char* PHYSICS_BACKEND_NAME = CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cuda ? "cuda"
    : (CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cpu ? "cpu" : "none");

template <typename SPEC>
using CameraBuffer = rlt::Tensor<typename rlt::rendering::raytracing::Renderer<SPEC>::CAMERA_TENSOR_SPEC>;

template <bool T_OUTPUT_RGB, bool T_OUTPUT_DEPTH, TI T_WIDTH = CAM_WIDTH, TI T_HEIGHT = CAM_HEIGHT, TI T_NUM_CAMERAS = NUM_ENVS, typename T_SHADING = ShadingProfile, bool T_ENABLE_AA = false, TI T_AA_GRID_SIZE = 1>
struct BenchmarkConfig: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = T_WIDTH, CAM_HEIGHT = T_HEIGHT, NUM_CAMERAS = T_NUM_CAMERAS, NUM_PROBES = 0;
    using SHADING = T_SHADING;
    using OUTPUTS = typename rro::Concat<
        rro::If<(T_OUTPUT_RGB), rro::RGB<>>,
        rro::If<(T_OUTPUT_DEPTH), rro::Depth<>>
    >::type;
    static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_AA;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = T_AA_GRID_SIZE;
};
template <bool T_OUTPUT_RGB, bool T_OUTPUT_DEPTH, typename T_SHADING = ShadingProfile>
using BenchmarkSpec = rlt::rendering::raytracing::Specification<BenchmarkConfig<T_OUTPUT_RGB, T_OUTPUT_DEPTH, CAM_WIDTH, CAM_HEIGHT, NUM_ENVS, T_SHADING>>;
template <bool T_OUTPUT_RGB, bool T_OUTPUT_DEPTH, TI T_WIDTH, TI T_HEIGHT, TI T_NUM_CAMERAS, typename T_SHADING, bool T_ENABLE_AA, TI T_AA_GRID_SIZE>
using SweepBenchmarkSpec = rlt::rendering::raytracing::Specification<BenchmarkConfig<T_OUTPUT_RGB, T_OUTPUT_DEPTH, T_WIDTH, T_HEIGHT, T_NUM_CAMERAS, T_SHADING, T_ENABLE_AA, T_AA_GRID_SIZE>>;

static constexpr long long SWEEP_REFERENCE_PIXELS = static_cast<long long>(NUM_ENVS) * static_cast<long long>(CAM_WIDTH) * static_cast<long long>(CAM_HEIGHT);

template <TI T_RESOLUTION>
struct SweepCameraCount {
    static constexpr long long RESOLUTION_PIXELS = static_cast<long long>(T_RESOLUTION) * static_cast<long long>(T_RESOLUTION);
    static constexpr long long RAW = SWEEP_REFERENCE_PIXELS / RESOLUTION_PIXELS;
    static constexpr TI VALUE = static_cast<TI>(RAW > 1 ? RAW : 1);
};

enum class SceneAxis { OBJECTS_20, PROCTHOR };
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
    std::string scene = CONFIG::PAPER ? "all" : "procthor";
    std::string output = CONFIG::PAPER ? "all" : "rgb";
    std::string workload = CONFIG::PAPER ? "all" : "render_only";
    std::string fidelity = CONFIG::PAPER ? "all" : "medium";
    std::string orientation_mode = "uniform_so3";
    std::string output_dir = ".";
    std::string output_png;
    std::string output_jsonl;
    std::string procthor_path = CONFIG::PAPER ? CONFIG::PAPER_SCENE : "conta:75d1d67a17ecd16bc3a214baa895240da8060df7";
    double seconds = CONFIG::PAPER ? 20.0 : 10.0;
    double warmup_seconds = 2.0;
    double cooldown_seconds = 10.0;
    int iterations = 0;
    int repeats = CONFIG::PAPER ? 3 : 1;
    int repeat = 1;
    int sync_interval = 10;
    int sweep_config_start = 0;
    int sweep_config_count = -1;
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
    double primary_mrays_per_s;
};


static bool has_prefix(const std::string& value, const char* prefix) {
    return value.compare(0, std::strlen(prefix), prefix) == 0;
}

static std::string value_after_prefix(const std::string& value, const char* prefix) {
    return value.substr(std::strlen(prefix));
}

static void print_help(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
        << "  --output <rgb|depth|all>         Default: " << (CONFIG::PAPER ? "all" : "rgb") << "\n"
        << "  --output-dir <dir>              Verification images (default: .)\n"
        << "  --procthor-path <file|conta:hash> Default: pinned ProcTHOR-Train-1\n";
    if constexpr (CONFIG::MODE == BenchmarkMode::frame) {
        std::cout
            << "  --output-png <file>              Requires one output; otherwise named automatically\n"
            << "  --position-flu X Y Z\n"
            << "  --forward-flu X Y Z\n"
            << "  --look-at-flu X Y Z\n"
            << "  --up-flu X Y Z\n"
            << "  --orientation-flu-wxyz W X Y Z\n";
    }
    else {
        std::cout
            << "  --scene <procthor|20_objects|all> Default: " << (CONFIG::PAPER ? "all" : "procthor") << "\n"
            << "  --orientation-mode <uniform_so3|random_yaw_pitch|look_at_scene_jitter>\n"
            << "  --seed <count>                   Default: 0\n"
            << "  --seconds <seconds>              Timed duration per case (default: " << (CONFIG::PAPER ? 20 : 10) << ")\n"
            << "  --iterations <count>             Fixed count overriding duration (default: 0)\n"
            << "  --warmup-seconds <seconds>       Default: 2; 0 runs " << (CONFIG::PAPER ? 10 : 1) << " warmup iterations\n"
            << "  --sync-interval <count>          Default: 10\n"
            << "  Compiled cameras: " << NUM_ENVS << ", resolution: " << CAM_WIDTH << "x" << CAM_HEIGHT << "\n";
    }
    if constexpr (CONFIG::MODE == BenchmarkMode::sweep) {
        std::cout
            << "  --cooldown-seconds <seconds>     Between configurations (default: 10)\n"
            << "  --sweep-config-start <index>     First configuration (default: 0)\n"
            << "  --sweep-config-count <count>     <=0 selects all remaining configurations\n";
    }
    if constexpr (CONFIG::PAPER) {
        std::cout << "  --fidelity <low|medium|high|all>  Default: all\n"
            << "  --workload <render_only|render_physics_static|render_physics_coupled|all>\n"
            << "    Default: all = render_only + render_physics_static (24 cells across all axes)\n"
            << "    Physics backend: " << PHYSICS_BACKEND_NAME << " (selected at build time).\n"
            << "  --repeats <count>                Complete matrix passes (default: 3, 72 measurements)\n"
            << "  Results are also saved to a timestamped .jsonl file in --output-dir.\n"
            << "  ProcTHOR must match the May asset; download_paper_scene.py fetches it.\n";
    }
}

static bool parse_int(const std::string& value, int& out) {
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if(errno == ERANGE || end == value.c_str() || *end != '\0' || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static bool parse_double(const std::string& value, double& out) {
    // Reject non-finite spellings before conversion: fast-math can eliminate floating-point checks.
    if(value.empty() || value.find_first_not_of("0123456789+-.eE") != std::string::npos) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value.c_str(), &end);
    if(errno == ERANGE || end == value.c_str() || *end != '\0') {
        return false;
    }
    out = parsed;
    return true;
}

template <std::size_t N>
static bool get_vector_option(int& i, int argc, char** argv, const std::string& arg,
                              const char* name, T (&out)[N], bool& seen) {
    const std::string prefix = std::string(name) + "=";
    const bool csv = has_prefix(arg, prefix.c_str());
    if(!csv && arg != name) {
        return false;
    }
    if(!csv && i + static_cast<int>(N) >= argc) {
        std::cerr << "Missing values for " << name << std::endl;
        return false;
    }
    std::istringstream values(csv ? arg.substr(prefix.size()) : "");
    for(std::size_t component = 0; component < N; component++) {
        std::string value;
        if(csv) {
            if(!std::getline(values, value, ',')) {
                return false;
            }
        }
        else {
            value = argv[++i];
        }
        double parsed;
        if(!parse_double(value, parsed) || std::abs(parsed) > std::numeric_limits<T>::max()) {
            std::cerr << "Invalid " << name << " component: " << value << std::endl;
            return false;
        }
        out[component] = static_cast<T>(parsed);
    }
    if(csv && !values.eof()) {
        return false;
    }
    seen = true;
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
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--scene", value)) {
            options.scene = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output", value)) {
            options.output = value;
        }
        else if(CONFIG::PAPER && get_option_value(i, argc, argv, arg, "--workload", value)) {
            options.workload = value;
        }
        else if(CONFIG::PAPER && get_option_value(i, argc, argv, arg, "--fidelity", value)) {
            options.fidelity = value;
        }
        else if(CONFIG::PAPER && get_option_value(i, argc, argv, arg, "--repeats", value)) {
            if(!parse_int(value, options.repeats) || options.repeats <= 0) {
                std::cerr << "Invalid --repeats (must be > 0): " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--orientation-mode", value)) {
            options.orientation_mode = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--output-dir", value)) {
            options.output_dir = value;
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--output-png", value)) {
            options.output_png = value;
        }
        else if(get_option_value(i, argc, argv, arg, "--procthor-path", value)) {
            options.procthor_path = value;
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_vector_option(i, argc, argv, arg, "--position-flu", options.position, options.has_position)) {
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_vector_option(i, argc, argv, arg, "--forward-flu", options.forward, options.has_forward)) {
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_vector_option(i, argc, argv, arg, "--look-at-flu", options.look_at, options.has_look_at)) {
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_vector_option(i, argc, argv, arg, "--up-flu", options.up, options.has_up)) {
        }
        else if(CONFIG::MODE == BenchmarkMode::frame && get_vector_option(i, argc, argv, arg, "--orientation-flu-wxyz", options.orientation_wxyz, options.has_orientation)) {
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--seconds", value)) {
            if(!parse_double(value, options.seconds)) {
                std::cerr << "Invalid --seconds: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--iterations", value)) {
            if(!parse_int(value, options.iterations)) {
                std::cerr << "Invalid --iterations: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--warmup-seconds", value)) {
            if(!parse_double(value, options.warmup_seconds)) {
                std::cerr << "Invalid --warmup-seconds: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE == BenchmarkMode::sweep && get_option_value(i, argc, argv, arg, "--cooldown-seconds", value)) {
            if(!parse_double(value, options.cooldown_seconds)) {
                std::cerr << "Invalid --cooldown-seconds: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE == BenchmarkMode::sweep && get_option_value(i, argc, argv, arg, "--sweep-config-start", value)) {
            if(!parse_int(value, options.sweep_config_start)) {
                std::cerr << "Invalid --sweep-config-start: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE == BenchmarkMode::sweep && get_option_value(i, argc, argv, arg, "--sweep-config-count", value)) {
            if(!parse_int(value, options.sweep_config_count)) {
                std::cerr << "Invalid --sweep-config-count: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--sync-interval", value)) {
            if(!parse_int(value, options.sync_interval)) {
                std::cerr << "Invalid --sync-interval: " << value << std::endl;
                return false;
            }
        }
        else if(CONFIG::MODE != BenchmarkMode::frame && get_option_value(i, argc, argv, arg, "--seed", value)) {
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

    if(options.fidelity != "low" && options.fidelity != "medium" && options.fidelity != "high" && options.fidelity != "all") {
        std::cerr << "Unsupported --fidelity: " << options.fidelity << std::endl;
        return false;
    }
    if(options.workload != "render_only" && options.workload != "render_physics_static"
       && options.workload != "render_physics_coupled" && options.workload != "all") {
        std::cerr << "Unsupported --workload: " << options.workload << std::endl;
        return false;
    }
    if(!CONFIG::PHYSICS_AVAILABLE && options.workload != "render_only") {
        std::cerr << "This target has no physics backend; use --workload render_only." << std::endl;
        return false;
    }
    if(CONFIG::PAPER && options.orientation_mode != "uniform_so3") {
        std::cerr << "The paper contract requires --orientation-mode uniform_so3." << std::endl;
        return false;
    }
    if(!options.output_png.empty() && options.output == "all") {
        std::cerr << "--output-png requires --output rgb or depth." << std::endl;
        return false;
    }
    if(options.has_forward && options.has_look_at) {
        std::cerr << "Use only one of --forward-flu or --look-at-flu." << std::endl;
        return false;
    }
    if(options.has_orientation && (options.has_forward || options.has_look_at || options.has_up)) {
        std::cerr << "Use --orientation-flu-wxyz without forward, look-at, or up vectors." << std::endl;
        return false;
    }
    if(options.iterations < 0 || (options.seconds <= 0 && options.iterations == 0)) {
        std::cerr << "Either --seconds must be > 0 or --iterations must be > 0." << std::endl;
        return false;
    }
    if(options.warmup_seconds < 0) {
        std::cerr << "--warmup-seconds must be >= 0." << std::endl;
        return false;
    }
    if(options.cooldown_seconds < 0) {
        std::cerr << "--cooldown-seconds must be >= 0." << std::endl;
        return false;
    }
    if(options.sweep_config_start < 0) {
        std::cerr << "--sweep-config-start must be >= 0." << std::endl;
        return false;
    }
    if(options.sync_interval <= 0) {
        std::cerr << "--sync-interval must be > 0." << std::endl;
        return false;
    }
    if(options.orientation_mode != "random_yaw_pitch" &&
       options.orientation_mode != "look_at_scene_jitter" &&
       options.orientation_mode != "uniform_so3") {
        std::cerr << "Unsupported --orientation-mode: " << options.orientation_mode << std::endl;
        return false;
    }
    return true;
}

static const char* scene_name(SceneAxis scene) {
    return scene == SceneAxis::OBJECTS_20 ? "20_objects" : "procthor";
}

template <typename SPEC>
static constexpr const char* shading_profile_name_for_spec() {
    return std::is_base_of<rlt::rendering::raytracing::Low, typename SPEC::SHADING>::value ? "low"
        : (std::is_same<typename SPEC::SHADING, rlt::rendering::raytracing::VeryHigh>::value ? "very_high"
        : (std::is_base_of<rlt::rendering::raytracing::High, typename SPEC::SHADING>::value ? "high" : "medium"));
}

template <typename SPEC>
static std::string anti_aliasing_name_for_spec() {
    return SPEC::ENABLE_ANTI_ALIASING ? std::string("aa") + std::to_string(SPEC::ANTI_ALIASING_GRID_SIZE) : std::string("none");
}

template <typename SPEC>
static constexpr int samples_per_pixel_for_spec() {
    return SPEC::HAS_DEPTH ? 1 : SPEC::RGB_SAMPLES;
}

template <typename SPEC>
static constexpr const char* output_name_for_spec() {
    return SPEC::HAS_DEPTH ? "depth" : "rgb";
}

static bool select_scenes(const std::string& requested, std::vector<SceneAxis>& scenes) {
    if(requested == "all") {
        scenes = {SceneAxis::OBJECTS_20, SceneAxis::PROCTHOR};
    }
    else if(requested == "20_objects") {
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
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if(error) {
        std::cerr << "Cannot create output directory '" << dir << "': " << error.message() << std::endl;
        return false;
    }
    return true;
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

static void add_box(rlt::rendering::raytracing::Object& objects20, T cx, T cy, T cz, T sx, T sy, T sz, T r, T g, T b) {
    rlt::rendering::raytracing::Mesh md;
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
    objects20.meshes.push_back(std::move(md));
}

static void add_sphere(rlt::rendering::raytracing::Object& objects20, T cx, T cy, T cz, T radius, T r, T g, T b) {
    static constexpr int SEGMENTS = 16;
    static constexpr int RINGS = 8;
    static constexpr T PI = static_cast<T>(3.14159265358979323846);
    rlt::rendering::raytracing::Mesh md;
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

    objects20.meshes.push_back(std::move(md));
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

static void make_20_object_scene(rlt::rendering::raytracing::Scene& render_scene) {
    render_scene = {};
    rlt::rendering::raytracing::Object objects20;
    for(int i = 0; i < 20; i++) {
        T sx, sy, sz, r, g, b;
        objects20_scale(i, sx, sy, sz);
        objects20_color(i, r, g, b);
        const auto& position = OBJECTS20_POSITIONS[i];
        if((i % 2) == 0) {
            add_box(objects20, position.x, position.y, position.z, sx, sy, sz, r, g, b);
        }
        else {
            add_sphere(objects20, position.x, position.y, position.z, sx * static_cast<T>(0.5), r, g, b);
        }
    }
    render_scene.objects.push_back(std::move(objects20));
    render_scene.instances.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}, true});
}

// the 20-object benchmark scene uses hand-picked bounds (closer orbit than the AABB-derived
// camera radius); they are baked into the bundle metadata before init so the renderer
// consumes them uniformly (probe budget, depth max), keeping outputs comparable across
// revisions (max_ray_length = 12 matches the previous camera_radius = 6 depth max)
static void apply_20_object_bounds(rlt::rendering::SceneMetadata<T>& metadata) {
    metadata.center[0] = static_cast<T>(0.25);
    metadata.center[1] = 0;
    metadata.center[2] = static_cast<T>(0.60);
    metadata.half_extent[0] = static_cast<T>(3.65);
    metadata.half_extent[1] = static_cast<T>(2.75);
    metadata.half_extent[2] = static_cast<T>(0.95);
    metadata.max_ray_length = static_cast<T>(12.0);
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
static std::vector<CameraPose> make_camera_poses(const rlt::rendering::SceneMetadata<T>& metadata, SceneAxis scene, const Options& options) {
    static constexpr T PI = static_cast<T>(3.14159265358979323846);
    const auto mode = options.orientation_mode == "look_at_scene_jitter" ? OrientationMode::LOOK_AT_SCENE_JITTER
        : (options.orientation_mode == "uniform_so3" ? OrientationMode::UNIFORM_SO3 : OrientationMode::RANDOM_YAW_PITCH);
    std::vector<CameraPose> poses(SPEC::NUM_CAMERAS);
    std::mt19937 rng(options.seed);
    std::uniform_real_distribution<T> unit_dist(static_cast<T>(0), static_cast<T>(1));
    std::uniform_real_distribution<T> look_at_yaw_dist(static_cast<T>(-0.08), static_cast<T>(0.08));
    std::uniform_real_distribution<T> look_at_pitch_dist(static_cast<T>(-0.06), static_cast<T>(0.06));
    std::uniform_real_distribution<T> yaw_dist(-PI, PI);
    std::uniform_real_distribution<T> pitch_dist(static_cast<T>(-0.35), static_cast<T>(0.35));

    const CameraOffset offset = camera_offset(scene);
    const T eye[3] = {offset.x, offset.y, offset.z};
    const T target[3] = {metadata.center[0], metadata.center[1], metadata.center[2]};
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
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_staging;
    malloc(device, camera_staging);
    for(int i = 0; i < static_cast<int>(SPEC::NUM_CAMERAS); i++) {
        const CameraPose& pose = poses[i];
        const T look_at[3] = {
            eye[0] + pose.direction[0],
            eye[1] + pose.direction[1],
            eye[2] + pose.direction[2]
        };
        data(camera_staging)[i] = make_camera_data(eye, look_at, pose.up, FOV, aspect);
    }
    copy(device, renderer.device, camera_staging, cameras(device, renderer));
    free(device, camera_staging);
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
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> staging;
    malloc(device, staging);
    data(staging)[0] = make_camera_data(eye, look_at, pose.up, FOV, aspect);
    copy(device, renderer.device, staging, cameras(device, renderer));
    free(device, staging);
}

[[maybe_unused]] static CameraPose make_single_frame_pose(const rlt::rendering::SceneMetadata<T>& metadata, const Options& options) {
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
        const T* target = options.has_look_at ? options.look_at : metadata.center;
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
static void step_paper_physics(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer,
                               paper::PhysicsSimulation& physics, CameraBuffer<SPEC>& physics_cameras,
                               const Options& options, int iteration) {
    if constexpr (CONFIG::PHYSICS_AVAILABLE) {
        if(options.workload != "render_only") {
            if constexpr (CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cuda) {
                void* destination = options.workload == "render_physics_coupled" ? data(cameras(device, renderer)) : nullptr;
                if(!physics_step_cameras(device, physics, destination, stream(device, renderer), iteration)) {
                    throw std::runtime_error(physics_last_error(device, physics));
                }
            }
            else {
                auto* destination = options.workload == "render_physics_coupled" ? data(physics_cameras) : nullptr;
                if(!physics_step_cameras(device, physics, destination, iteration)) {
                    throw std::runtime_error(physics_last_error(device, physics));
                }
                if(options.workload == "render_physics_coupled") {
                    copy(device, renderer.device, physics_cameras, cameras(device, renderer));
                }
            }
        }
    }
}

template <typename DEVICE, typename SPEC>
static BenchmarkResult run_benchmark(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer,
                                     const Options& options, paper::PhysicsSimulation& physics,
                                     CameraBuffer<SPEC>& physics_cameras) {
    using Clock = std::chrono::steady_clock;
    const auto warmup_start = Clock::now();
    int warmup_iterations = 0;
    do {
        step_paper_physics(device, renderer, physics, physics_cameras, options, warmup_iterations++);
        render(device, renderer);
    } while(options.warmup_seconds > 0
        ? std::chrono::duration<double>(Clock::now() - warmup_start).count() < options.warmup_seconds
        : warmup_iterations < (CONFIG::PAPER ? 10 : 1));

    synchronize(device, renderer);
    if constexpr (CONFIG::PHYSICS_AVAILABLE) {
        if(!synchronize_physics(device, physics)) {
            throw std::runtime_error(physics_last_error(device, physics));
        }
    }
    const auto start = Clock::now();
    int iterations = 0;
    for(;;) {
        step_paper_physics(device, renderer, physics, physics_cameras, options, iterations);
        render_launch(device, renderer);
        iterations++;
        if(iterations % options.sync_interval == 0) {
            render_sync(device, renderer);
            if(options.iterations == 0 && std::chrono::duration<double>(Clock::now() - start).count() >= options.seconds) {
                break;
            }
        }
        if(options.iterations > 0 && iterations == options.iterations) {
            break;
        }
    }
    synchronize(device, renderer);
    if constexpr (CONFIG::PHYSICS_AVAILABLE) {
        if(!synchronize_physics(device, physics)) {
            throw std::runtime_error(physics_last_error(device, physics));
        }
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const double frames = static_cast<double>(iterations) * SPEC::NUM_CAMERAS;
    const double pixels = frames * SPEC::CAM_PIXELS;
    return {iterations, elapsed, frames / elapsed, pixels / elapsed,
        pixels * samples_per_pixel_for_spec<SPEC>() / (1e6 * elapsed)};
}

template <typename SPEC, typename DEVICE>
static bool setup_scene(DEVICE& device, rlt::rendering::Bundle<T>& bundle, SceneAxis scene, const Options& options) {
    if(scene == SceneAxis::OBJECTS_20) {
        make_20_object_scene(bundle.scene);
        apply_20_object_bounds(bundle.metadata);
        return true;
    }
    if constexpr (CONFIG::PAPER) {
        std::string hash;
        if(!conta::detail::sha1_file(options.procthor_path, hash) || hash != PAPER_SCENE_SHA1) {
            std::cerr << "Paper ProcTHOR asset missing or mismatched: " << options.procthor_path
                << ". Run download_paper_scene.py; expected SHA1 " << PAPER_SCENE_SHA1 << std::endl;
            return false;
        }
    }
    rlt::log(device, device.logger, "Loading ProcTHOR scene: ", options.procthor_path);
    return load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, options.procthor_path);
}

template <typename DEVICE, typename SPEC>
static void save_verification_image(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, const std::string& filename) {
    if constexpr (SPEC::HAS_DEPTH) {
        save_depth_image(device, renderer, filename.c_str());
    }
    else {
        save_image(device, renderer, filename.c_str());
    }
}

template <typename VALUE>
static std::string number(VALUE value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}

static std::string vector_string(const T* values, int count) {
    std::string result;
    for(int i = 0; i < count; i++) {
        if(i > 0) {
            result += ' ';
        }
        result += number(values[i]);
    }
    return result;
}

template <typename SPEC>
static std::string case_name(SceneAxis scene, const Options& options, const std::string& device,
                             const std::string& scene_id, const CameraPose* pose) {
    std::string orientation = options.orientation_mode + "_seed" + std::to_string(options.seed);
    if(pose != nullptr) {
        const std::string camera = vector_string(options.position, 3) + ";"
            + vector_string(pose->direction, 3) + ";" + vector_string(pose->up, 3);
        conta::detail::Sha1 hash;
        hash.update(reinterpret_cast<const unsigned char*>(camera.data()), camera.size());
        orientation = "pose" + hash.finalize().substr(0, 12);
    }
    const std::string prefix = CONFIG::PAPER ? "paper_may2026_" + options.workload + "_"
        : (CONFIG::MODE == BenchmarkMode::frame ? "frame_" : "render_");
    return prefix + BACKEND_NAME + "_"
        + sanitize_label(device) + "_" + scene_name(scene) + "_" + scene_id + "_" + output_name_for_spec<SPEC>() + "_"
        + shading_profile_name_for_spec<SPEC>() + "_" + anti_aliasing_name_for_spec<SPEC>() + "_"
        + std::to_string(SPEC::NUM_CAMERAS) + "views_" + std::to_string(SPEC::CAM_WIDTH) + "x"
        + std::to_string(SPEC::CAM_HEIGHT) + "_" + orientation;
}

template <typename SPEC>
static bool report(SceneAxis scene, const Options& options, const std::string& device,
                   const BenchmarkResult& result, const std::string& image, const std::string& scene_id,
                   const std::string& identity, const CameraPose* pose) {
    const auto offset = camera_offset(scene);
    const T eye[3] = {offset.x, offset.y, offset.z};
    const std::string reference = scene == SceneAxis::PROCTHOR ? options.procthor_path : OBJECTS20_LAYOUT_NAME;
    const bool frame = CONFIG::MODE == BenchmarkMode::frame;
    const T* position = frame ? options.position : eye;
    nlohmann::json record = {
        {"schema", "render-v1"},
        {"workload", frame ? "frame" : options.workload},
        {"backend", BACKEND_NAME},
        {"device", device},
        {"scene", scene_name(scene)},
        {"scene_source", reference},
        {"scene_id", scene_id},
        {"output", output_name_for_spec<SPEC>()},
        {"shading_profile", shading_profile_name_for_spec<SPEC>()},
        {"num_cameras", SPEC::NUM_CAMERAS},
        {"width", SPEC::CAM_WIDTH},
        {"height", SPEC::CAM_HEIGHT},
        {"fov_degrees", FOV},
        {"aa_grid_size", SPEC::ANTI_ALIASING_GRID_SIZE},
        {"samples_per_pixel", samples_per_pixel_for_spec<SPEC>()},
        {"motion_blur", false},
        {"num_probes", SPEC::NUM_PROBES},
        {"camera_position_flu", {position[0], position[1], position[2]}},
        {"orientation_mode", frame ? "explicit" : options.orientation_mode},
        {"camera_forward_flu", pose == nullptr ? nlohmann::json(nullptr) : nlohmann::json(pose->direction)},
        {"camera_up_flu", pose == nullptr ? nlohmann::json(nullptr) : nlohmann::json(pose->up)},
        {"seed", options.seed},
        {"warmup_seconds", frame ? 0.0 : options.warmup_seconds},
        {"sync_interval", frame ? 1 : options.sync_interval},
        {"iterations", result.iterations},
        {"elapsed_s", result.elapsed_s},
        {"frames_per_s", result.frames_per_s},
        {"pixels_per_s", result.pixels_per_s},
        {"primary_mrays_per_s", result.primary_mrays_per_s},
        {"verification_png", image}
    };
    if constexpr (CONFIG::PAPER) {
        record["benchmark_contract"] = "hyperdrone-paper-may2026";
        record["repeat"] = options.repeat;
        record["repeats"] = options.repeats;
        record["physics_backend"] = options.workload == "render_only" ? nlohmann::json(nullptr) : nlohmann::json(PHYSICS_BACKEND_NAME);
        record["rltools_commit"] = RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH);
        record["workload_source_sha256"] = CONFIG::WORKLOAD_SOURCE_SHA256;
#ifdef RL_TOOLS_EXTRACK_GIT_DIFF
        record["rltools_commit"] = rlt::utils::extrack::git::rl_tools::commit;
        record["tracked_source_dirty"] = rlt::utils::extrack::git::rl_tools::diff[0] != '\0'
            || rlt::utils::extrack::git::rl_tools::diff_staged[0] != '\0';
#else
        record["tracked_source_dirty"] = nullptr;
#endif
        record["historical_source_commit"] = options.workload == "render_physics_coupled"
            ? "6c7fc4c6e0c912c451e40856b80740e65f129635" : "10bee5cdc6616e6c42c16cf8966abff3f007ef06";
        record["historical_commit_attribution"] = "inferred_from_paper_runs_and_history";
        record["scene_sha256"] = scene == SceneAxis::PROCTHOR
            ? nlohmann::json("76c463017f2a25683abaea3b29786294602bd2629b2f0725b7215dabf58f35aa") : nlohmann::json(nullptr);
        record["physics"] = options.workload == "render_only" ? nlohmann::json(nullptr) : nlohmann::json{
            {"model", "crazyflie"}, {"action_interface", "direct_motor"}, {"integrator", "rk4"},
            {"dt", 0.01}, {"substeps", 1}, {"reset_interval", 10}, {"action_amplitude", 0.01},
            {"camera_coupled", options.workload == "render_physics_coupled"}, {"collision_response", false}
        };
    }
    const std::string line = record.dump();
    std::cout << line << '\n';
    if constexpr (CONFIG::PAPER) {
        std::ofstream results(options.output_jsonl, std::ios::app);
        results << line << '\n';
        results.close();
        if(!results) {
            std::cerr << "Cannot write benchmark results to '" << options.output_jsonl << "'" << std::endl;
            return false;
        }
    }

    const std::string physics_suffix = CONFIG::PAPER && options.workload != "render_only"
        ? std::string("/physics-") + PHYSICS_BACKEND_NAME : "";
    const std::string prefix = "raytracing/benchmark/render-v1/" + identity + physics_suffix + "/sync" + std::to_string(frame ? 1 : options.sync_interval);
    metra::log(prefix + "/elapsed_s", result.elapsed_s);
    metra::log(prefix + "/frames_per_s", result.frames_per_s);
    metra::log(prefix + "/pixels_per_s", result.pixels_per_s);
    metra::log(prefix + "/primary_mrays_per_s", result.primary_mrays_per_s);
    return true;
}

template <typename SPEC, typename DEVICE>
static bool run_case(DEVICE& device, RUNTIME& runtime, SceneAxis scene, const Options& options) {
    rlt::rendering::Bundle<T> bundle;
    if(!setup_scene<SPEC>(device, bundle, scene, options)) {
        return false;
    }
    rlt::rendering::raytracing::Renderer<SPEC> renderer;
    malloc(device, renderer, runtime);
    init(device, renderer, bundle);
    const std::string device_label = device_name(device, renderer);
    rlt::log(device, device.logger, "Rendering ", scene_name(scene), " on ", BACKEND_NAME, " / ", device_label);

    CameraPose pose{};
    BenchmarkResult result{};
    paper::PhysicsSimulation physics;
    CameraBuffer<SPEC> physics_cameras;
    if constexpr (CONFIG::MODE == BenchmarkMode::frame) {
        pose = make_single_frame_pose(bundle.metadata, options);
        write_single_camera(device, renderer, options.position, pose);
        synchronize(device, renderer);
        const auto start = std::chrono::steady_clock::now();
        render(device, renderer);
        synchronize(device, renderer);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result = {1, elapsed, 1 / elapsed, SPEC::CAM_PIXELS / elapsed,
            SPEC::CAM_PIXELS * samples_per_pixel_for_spec<SPEC>() / (1e6 * elapsed)};
    }
    else {
        const auto poses = make_camera_poses<SPEC>(bundle.metadata, scene, options);
        write_cameras(device, renderer, scene, poses);
        if constexpr (CONFIG::PHYSICS_AVAILABLE) {
            if(options.workload != "render_only") {
                std::vector<paper::PhysicsCameraPose> physics_poses(poses.size());
                for(size_t i = 0; i < poses.size(); i++) {
                    std::copy_n(poses[i].direction, 3, physics_poses[i].direction);
                    std::copy_n(poses[i].up, 3, physics_poses[i].up);
                }
                const auto offset = camera_offset(scene);
                const T position[3] = {offset.x, offset.y, offset.z};
                if(!init_physics_simulation(device, physics, SPEC::NUM_CAMERAS, position, physics_poses.data(),
                                           FOV, static_cast<T>(SPEC::CAM_WIDTH) / SPEC::CAM_HEIGHT, options.seed)) {
                    std::cerr << physics_last_error(device, physics) << std::endl;
                    free(device, renderer);
                    return false;
                }
            }
        }
        if constexpr (CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cpu) {
            if(options.workload == "render_physics_coupled") {
                malloc(device, physics_cameras);
            }
        }
        try {
            result = run_benchmark(device, renderer, options, physics, physics_cameras);
        }
        catch(const std::runtime_error& error) {
            std::cerr << error.what() << std::endl;
            if constexpr (CONFIG::PHYSICS_AVAILABLE) {
                free_physics_simulation(device, physics);
            }
            if constexpr (CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cpu) {
                if(options.workload == "render_physics_coupled") {
                    free(device, physics_cameras);
                }
            }
            free(device, renderer);
            return false;
        }
        if constexpr (CONFIG::PHYSICS_AVAILABLE) {
            free_physics_simulation(device, physics);
        }
        if constexpr (CONFIG::PHYSICS_BACKEND == paper::PhysicsBackend::cpu) {
            if(options.workload == "render_physics_coupled") {
                free(device, physics_cameras);
            }
        }
    }

    const auto* frame_pose = CONFIG::MODE == BenchmarkMode::frame ? &pose : nullptr;
    const std::string scene_id = scene == SceneAxis::PROCTHOR ? bundle.metadata.content_hash : OBJECTS20_LAYOUT_NAME;
    const std::string identity = case_name<SPEC>(scene, options, device_label, scene_id, frame_pose);
    const std::string image_name = identity + (CONFIG::PAPER ? "_repeat" + std::to_string(options.repeat) : "") + ".png";
    const std::string image = options.output_png.empty()
        ? join_path(options.output_dir, image_name)
        : options.output_png;
    const auto parent = std::filesystem::path(image).parent_path();
    if(!parent.empty() && !ensure_output_dir(parent.string())) {
        free(device, renderer);
        return false;
    }
    save_verification_image(device, renderer, image);
    const bool reported = report<SPEC>(scene, options, device_label, result, image, scene_id, identity, frame_pose);
    free(device, renderer);
    return reported;
}

template <typename SPEC, typename DEVICE>
static bool run_scenes(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes, const Options& options) {
    for(SceneAxis scene : scenes) {
        if(options.workload == "all") {
            for(const char* workload : {"render_only", "render_physics_static"}) {
                auto selected = options;
                selected.workload = workload;
                if(!run_case<SPEC>(device, runtime, scene, selected)) {
                    return false;
                }
            }
        }
        else {
            if(!run_case<SPEC>(device, runtime, scene, options)) {
                return false;
            }
        }
    }
    return true;
}

template <typename SPEC, typename DEVICE>
static bool run_sweep_spec(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes,
                           const Options& options, bool& first, int& index) {
    const int current = index++;
    if(current < options.sweep_config_start || (options.sweep_config_count > 0
       && current - options.sweep_config_start >= options.sweep_config_count)) {
        return true;
    }
    if(!first && options.cooldown_seconds > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(options.cooldown_seconds));
    }
    first = false;
    rlt::log(device, device.logger, "Sweep configuration ", current);
    return run_scenes<SPEC>(device, runtime, scenes, options);
}

template <TI RESOLUTION, bool AA, typename DEVICE>
static bool run_sweep_resolution(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes,
                                const std::vector<OutputAxis>& outputs, const Options& options, bool& first, int& index) {
    constexpr TI COUNT = SweepCameraCount<RESOLUTION>::VALUE;
    constexpr TI GRID = AA ? 2 : 1;
    for(OutputAxis output : outputs) {
        if(output == OutputAxis::RGB) {
            using MEDIUM = SweepBenchmarkSpec<true, false, RESOLUTION, RESOLUTION, COUNT, rlt::rendering::raytracing::Medium, AA, GRID>;
            using HIGH = SweepBenchmarkSpec<true, false, RESOLUTION, RESOLUTION, COUNT, rlt::rendering::raytracing::High, AA, GRID>;
            using LOW = SweepBenchmarkSpec<true, false, RESOLUTION, RESOLUTION, COUNT, rlt::rendering::raytracing::Low, AA, GRID>;
            if(!run_sweep_spec<MEDIUM>(device, runtime, scenes, options, first, index)
               || !run_sweep_spec<HIGH>(device, runtime, scenes, options, first, index)
               || !run_sweep_spec<LOW>(device, runtime, scenes, options, first, index)) {
                return false;
            }
        }
        else {
            using DEPTH = SweepBenchmarkSpec<false, true, RESOLUTION, RESOLUTION, COUNT, rlt::rendering::raytracing::Medium, AA, GRID>;
            if(!run_sweep_spec<DEPTH>(device, runtime, scenes, options, first, index)) {
                return false;
            }
        }
    }
    return true;
}

template <TI RESOLUTION = 64, typename DEVICE>
static bool run_sweep(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes,
                      const std::vector<OutputAxis>& outputs, const Options& options, bool& first, int& index) {
    if(!run_sweep_resolution<RESOLUTION, false>(device, runtime, scenes, outputs, options, first, index)
       || !run_sweep_resolution<RESOLUTION, true>(device, runtime, scenes, outputs, options, first, index)) {
        return false;
    }
    if constexpr (RESOLUTION < 2048) {
        return run_sweep<RESOLUTION * 2>(device, runtime, scenes, outputs, options, first, index);
    }
    if(first) {
        std::cerr << "Sweep selection contains no configurations (available: " << index << ")." << std::endl;
        return false;
    }
    return true;
}

template <typename SHADING, typename DEVICE>
static bool run_profile(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes,
                        const std::vector<OutputAxis>& outputs, const Options& options) {
    for(OutputAxis output : outputs) {
        if(output == OutputAxis::RGB) {
            if(!run_scenes<BenchmarkSpec<true, false, SHADING>>(device, runtime, scenes, options)) {
                return false;
            }
        }
        else {
            if(!run_scenes<BenchmarkSpec<false, true, SHADING>>(device, runtime, scenes, options)) {
                return false;
            }
        }
    }
    return true;
}

template <typename CONFIG, typename DEVICE>
static bool run(DEVICE& device, RUNTIME& runtime, const std::vector<SceneAxis>& scenes,
                const std::vector<OutputAxis>& outputs, const Options& options) {
    if constexpr (CONFIG::MODE == BenchmarkMode::sweep) {
        bool first = true;
        int index = 0;
        return run_sweep(device, runtime, scenes, outputs, options, first, index);
    }
    else if constexpr (CONFIG::PAPER) {
        if((options.fidelity == "low" || options.fidelity == "all")
           && !run_profile<paper::PaperLow>(device, runtime, scenes, outputs, options)) {
            return false;
        }
        if((options.fidelity == "medium" || options.fidelity == "all")
           && !run_profile<rlt::rendering::raytracing::Medium>(device, runtime, scenes, outputs, options)) {
            return false;
        }
        if((options.fidelity == "high" || options.fidelity == "all")
           && !run_profile<paper::PaperHigh>(device, runtime, scenes, outputs, options)) {
            return false;
        }
        return true;
    }
    else {
        return run_profile<ShadingProfile>(device, runtime, scenes, outputs, options);
    }
}

int rl_tools::rendering::raytracing::benchmark::benchmark_main(int argc, char** argv) {
    Options options;
    std::vector<SceneAxis> scenes;
    std::vector<OutputAxis> outputs;
    if(!parse_options(argc, argv, options) || !select_scenes(options.scene, scenes)
       || !select_outputs(options.output, outputs) || !ensure_output_dir(options.output_dir)) {
        return 1;
    }
    if constexpr (CONFIG::PAPER) {
        const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() % 1000000;
        std::ostringstream filename;
        filename << "rendering_raytracing_paper_benchmark_results_"
                 << rlt::utils::extrack::get_timestamp_string() << '_'
                 << std::setfill('0') << std::setw(6) << microseconds << ".jsonl";
        options.output_jsonl = join_path(options.output_dir, filename.str());
        std::ofstream results(options.output_jsonl, std::ios::app);
        if(!results) {
            std::cerr << "Cannot open benchmark results file '" << options.output_jsonl << "'" << std::endl;
            return 1;
        }
        std::cout << "Benchmark results: " << options.output_jsonl << std::endl;
    }
    DEVICE device;
    init(device);
    RUNTIME runtime;
    malloc(device, runtime);
    int exit_code = 0;
    for(int repeat = 0; repeat < options.repeats; repeat++) {
        options.repeat = repeat + 1;
        if(!run<CONFIG>(device, runtime, scenes, outputs, options)) {
            exit_code = 1;
            break;
        }
    }
    free(device, runtime);
    return exit_code;
}
