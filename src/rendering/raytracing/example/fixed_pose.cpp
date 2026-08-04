#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#ifndef RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING
#define RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING 3
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE 0
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>


#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace rlt = rl_tools;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 2048;
static constexpr TI CAM_HEIGHT = 1024;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;

#if RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING == 0
using SHADING = rlt::rendering::raytracing::Low;
#elif RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING == 1
using SHADING = rlt::rendering::raytracing::Medium;
#elif RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_SHADING == 2
using SHADING = rlt::rendering::raytracing::High;
#else
using SHADING = rlt::rendering::raytracing::VeryHigh;
#endif

#if RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE == 2
static constexpr auto OUTPUT_MODE = rlt::rendering::raytracing::OutputMode::DEPTH;
#elif RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE == 1
static constexpr auto OUTPUT_MODE = rlt::rendering::raytracing::OutputMode::RGBD;
#else
static constexpr auto OUTPUT_MODE = rlt::rendering::raytracing::OutputMode::RGB;
#endif

using SPEC = rlt::rendering::raytracing::Specification<
    T,
    TI,
    CAM_WIDTH,
    CAM_HEIGHT,
    NUM_CAMERAS,
    NUM_PROBES,
    SHADING,
    false,
    1,
    true,
    2,
    OUTPUT_MODE
>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr char DEFAULT_SCENE_PATH[] = "/home/jonas/git/hssd-hab/glb/102343992.glb";

struct Options {
    std::string scene_path;
    std::string output_path = "rendering_raytracing_fixed_pose.png";
    T position[3] = {static_cast<T>(12.08), static_cast<T>(7.23), static_cast<T>(2.26)};
    T orientation_wxyz[4] = {static_cast<T>(0.656), static_cast<T>(0.117), static_cast<T>(0.104), static_cast<T>(-0.738)};
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--scene path.glb|conta:HASH] [--output path.png] [--position x,y,z] [--orientation w,x,y,z]\n"
        << "Default scene: " << DEFAULT_SCENE_PATH << "\n";
}

static std::string resolve_scene_arg(const std::string& scene_arg) {
    if(scene_arg.empty()) {
        return DEFAULT_SCENE_PATH;
    }
    if(scene_arg.compare(0, 6, "conta:") == 0) {
        const char* conta_root = std::getenv("CONTA_ROOT");
        if(!conta_root) {
            return "";
        }
        return std::string(conta_root) + "/data/" + scene_arg.substr(6);
    }
    return scene_arg;
}

static bool parse_options(int argc, char** argv, Options& options) {
    std::string scene_arg;
    for(int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if(arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if(arg == "--scene" || arg == "-s") {
            if(i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return false;
            }
            scene_arg = argv[++i];
        }
        else if(arg == "--output" || arg == "-o") {
            if(i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return false;
            }
            options.output_path = argv[++i];
        }
        else if(arg == "--position" || arg == "-p") {
            if(i + 1 >= argc || std::sscanf(argv[++i], "%f,%f,%f", &options.position[0], &options.position[1], &options.position[2]) != 3) {
                std::cerr << "Invalid value for " << arg << " (expected x,y,z)" << std::endl;
                return false;
            }
        }
        else if(arg == "--orientation" || arg == "-q") {
            if(i + 1 >= argc || std::sscanf(argv[++i], "%f,%f,%f,%f", &options.orientation_wxyz[0], &options.orientation_wxyz[1], &options.orientation_wxyz[2], &options.orientation_wxyz[3]) != 4) {
                std::cerr << "Invalid value for " << arg << " (expected w,x,y,z)" << std::endl;
                return false;
            }
        }
        else if(!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
        else if(scene_arg.empty()) {
            scene_arg = arg;
        }
        else {
            std::cerr << "Multiple scene paths provided" << std::endl;
            return false;
        }
    }
    options.scene_path = resolve_scene_arg(scene_arg);
    if(options.scene_path.empty()) {
        std::cerr << "CONTA_ROOT is required for conta: scene paths" << std::endl;
        return false;
    }
    return true;
}

static void rotate_vector_by_quaternion(const T q[4], const T v[3], T out[3]) {
    const T uv[3] = {
        q[2] * v[2] - q[3] * v[1],
        q[3] * v[0] - q[1] * v[2],
        q[1] * v[1] - q[2] * v[0]
    };
    const T uuv[3] = {
        q[2] * uv[2] - q[3] * uv[1],
        q[3] * uv[0] - q[1] * uv[2],
        q[1] * uv[1] - q[2] * uv[0]
    };
    out[0] = v[0] + static_cast<T>(2) * (q[0] * uv[0] + uuv[0]);
    out[1] = v[1] + static_cast<T>(2) * (q[0] * uv[1] + uuv[1]);
    out[2] = v[2] + static_cast<T>(2) * (q[0] * uv[2] + uuv[2]);
}

int main(int argc, char** argv) {
    Options options;
    if(!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }

    DEVICE device;
    rlt::init(device);

    Renderer renderer;
    rlt::malloc(device, renderer);

    rlt::rendering::raytracing::Scene scene;
    if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        rlt::free(device, renderer);
        return 1;
    }

    rlt::init(device, renderer, scene);

    constexpr T forward_body[3] = {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)};
    constexpr T up_body[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
    T forward[3];
    T up[3];
    rotate_vector_by_quaternion(options.orientation_wxyz, forward_body, forward);
    rotate_vector_by_quaternion(options.orientation_wxyz, up_body, up);

    const T look_at[3] = {
        options.position[0] + forward[0],
        options.position[1] + forward[1],
        options.position[2] + forward[2]
    };
    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);

    rlt::set(device, renderer.cameras, rlt::make_camera_data(options.position, look_at, up, SPEC::COS_FOVY, aspect), static_cast<TI>(0));
    rlt::set_cameras(device, renderer, renderer.cameras);
#if RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE == 2
    rlt::render_depth_only(device, renderer);
    rlt::synchronize(device, renderer);
    rlt::save_depth_image(device, renderer, options.output_path.c_str());
    rlt::save_depth(device, renderer, (options.output_path + ".bin").c_str());
#elif RL_TOOLS_RENDERING_RAYTRACING_FIXED_POSE_OUTPUT_MODE == 1
    rlt::render_rgb_depth_only(device, renderer);
    rlt::synchronize(device, renderer);
    rlt::save_image(device, renderer, options.output_path.c_str());
    rlt::save_depth_image(device, renderer, (options.output_path + ".depth.png").c_str());
    rlt::save_depth(device, renderer, (options.output_path + ".depth.bin").c_str());
#else
    rlt::render_rgb_only(device, renderer);
    rlt::synchronize(device, renderer);
    rlt::save_image(device, renderer, options.output_path.c_str());
#endif

    std::cout << "Wrote " << options.output_path << std::endl;

    rlt::free(device, renderer);
    return 0;
}
