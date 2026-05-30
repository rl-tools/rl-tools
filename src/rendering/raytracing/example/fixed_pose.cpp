#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>

namespace rlt = rl_tools;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 640;
static constexpr TI CAM_HEIGHT = 480;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;

using SPEC = rlt::rendering::raytracing::Specification<
    T,
    TI,
    CAM_WIDTH,
    CAM_HEIGHT,
    NUM_CAMERAS,
    NUM_PROBES,
    rlt::rendering::raytracing::High
>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

struct Options {
    std::string scene_path;
    std::string output_path = "rendering_raytracing_fixed_pose.png";
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--scene path.glb|conta:HASH] [--output path.png]\n";
}

static std::string resolve_scene_arg(const std::string& scene_arg) {
    if(scene_arg.empty()) {
        const char* conta_root = std::getenv("CONTA_ROOT");
        if(conta_root) {
            static constexpr char DEFAULT_CONTA_HASH[] = "7f1c9129532798e0b63bc41edb6b4c09251cf8a0";
            return std::string(conta_root) + "/data/" + DEFAULT_CONTA_HASH;
        }
        return "tests/data/ProcTHOR-Train-1.glb";
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

static void yaw_pitch_from_interactive_quaternion(const T q[4], T& yaw, T& pitch) {
    yaw = static_cast<T>(2) * std::atan2(q[2], q[0]);
    if(std::abs(q[0]) > std::abs(q[2])) {
        pitch = static_cast<T>(2) * std::atan2(q[1], q[0]);
    }
    else {
        pitch = static_cast<T>(2) * std::atan2(-q[3], q[2]);
    }
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

    if(!rlt::load_model(device, renderer, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        rlt::free(device, renderer);
        return 1;
    }

    rlt::upload_geometry(device, renderer);

    constexpr T position[3] = {
        static_cast<T>(-11.35),
        static_cast<T>(9.93),
        static_cast<T>(1.48)
    };
    constexpr T orientation_wxyz[4] = {
        static_cast<T>(0.923),
        static_cast<T>(-0.119),
        static_cast<T>(0.363),
        static_cast<T>(0.047)
    };
    T yaw;
    T pitch;
    yaw_pitch_from_interactive_quaternion(orientation_wxyz, yaw, pitch);
    const T forward[3] = {
        std::cos(yaw) * std::cos(pitch),
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch)
    };
    const T up[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};

    const T look_at[3] = {
        position[0] + forward[0],
        position[1] + forward[1],
        position[2] + forward[2]
    };
    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);

    rlt::set(device, renderer.cameras, rlt::make_camera_data(position, look_at, up, SPEC::COS_FOVY, aspect), static_cast<TI>(0));
    rlt::set_cameras(device, renderer, renderer.cameras);
    rlt::build_pipeline(device, renderer);
    rlt::render_rgb_only(device, renderer);
    cudaDeviceSynchronize();
    rlt::save_image(device, renderer, options.output_path.c_str());

    std::cout << "Wrote " << options.output_path << std::endl;

    rlt::free(device, renderer);
    return 0;
}
