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

static constexpr TI CAM_WIDTH = 2048;
static constexpr TI CAM_HEIGHT = 1024;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;

using SPEC = rlt::rendering::raytracing::Specification<
    T,
    TI,
    CAM_WIDTH,
    CAM_HEIGHT,
    NUM_CAMERAS,
    NUM_PROBES,
    rlt::rendering::raytracing::VeryHigh,
    false,
    1,
    true,
    2
>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr char DEFAULT_SCENE_PATH[] = "/home/jonas/git/hssd-hab/glb/102343992.glb";

struct Options {
    std::string scene_path;
    std::string output_path = "rendering_raytracing_fixed_pose.png";
};

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--scene path.glb|conta:HASH] [--output path.png]\n"
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

    if(!rlt::load_model(device, renderer, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        rlt::free(device, renderer);
        return 1;
    }

    rlt::upload_geometry(device, renderer);

    constexpr T position[3] = {
        static_cast<T>(12.08),
        static_cast<T>(7.23),
        static_cast<T>(2.26)
    };
    constexpr T orientation_wxyz[4] = {
        static_cast<T>(0.656),
        static_cast<T>(0.117),
        static_cast<T>(0.104),
        static_cast<T>(-0.738)
    };
    constexpr T forward_body[3] = {static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)};
    constexpr T up_body[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
    T forward[3];
    T up[3];
    rotate_vector_by_quaternion(orientation_wxyz, forward_body, forward);
    rotate_vector_by_quaternion(orientation_wxyz, up_body, up);

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
