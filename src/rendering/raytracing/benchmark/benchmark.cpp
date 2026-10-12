#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_WIDTH
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_WIDTH 128
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_HEIGHT
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_HEIGHT 128
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_CAMERAS
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_CAMERAS 4096
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_PROBES
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_PROBES 64
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include "../camera_orbit.h"
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include <string>
#include <iostream>
#include <chrono>

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;

using T = float;
static constexpr T FOV = 80;
using TI = int;
struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_WIDTH;
    static constexpr TI CAM_HEIGHT = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_HEIGHT;
    static constexpr TI NUM_CAMERAS = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_CAMERAS;
    static constexpr TI NUM_PROBES = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_NUM_PROBES;
    using OUTPUTS = typename rro::Concat<
        rro::If<(RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH), rro::RGB<>>,
        rro::If<(RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB), rro::Depth<>>,
        rro::List<rro::Probes>
    >::type;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
static constexpr double BENCHMARK_SECONDS = 10.0;

int main(int ac, char** av){
    DEVICE device;
    rlt::init(device);
    rlt::log(device, device.logger, "rl_tools::rendering::raytracing benchmark '", av[0], "' starting up");

    // Parse command line
    std::string model_file;
    for(int i = 1; i < ac; i++){
        std::string arg = av[i];
        if((arg == "-m" || arg == "--model") && i + 1 < ac){
            model_file = av[++i];
        } else if(arg == "-h" || arg == "--help"){
            std::cout << "Usage: " << av[0] << " [options]\n"
                      << "Options:\n"
                      << "  -m, --model <file>  Load a 3D model (GLB, OBJ, FBX, etc.)\n"
                      << "  -h, --help          Show this help message\n";
            return 0;
        } else if(arg[0] != '-'){
            model_file = arg;
        }
    }

    rlt::rendering::raytracing::Renderer<SPEC> renderer;

    rlt::malloc(device, renderer);

    if(model_file.empty()){
        rlt::log(device, device.logger, rlt::devices::logging::Verbosity::error, "No model specified. Use --model <file>.");
        return 1;
    }
    rlt::log(device, device.logger, "Loading model: ", model_file);
    rlt::rendering::Bundle<T> bundle;
    if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, model_file)){
        rlt::log(device, device.logger, rlt::devices::logging::Verbosity::error, "Failed to load model: ", model_file);
        return 1;
    }

    rlt::init(device, renderer, bundle);

    // Generate cameras
    const T scene_center[3] = {bundle.metadata.center[0], bundle.metadata.center[1], bundle.metadata.center[2]};
    const T look_up[3] = {0.f, 0.f, 1.f};

    camera_orbit::write(device, renderer, scene_center, (bundle.metadata.max_ray_length / 2), look_up, FOV);
    rlt::generate_probe_directions(device, renderer);

    // Warmup
    rlt::render(device, renderer);
    rlt::probe(device, renderer);
    rlt::log(device, device.logger, "Warmup launch complete");

    // Benchmark
    int num_iterations = 0;
    rlt::log(device, device.logger, "Starting benchmark (async): ", SPEC::NUM_CAMERAS, " cameras at ",
          SPEC::CAM_WIDTH, "x", SPEC::CAM_HEIGHT, " + ", SPEC::NUM_PROBES,
          " probes/cam for ~", BENCHMARK_SECONDS, "s ...");

    rlt::render_sync(device, renderer);
    rlt::probe_sync(device, renderer);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for(;;){
        rlt::render_launch(device, renderer);
        rlt::probe_launch(device, renderer);
        num_iterations++;

        if(num_iterations % 10 == 0){
            rlt::render_sync(device, renderer);
            rlt::probe_sync(device, renderer);
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            if(elapsed >= BENCHMARK_SECONDS) break;
        }
    }

    rlt::render_sync(device, renderer);
    rlt::probe_sync(device, renderer);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Report
    long long total_frames = (long long)num_iterations * SPEC::NUM_CAMERAS;
    long long total_rgb_pixels = (long long)SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT * total_frames;
    long long total_rgb_rays = SPEC::HAS_RGB ? total_rgb_pixels * SPEC::RGB_SAMPLES : 0;
    long long total_depth_rays = SPEC::HAS_DEPTH ? total_rgb_pixels : 0;
    long long total_probe_rays = (long long)num_iterations * SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    long long total_rays = total_rgb_rays + total_depth_rays + total_probe_rays;
    double total_mrays = total_rays / 1e6;

    rlt::log(device, device.logger, "=== BENCHMARK RESULTS (async) ===");
    rlt::log(device, device.logger, "  Cameras per batch:   ", SPEC::NUM_CAMERAS);
    rlt::log(device, device.logger, "  Resolution per cam:  ", SPEC::CAM_WIDTH, "x", SPEC::CAM_HEIGHT);
    rlt::log(device, device.logger, "  Probes per camera:   ", SPEC::NUM_PROBES);
    rlt::log(device, device.logger, "  Batch iterations:    ", num_iterations);
    rlt::log(device, device.logger, "  Total frames:        ", total_frames);
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    rlt::log(device, device.logger, "  Total RGB pixels:    ", total_rgb_pixels);
    rlt::log(device, device.logger, "  RGB samples/pixel:   ", SPEC::RGB_SAMPLES);
#endif
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
    rlt::log(device, device.logger, "  Depth samples/pixel: ", 1);
    rlt::log(device, device.logger, "  Total depth rays:    ", total_depth_rays);
#endif
    rlt::log(device, device.logger, "  Total probe rays:    ", total_probe_rays);
    rlt::log(device, device.logger, "  Wall-clock time:     ", wall_ms, " ms");
    rlt::log(device, device.logger, "  Avg per iter:        ", wall_ms / num_iterations, " ms");
    rlt::log(device, device.logger, "  Avg per frame:       ", wall_ms / total_frames, " ms");
    rlt::log(device, device.logger, "  Throughput:          ", (total_frames / (wall_ms / 1000.0)), " frames/sec");
    rlt::log(device, device.logger, "  Total MRays/sec:     ", (total_mrays / (wall_ms / 1000.0)));
    rlt::log(device, device.logger, "=========================");

    // Save outputs
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    const char* out_filename = "owl_test.png";
#endif
    const char* probe_out_filename = "owl_probes.bin";

#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    rlt::save_image(device, renderer, out_filename);
#endif
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
    rlt::save_depth_image(device, renderer, "owl_depth.png");
    rlt::save_depth(device, renderer, "owl_depth.bin");
#endif
    rlt::save_probes(device, renderer, probe_out_filename);

    rlt::free(device, renderer);

    rlt::log(device, device.logger, "seems all went OK; app is done, this should be the last output ...");
    return 0;
}
