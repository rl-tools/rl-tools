#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include <string>
#include <iostream>
#include <chrono>

namespace rlt = rl_tools;

using T = float;
using TI = int;
static constexpr auto OUTPUT_MODE =
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        ? rlt::rendering::raytracing::OutputMode::DEPTH
        : (RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
            ? rlt::rendering::raytracing::OutputMode::RGBD
            : rlt::rendering::raytracing::OutputMode::RGB);
using SPEC = rlt::rendering::raytracing::Specification<T, TI, 128, 128, 4096, 64, rlt::rendering::raytracing::BasicShading, false, 1, false, 1, OUTPUT_MODE>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

int main(int ac, char** av){
    RL_TOOLS_RENDERING_RAYTRACING_LOG("rl_tools::rendering::raytracing benchmark '" << av[0] << "' starting up");

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

    DEVICE device;
    rlt::init(device);
    rlt::rendering::raytracing::Renderer<SPEC> renderer;

    rlt::malloc(device, renderer);

    if(model_file.empty()){
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("No model specified. Use --model <file>.");
        return 1;
    }
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Loading model: " << model_file);
    if(!rlt::load_model(device, renderer, model_file)){
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to load model: " << model_file);
        return 1;
    }

    rlt::upload_geometry(device, renderer);

    // Generate cameras
    const T scene_center[3] = {renderer.scene_center[0], renderer.scene_center[1], renderer.scene_center[2]};
    const T look_up[3] = {0.f, 0.f, 1.f};

    rlt::generate_cameras(device, renderer, scene_center, renderer.camera_radius, look_up, SPEC::COS_FOVY);
    rlt::generate_probe_directions(device, renderer);
    rlt::build_pipeline(device, renderer);

    // Warmup
    rlt::render(device, renderer);
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Warmup launch complete");

    // Benchmark
    int num_iterations = 0;
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Starting benchmark (async): " << SPEC::NUM_CAMERAS << " cameras at "
          << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT << " + " << SPEC::NUM_PROBES
          << " probes/cam for ~" << SPEC::BENCHMARK_SECONDS << "s ...");

    rlt::render_sync(device, renderer);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for(;;){
        rlt::render_launch(device, renderer);
        num_iterations++;

        if(num_iterations % 10 == 0){
            rlt::render_sync(device, renderer);
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            if(elapsed >= SPEC::BENCHMARK_SECONDS) break;
        }
    }

    rlt::render_sync(device, renderer);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Report
    long long total_frames = (long long)num_iterations * SPEC::NUM_CAMERAS;
    long long total_rgb_pixels = (long long)SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT * total_frames;
    long long total_rgb_rays = SPEC::HAS_RGB ? total_rgb_pixels * SPEC::RGB_SAMPLES : 0;
    long long total_depth_rays = SPEC::HAS_DEPTH ? total_rgb_pixels * SPEC::DEPTH_SAMPLES : 0;
    long long total_probe_rays = (long long)num_iterations * SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    long long total_rays = total_rgb_rays + total_depth_rays + total_probe_rays;
    double total_mrays = total_rays / 1e6;

    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("=== BENCHMARK RESULTS (async) ===");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Cameras per batch:   " << SPEC::NUM_CAMERAS);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Resolution per cam:  " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Probes per camera:   " << SPEC::NUM_PROBES);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Batch iterations:    " << num_iterations);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total frames:        " << total_frames);
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total RGB pixels:    " << total_rgb_pixels);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  RGB samples/pixel:   " << SPEC::RGB_SAMPLES);
#endif
#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Depth samples/pixel: " << SPEC::DEPTH_SAMPLES);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total depth rays:    " << total_depth_rays);
#endif
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total probe rays:    " << total_probe_rays);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Wall-clock time:     " << wall_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Avg per iter:        " << wall_ms / num_iterations << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Avg per frame:       " << wall_ms / total_frames << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Throughput:          " << (total_frames / (wall_ms / 1000.0)) << " frames/sec");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total MRays/sec:     " << (total_mrays / (wall_ms / 1000.0)));
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("=========================");

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

    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("seems all went OK; app is done, this should be the last output ...");
    return 0;
}
