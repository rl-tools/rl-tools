#include <rl_tools/rendering/raytracing/operations_cuda.h>

#include <string>
#include <iostream>
#include <chrono>

namespace rlt = rl_tools;

using T = float;
using TI = int;
using SPEC = rlt::rendering::raytracing::Specification<T, TI, 128, 128, 4096, 64>;

struct Device{};

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

    Device device;
    rlt::rendering::raytracing::Renderer<SPEC> renderer;

    rlt::malloc(device, renderer);

    // Load model or default cube
    bool model_loaded = false;
    if(!model_file.empty()){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Loading model: " << model_file);
        model_loaded = rlt::load_model(device, renderer, model_file);
        if(!model_loaded){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to load model, using default cube");
        }
    }

    if(!model_loaded){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Using default cube");
        rlt::load_default_cube(device, renderer);
    }

    rlt::upload_geometry(device, renderer);

    // Generate cameras
    vec3f scene_center(renderer.scene_center[0], renderer.scene_center[1], renderer.scene_center[2]);
    const vec3f look_up(0.f, 1.f, 0.f);

    rlt::generate_cameras(device, renderer, scene_center, renderer.camera_radius, look_up, SPEC::COS_FOVY);
    rlt::generate_probe_directions(device, renderer);
    rlt::build_pipeline(device, renderer);

    // Warmup
    rlt::render(device, renderer);
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Warmup launch complete (RGB + collision)");

    // Benchmark
    int num_iterations = 0;
    RL_TOOLS_RENDERING_RAYTRACING_LOG("Starting benchmark (async): " << SPEC::NUM_CAMERAS << " cameras at "
          << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT << " + " << SPEC::NUM_PROBES
          << " probes/cam for ~" << SPEC::BENCHMARK_SECONDS << "s ...");

    OWLParams rgb_lp = (OWLParams)renderer.rgb_launch_params;
    OWLParams coll_lp = (OWLParams)renderer.coll_launch_params;
    OWLRayGen ray_gen = (OWLRayGen)renderer.ray_gen;
    OWLRayGen collision_ray_gen = (OWLRayGen)renderer.collision_ray_gen;

    owlLaunchSync(rgb_lp);
    owlLaunchSync(coll_lp);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for(;;){
        owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, rgb_lp);
        owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
        num_iterations++;

        if(num_iterations % 10 == 0){
            owlLaunchSync(rgb_lp);
            owlLaunchSync(coll_lp);
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            if(elapsed >= SPEC::BENCHMARK_SECONDS) break;
        }
    }

    owlLaunchSync(rgb_lp);
    owlLaunchSync(coll_lp);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Report
    long long total_frames = (long long)num_iterations * SPEC::NUM_CAMERAS;
    long long total_rgb_pixels = (long long)SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT * total_frames;
    long long total_probe_rays = (long long)num_iterations * SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    long long total_rays = total_rgb_pixels + total_probe_rays;
    double total_mrays = total_rays / 1e6;

    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("=== BENCHMARK RESULTS (RGB + COLLISION, async) ===");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Cameras per batch:   " << SPEC::NUM_CAMERAS);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Resolution per cam:  " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Probes per camera:   " << SPEC::NUM_PROBES);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Batch iterations:    " << num_iterations);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total frames:        " << total_frames);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total RGB pixels:    " << total_rgb_pixels);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total probe rays:    " << total_probe_rays);
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Wall-clock time:     " << wall_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Avg per iter:        " << wall_ms / num_iterations << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Avg per frame:       " << wall_ms / total_frames << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Throughput:          " << (total_frames / (wall_ms / 1000.0)) << " frames/sec");
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total MRays/sec:     " << (total_mrays / (wall_ms / 1000.0)));
    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("=========================");

    // Save outputs
    const char* out_filename = "owl_test.png";
    const char* probe_out_filename = "owl_probes.bin";

    rlt::save_image(device, renderer, out_filename);
    rlt::save_probes(device, renderer, probe_out_filename);

    rlt::free(device, renderer);

    RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("seems all went OK; app is done, this should be the last output ...");
    return 0;
}
