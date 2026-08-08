#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include "golden_cases.h"
#include "golden_io.h"
#include "../../utils/utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef RL_TOOLS_TEST_DATA_PATH
#error "RL_TOOLS_TEST_DATA_PATH is required"
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using CASES = golden::Cases<T, TI>;

static const std::string SCENE_PATH = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/ProcTHOR-Train-1.glb";
static const std::string OUTPUT_DIR = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/rendering_raytracing_golden";

template <typename SPEC>
bool run_case(DEVICE& device, const char* name, bool write_probes) {
    using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;
    std::cout << "[golden] rendering " << name << std::endl;

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::rendering::raytracing::Scene scene;
    if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, SCENE_PATH)) {
        std::cerr << "[golden] " << name << ": failed to load scene: " << SCENE_PATH << std::endl;
        rlt::free(device, renderer);
        return false;
    }
    rlt::init(device, renderer, scene);

    constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        const golden::Pose<T>& pose = CASES::POSES[camera_i];
        rlt::set(device, renderer.cameras, rlt::make_camera_data(pose.position, pose.look_at, pose.up, SPEC::COS_FOVY, aspect), camera_i);
    }
    if constexpr(SPEC::ENABLE_MOTION_BLUR) {
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const golden::Pose<T>& pose = CASES::POSES[camera_i];
            T position[3], look_at[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++) {
                position[dim_i] = pose.position[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                look_at[dim_i] = pose.look_at[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
            }
            rlt::set(device, renderer.cameras_open, rlt::make_camera_data(position, look_at, pose.up, SPEC::COS_FOVY, aspect), camera_i);
        }
        rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
    }
    else {
        rlt::set_cameras(device, renderer, renderer.cameras);
    }
    rlt::generate_probe_directions(device, renderer);
    rlt::render(device, renderer);
        rlt::probe(device, renderer);
    cudaDeviceSynchronize();

    bool ok = true;
    rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
    const uint32_t* frame_buffer = rlt::data(renderer.frame_buffer);
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(std::all_of(frame_buffer, frame_buffer + pixel_count, [&](uint32_t pixel){ return pixel == frame_buffer[0]; })) {
        std::cerr << "[golden] " << name << ": frame buffer is constant" << std::endl;
        ok = false;
    }
    if constexpr(SPEC::HAS_DEPTH) {
        rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
        const float* depth_buffer = rlt::data(renderer.depth_buffer);
        if(std::all_of(depth_buffer, depth_buffer + pixel_count, [&](float depth){ return depth == depth_buffer[0]; })) {
            std::cerr << "[golden] " << name << ": depth buffer is constant" << std::endl;
            ok = false;
        }
    }

    if(ok) {
        // per-pose layout: <OUTPUT_DIR>/<pose_id>/<case>.png etc. (see golden_io.h)
        const rlt::rendering::raytracing::CollisionResult* probe_results = nullptr;
        if(write_probes) {
            probe_results = rlt::read_collision_results_raw(device, renderer);
        }
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const std::string directory = OUTPUT_DIR + "/" + CASES::POSES[camera_i].id;
            std::filesystem::create_directories(directory);
            ok &= golden::write_camera_png(directory + "/" + std::string(name) + ".png", frame_buffer + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            if constexpr(SPEC::HAS_DEPTH) {
                const float* depth_buffer = rlt::data(renderer.depth_buffer);
                ok &= golden::write_camera_depth_bin(directory + "/" + std::string(name) + "_depth.bin", depth_buffer + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            if(write_probes && probe_results != nullptr) {
                ok &= golden::write_camera_probes(directory + "/probes.bin", probe_results + camera_i * SPEC::NUM_PROBES, SPEC::NUM_PROBES);
            }
        }
        if(!ok) {
            std::cerr << "[golden] " << name << ": failed to write outputs" << std::endl;
        }
    }

    rlt::free(device, renderer);
    return ok;
}

int main() {
    std::filesystem::create_directories(OUTPUT_DIR);

    DEVICE device;
    rlt::init(device);

    bool ok = true;
    ok &= run_case<CASES::LOW_RGB>(device, "low_rgb", true);
    ok &= run_case<CASES::MEDIUM_RGB>(device, "medium_rgb", false);
    ok &= run_case<CASES::HIGH_RGB>(device, "high_rgb", false);
    ok &= run_case<CASES::VERY_HIGH_RGB>(device, "very_high_rgb", false);
    ok &= run_case<CASES::HIGH_RGB_AA2>(device, "high_rgb_aa2", false);
    ok &= run_case<CASES::HIGH_RGB_MB4>(device, "high_rgb_mb4", false);
    ok &= run_case<CASES::LOW_RGBD>(device, "low_rgbd", false);
    ok &= run_case<CASES::HIGH_RGBD>(device, "high_rgbd", false);

    if(!ok) {
        std::cerr << "[golden] FAILED" << std::endl;
        return 1;
    }
    std::cout << "[golden] done: " << OUTPUT_DIR << std::endl;
    return 0;
}
