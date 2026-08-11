#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_GENERATOR_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#include "golden_io.h"
#include "golden_render.h"
#include "../../utils/utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef RL_TOOLS_TEST_DATA_PATH
#error "RL_TOOLS_TEST_DATA_PATH is required"
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME
#error "RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME is required"
#endif

namespace rlt = rl_tools;

#if defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_GENERATOR_ACTIVE_BACKEND)
using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using CASES = golden::Cases<T, TI>;

static const std::string SCENE_PATH = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/ProcTHOR-Train-1.glb";
static const std::string DEFAULT_OUTPUT_DIR = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/rendering_raytracing_golden/backend/" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME;

template <typename SPEC, bool T_CAMERA_MOTION = true>
bool run_case(DEVICE& device, const std::string& output_dir, const char* name, bool write_probes) {
    std::cout << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] rendering " << name << std::endl;

    golden::Rendered<T> rendered;
    if(!golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered)) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": failed to load scene: " << SCENE_PATH << std::endl;
        return false;
    }

    bool ok = true;
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(rendered.frame_buffer.size() != pixel_count) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": frame buffer has the wrong size" << std::endl;
        ok = false;
    }
    else if(std::all_of(rendered.frame_buffer.begin(), rendered.frame_buffer.end(), [&](uint32_t pixel){ return pixel == rendered.frame_buffer[0]; })) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": frame buffer is constant" << std::endl;
        ok = false;
    }
    if constexpr(SPEC::HAS_DEPTH) {
        if(rendered.depth_buffer.size() != pixel_count) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": depth buffer has the wrong size" << std::endl;
            ok = false;
        }
        else if(std::all_of(rendered.depth_buffer.begin(), rendered.depth_buffer.end(), [&](T depth){ return depth == rendered.depth_buffer[0]; })) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": depth buffer is constant" << std::endl;
            ok = false;
        }
    }
    if(write_probes && rendered.probes.size() != (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": probe buffer is unavailable" << std::endl;
        ok = false;
    }

    if(ok) {
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const std::string directory = output_dir + "/" + CASES::POSES[camera_i].id;
            std::filesystem::create_directories(directory);
            ok &= golden::write_camera_png(directory + "/" + std::string(name) + ".png", rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            if constexpr(SPEC::HAS_DEPTH) {
                ok &= golden::write_camera_depth_bin(directory + "/" + std::string(name) + "_depth.bin", rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            if(write_probes) {
                ok &= golden::write_camera_probes(directory + "/probes.bin", rendered.probes.data() + camera_i * SPEC::NUM_PROBES, SPEC::NUM_PROBES);
            }
        }
        if(!ok) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": failed to write outputs" << std::endl;
        }
    }

    return ok;
}

bool run_geometry_case(DEVICE& device, const std::string& output_dir) {
    std::cout << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] rendering geometry (normals + segmentation)" << std::endl;
    using SPEC = CASES::GEOMETRY;

    golden::Rendered<T> rendered;
    if(!golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered)) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: failed to load scene: " << SCENE_PATH << std::endl;
        return false;
    }

    bool ok = true;
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(rendered.normals.size() != pixel_count * 3) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: normals buffer has the wrong size" << std::endl;
        ok = false;
    }
    else if(rendered.segmentation.size() != pixel_count) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: segmentation buffer has the wrong size" << std::endl;
        ok = false;
    }
    else {
        size_t hits = 0;
        size_t invalid = 0;
        size_t invalid_ids = 0;
        size_t topology_mismatches = 0;
        for(size_t pixel_i = 0; pixel_i < pixel_count; pixel_i++) {
            const T x = rendered.normals[pixel_i * 3 + 0];
            const T y = rendered.normals[pixel_i * 3 + 1];
            const T z = rendered.normals[pixel_i * 3 + 2];
            const bool normals_miss = x == 0 && y == 0 && z == 0;
            const bool segmentation_miss = rendered.segmentation[pixel_i] == golden::SEGMENTATION_BACKGROUND_ID;
            topology_mismatches += normals_miss != segmentation_miss;
            if(normals_miss) {
                continue;
            }
            hits++;
            const T norm = std::sqrt(x * x + y * y + z * z);
            invalid += norm < (T)0.999 || norm > (T)1.001;
            invalid_ids += rendered.segmentation[pixel_i] >= rendered.num_instances;
        }
        if(invalid > 0) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: " << invalid << " normal(s) are neither unit-length nor the zero miss sentinel" << std::endl;
            ok = false;
        }
        if(invalid_ids > 0) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: " << invalid_ids << " segmentation id(s) exceed the scene instance count " << rendered.num_instances << std::endl;
            ok = false;
        }
        if(topology_mismatches > 0) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: " << topology_mismatches << " pixel(s) disagree between the normals and segmentation hit/miss topology" << std::endl;
            ok = false;
        }
        if(hits == 0) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: no hit pixels" << std::endl;
            ok = false;
        }
    }

    if(ok) {
        std::vector<uint32_t> encoded(pixel_count);
        golden::colorize_normals(rendered.normals.data(), pixel_count, encoded.data());
        std::vector<uint32_t> segmentation_encoded(pixel_count);
        golden::colorize_segmentation(rendered.segmentation.data(), pixel_count, segmentation_encoded.data());
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const std::string directory = output_dir + "/" + CASES::POSES[camera_i].id;
            std::filesystem::create_directories(directory);
            ok &= golden::write_camera_png(directory + "/normals.png", encoded.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            ok &= golden::write_multi_camera_uint32_bin(directory + "/segmentation.bin", rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            ok &= golden::write_camera_png(directory + "/segmentation.png", segmentation_encoded.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if(!ok) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] geometry: failed to write outputs" << std::endl;
        }
    }

    return ok;
}

int main(int argc, char** argv) {
    std::string output_dir = DEFAULT_OUTPUT_DIR;
    for(int arg_i = 1; arg_i < argc; arg_i++) {
        const std::string argument = argv[arg_i];
        if(argument == "--output-dir") {
            if(arg_i + 1 >= argc) {
                std::cerr << "--output-dir requires a path" << std::endl;
                return 2;
            }
            output_dir = argv[++arg_i];
        }
        else if(argument == "--help" || argument == "-h") {
            std::cout << "Usage: " << argv[0] << " [--output-dir path]" << std::endl;
            std::cout << "Default output: " << DEFAULT_OUTPUT_DIR << std::endl;
            return 0;
        }
        else {
            std::cerr << "unknown argument: " << argument << std::endl;
            return 2;
        }
    }
    if(output_dir.empty()) {
        std::cerr << "--output-dir must not be empty" << std::endl;
        return 2;
    }

    std::filesystem::create_directories(output_dir);

    DEVICE device;
    rlt::init(device);

    bool ok = true;
    ok &= run_case<CASES::LOW_RGB>(device, output_dir, "low_rgb", true);
    ok &= run_case<CASES::MEDIUM_RGB>(device, output_dir, "medium_rgb", false);
    ok &= run_case<CASES::HIGH_RGB>(device, output_dir, "high_rgb", false);
    ok &= run_case<CASES::VERY_HIGH_RGB>(device, output_dir, "very_high_rgb", false);
    ok &= run_case<CASES::HIGH_RGB_AA2>(device, output_dir, "high_rgb_aa2", false);
    ok &= run_case<CASES::HIGH_RGB_MB4>(device, output_dir, "high_rgb_mb4", false);
    ok &= run_case<CASES::HIGH_RGB_MB4_DYNAMIC>(device, output_dir, "high_rgb_mb4_dynamic", false);
    ok &= run_case<CASES::HIGH_RGB_MB4_DYNAMIC, false>(device, output_dir, "high_rgb_mb4_object", false);
    ok &= run_case<CASES::LOW_RGBD>(device, output_dir, "low_rgbd", false);
    ok &= run_case<CASES::HIGH_RGBD>(device, output_dir, "high_rgbd", false);
    ok &= run_geometry_case(device, output_dir);

    if(!ok) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] FAILED" << std::endl;
        return 1;
    }
    std::cout << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] done: " << output_dir << std::endl;
    return 0;
}
