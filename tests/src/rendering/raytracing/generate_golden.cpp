#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_GENERATOR_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include "golden_io.h"
#include "golden_manifest.h"
#include "golden_render.h"
#include "../../utils/utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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

// one flow direction: the flow values, their status buffer, and the invariants the corpus pins
// (finite and bounded, legal status codes, MISS and BEHIND carry the exact zero, some VALID flow)
struct FlowDirection {
    const char* suffix;          // appended to the case name: "" (backward) or "_forward"
    const std::vector<T>* flow;
    const std::vector<uint32_t>* status;
};

template <typename SPEC>
bool validate_flow_direction(const char* name, const FlowDirection& direction) {
    using FlowStatus = rlt::rendering::raytracing::FlowStatus;
    const std::string label = std::string(name) + direction.suffix;
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(direction.flow->size() != pixel_count * 2 || direction.status->size() != pixel_count) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << label << ": flow or status buffer has the wrong size" << std::endl;
        return false;
    }
    size_t valid = 0;
    size_t nonzero = 0;
    size_t invalid = 0;
    size_t illegal_status = 0;
    size_t nonzero_sentinel = 0;
    for(size_t pixel_i = 0; pixel_i < pixel_count; pixel_i++) {
        const T u = (*direction.flow)[pixel_i * 2 + 0];
        const T v = (*direction.flow)[pixel_i * 2 + 1];
        const uint32_t status = (*direction.status)[pixel_i];
        invalid += !std::isfinite(u) || !std::isfinite(v)
            || std::fabs(u) > (T)SPEC::CAM_WIDTH || std::fabs(v) > (T)SPEC::CAM_HEIGHT;
        illegal_status += status > FlowStatus::OCCLUDED;
        nonzero_sentinel += (status == FlowStatus::MISS || status == FlowStatus::BEHIND) && (u != 0 || v != 0);
        valid += status == FlowStatus::VALID;
        nonzero += status == FlowStatus::VALID && (u != 0 || v != 0);
    }
    bool ok = true;
    if(invalid > 0) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << label << ": " << invalid << " non-finite or unbounded flow pixel(s)" << std::endl;
        ok = false;
    }
    if(illegal_status > 0) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << label << ": " << illegal_status << " pixel(s) with an unknown flow status" << std::endl;
        ok = false;
    }
    if(nonzero_sentinel > 0) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << label << ": " << nonzero_sentinel << " MISS/BEHIND pixel(s) carry nonzero flow" << std::endl;
        ok = false;
    }
    if(valid == 0 || nonzero == 0) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << label << ": no VALID pixel with nonzero flow (missing camera or object motion?)" << std::endl;
        ok = false;
    }
    return ok;
}

template <typename SPEC>
bool write_flow_direction(const std::string& output_dir, const char* name, const FlowDirection& direction) {
    bool ok = true;
    const std::string base = std::string(name) + direction.suffix;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        const std::string directory = output_dir + "/" + CASES::POSES[camera_i].id;
        std::filesystem::create_directories(directory);
        const float* camera_flow = direction.flow->data() + (size_t)camera_i * SPEC::CAM_PIXELS * 2;
        const uint32_t* camera_status = direction.status->data() + (size_t)camera_i * SPEC::CAM_PIXELS;
        ok &= golden::write_multi_camera_float_bin(directory + "/" + base + ".bin", camera_flow, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 2);
        ok &= golden::write_multi_camera_uint32_bin(directory + "/" + base + "_status.bin", camera_status, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        // review images: flow.png normalises per image (advisory), the status PNG is the pinned false color
        rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(camera_flow, (directory + "/" + base + ".png").c_str());
        std::vector<uint32_t> status_encoded(SPEC::CAM_PIXELS);
        golden::colorize_flow_status(camera_status, SPEC::CAM_PIXELS, status_encoded.data());
        ok &= golden::write_camera_png(directory + "/" + base + "_status.png", status_encoded.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
    }
    return ok;
}

template <typename SPEC, bool T_CAMERA_MOTION = true>
bool run_flow_case(DEVICE& device, const std::string& output_dir, const char* name) {
    std::cout << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] rendering " << name << std::endl;
    static_assert(SPEC::HAS_FLOW && SPEC::HAS_FLOW_FORWARD, "the flow goldens cover both directions");

    golden::Rendered<T> rendered;
    if(!golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered)) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": failed to load scene: " << SCENE_PATH << std::endl;
        return false;
    }

    const FlowDirection backward{"", &rendered.flow, &rendered.flow_status};
    const FlowDirection forward{"_forward", &rendered.flow_forward, &rendered.flow_forward_status};
    bool ok = validate_flow_direction<SPEC>(name, backward);
    ok = validate_flow_direction<SPEC>(name, forward) && ok;
    if(ok) {
        ok = write_flow_direction<SPEC>(output_dir, name, backward) && write_flow_direction<SPEC>(output_dir, name, forward);
        if(!ok) {
            std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] " << name << ": failed to write outputs" << std::endl;
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
    ok &= run_flow_case<CASES::FLOW>(device, output_dir, "flow");
    ok &= run_flow_case<CASES::FLOW_DYNAMIC, false>(device, output_dir, "flow_dynamic");

    if(ok) {
        ok = golden::write_manifest(output_dir + "/manifest.json", golden::procthor_manifest(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME));
    }

    if(!ok) {
        std::cerr << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] FAILED" << std::endl;
        return 1;
    }
    std::cout << "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] done: " << output_dir << std::endl;
    return 0;
}
