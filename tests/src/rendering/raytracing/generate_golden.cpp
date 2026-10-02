#include <rl_tools/operations/cpu.h>
#ifdef RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include "golden_build.h"
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

#ifdef RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND
using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using CASES = golden::Cases<T, TI>;
using Rendered = golden::Rendered<T>;

static const std::string SCENE_PATH = golden::PROCTHOR_TRAIN_1_SCENE;
static const std::string DEFAULT_OUTPUT_DIR = golden::layout::procthor_directory(golden::layout::goldens_root(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)));
static constexpr const char* LOG_PREFIX = "[golden:" RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME "] ";
static constexpr size_t MIN_OVERLAY_PIXELS = 16; // every view must see its private overlay, the shared one shows in some view

// <output_dir>/<pose>/<state>/<output>/golden-vNN.png or data/golden-vNN.bin of the pinned version, directories created on demand
std::string golden_png(const std::string& output_dir, TI camera_i, const char* state, const std::string& output){
    const std::string path = golden::layout::golden_png(golden::layout::output_directory(golden::layout::join(output_dir, CASES::POSES[camera_i].id), state, output), golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    return path;
}

std::string golden_bin(const std::string& output_dir, TI camera_i, const char* state, const std::string& output){
    const std::string path = golden::layout::golden_bin(golden::layout::output_directory(golden::layout::join(output_dir, CASES::POSES[camera_i].id), state, output), golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    return path;
}

bool fail(const std::string& name, const std::string& message){
    std::cerr << LOG_PREFIX << name << ": " << message << std::endl;
    return false;
}

template <typename CASE>
bool write_rgb(const std::string& output_dir, const Rendered& rendered) {
    using SPEC = typename CASE::SPEC;
    const std::string output = CASES::rgb_output<SPEC>();
    const std::string name = std::string(CASE::STATE) + "/" + output;
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(rendered.frame_buffer.size() != pixel_count) {
        return fail(name, "frame buffer has the wrong size");
    }
    if(std::all_of(rendered.frame_buffer.begin(), rendered.frame_buffer.end(), [&](uint32_t pixel){ return pixel == rendered.frame_buffer[0]; })) {
        return fail(name, "frame buffer is constant");
    }
    bool ok = true;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        ok &= golden::write_camera_png(golden_png(output_dir, camera_i, CASE::STATE, output), rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
    }
    return ok || fail(name, "failed to write outputs");
}

// one camera of NUM_PROBES x 1 with two channels: distance, hit
template <typename CASE>
bool write_probes(const std::string& output_dir, const Rendered& rendered) {
    using SPEC = typename CASE::SPEC;
    const std::string name = std::string(CASE::STATE) + "/probes";
    if(rendered.probes.size() != (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES) {
        return fail(name, "probe buffer is unavailable");
    }
    bool ok = true;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        std::vector<float> probes(SPEC::NUM_PROBES * 2);
        for(TI probe_i = 0; probe_i < SPEC::NUM_PROBES; probe_i++) {
            const auto& probe = rendered.probes[camera_i * SPEC::NUM_PROBES + probe_i];
            probes[probe_i * 2 + 0] = probe.distance;
            probes[probe_i * 2 + 1] = probe.hit != 0 ? 1.f : 0.f;
        }
        ok &= golden::write_multi_camera_float_bin(golden_bin(output_dir, camera_i, CASE::STATE, "probes"), probes.data(), 1, SPEC::NUM_PROBES, 1, 2);
    }
    return ok || fail(name, "failed to write outputs");
}

template <typename CASE>
bool write_depth(const std::string& output_dir, const Rendered& rendered) {
    using SPEC = typename CASE::SPEC;
    const std::string name = std::string(CASE::STATE) + "/depth";
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(rendered.depth_buffer.size() != pixel_count) {
        return fail(name, "depth buffer has the wrong size");
    }
    if(std::all_of(rendered.depth_buffer.begin(), rendered.depth_buffer.end(), [&](T depth){ return depth == rendered.depth_buffer[0]; })) {
        return fail(name, "depth buffer is constant");
    }
    bool ok = true;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        const float* depth = rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS;
        ok &= golden::write_multi_camera_float_bin(golden_bin(output_dir, camera_i, CASE::STATE, "depth"), depth, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        ok &= golden::write_camera_depth_png(golden_png(output_dir, camera_i, CASE::STATE, "depth"), depth, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
    }
    return ok || fail(name, "failed to write outputs");
}

// normals and segmentation come from one render; the invariants the corpus pins: unit normals
// or the zero miss sentinel, the same hit/miss topology in both outputs, ids inside the scene +
// overlay id range, and with overlays every view sees its private overlay and no other view's
template <typename CASE>
bool write_geometry(const std::string& output_dir, const Rendered& rendered) {
    using SPEC = typename CASE::SPEC;
    static_assert(SPEC::HAS_NORMALS && SPEC::HAS_SEGMENTATION, "the geometry case files both outputs");
    const std::string name = std::string(CASE::STATE) + "/geometry";
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(rendered.normals.size() != pixel_count * 3) {
        return fail(name, "normals buffer has the wrong size");
    }
    if(rendered.segmentation.size() != pixel_count) {
        return fail(name, "segmentation buffer has the wrong size");
    }
    const size_t id_limit = rendered.num_instances + (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
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
        invalid_ids += rendered.segmentation[pixel_i] >= id_limit;
    }
    bool ok = true;
    if(invalid > 0) {
        ok = fail(name, std::to_string(invalid) + " normal(s) are neither unit-length nor the zero miss sentinel");
    }
    if(invalid_ids > 0) {
        ok = fail(name, std::to_string(invalid_ids) + " segmentation id(s) exceed the id range " + std::to_string(id_limit));
    }
    if(topology_mismatches > 0) {
        ok = fail(name, std::to_string(topology_mismatches) + " pixel(s) disagree between the normals and segmentation hit/miss topology");
    }
    if(hits == 0) {
        ok = fail(name, "no hit pixels");
    }
    if constexpr(SPEC::ENABLE_OVERLAYS) {
        const auto overlay_id = [&](TI overlay){ return (uint32_t)(rendered.num_instances + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES); };
        size_t shared_pixels = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const uint32_t* segmentation = rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS;
            size_t private_pixels = 0;
            size_t foreign_pixels = 0;
            for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++) {
                shared_pixels += segmentation[pixel_i] == overlay_id(0);
                private_pixels += segmentation[pixel_i] == overlay_id(1 + camera_i);
                foreign_pixels += segmentation[pixel_i] > overlay_id(0) && segmentation[pixel_i] != overlay_id(1 + camera_i) && segmentation[pixel_i] != golden::SEGMENTATION_BACKGROUND_ID;
            }
            if(private_pixels < MIN_OVERLAY_PIXELS) {
                ok = fail(name, std::string("pose ") + CASES::POSES[camera_i].id + " sees only " + std::to_string(private_pixels) + " pixel(s) of its private overlay");
            }
            if(foreign_pixels > 0) {
                ok = fail(name, std::string("pose ") + CASES::POSES[camera_i].id + " sees " + std::to_string(foreign_pixels) + " pixel(s) of another view's private overlay");
            }
        }
        if(shared_pixels < MIN_OVERLAY_PIXELS) {
            ok = fail(name, "the shared overlay is not visible");
        }
    }
    if(!ok) {
        return false;
    }
    std::vector<uint32_t> encoded(pixel_count);
    golden::colorize_normals(rendered.normals.data(), pixel_count, encoded.data());
    std::vector<uint32_t> segmentation_encoded(pixel_count);
    golden::colorize_segmentation(rendered.segmentation.data(), pixel_count, segmentation_encoded.data());
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        ok &= golden::write_camera_png(golden_png(output_dir, camera_i, CASE::STATE, "normals"), encoded.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        ok &= golden::write_multi_camera_uint32_bin(golden_bin(output_dir, camera_i, CASE::STATE, "segmentation"), rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        ok &= golden::write_camera_png(golden_png(output_dir, camera_i, CASE::STATE, "segmentation"), segmentation_encoded.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
    }
    return ok || fail(name, "failed to write outputs");
}

// one flow direction: the flow values, their status buffer, and the invariants the corpus pins
// (finite and bounded, legal status codes, MISS and BEHIND carry the exact zero, every view has
// VALID nonzero flow, which is how the motion of every state stays visible in every view)
struct FlowDirection {
    const char* suffix;          // appended to the output: "" (backward) or "-forward"
    const std::vector<T>* flow;
    const std::vector<uint32_t>* status;
};

template <typename CASE>
bool validate_flow_direction(const std::string& name, const FlowDirection& direction) {
    using SPEC = typename CASE::SPEC;
    using FlowStatus = rlt::rendering::raytracing::FlowStatus;
    const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
    if(direction.flow->size() != pixel_count * 2 || direction.status->size() != pixel_count) {
        return fail(name, "flow or status buffer has the wrong size");
    }
    bool ok = true;
    size_t invalid = 0;
    size_t illegal_status = 0;
    size_t nonzero_sentinel = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        size_t nonzero = 0;
        for(size_t pixel_i = camera_i * (size_t)SPEC::CAM_PIXELS; pixel_i < (camera_i + 1) * (size_t)SPEC::CAM_PIXELS; pixel_i++) {
            const T u = (*direction.flow)[pixel_i * 2 + 0];
            const T v = (*direction.flow)[pixel_i * 2 + 1];
            const uint32_t status = (*direction.status)[pixel_i];
            invalid += !std::isfinite(u) || !std::isfinite(v)
                || std::fabs(u) > (T)SPEC::CAM_WIDTH || std::fabs(v) > (T)SPEC::CAM_HEIGHT;
            illegal_status += status > FlowStatus::OCCLUDED;
            nonzero_sentinel += (status == FlowStatus::MISS || status == FlowStatus::BEHIND) && (u != 0 || v != 0);
            nonzero += status == FlowStatus::VALID && (u != 0 || v != 0);
        }
        if(nonzero == 0) {
            ok = fail(name, std::string("pose ") + CASES::POSES[camera_i].id + " has no VALID pixel with nonzero flow (missing camera or object motion?)");
        }
    }
    if(invalid > 0) {
        ok = fail(name, std::to_string(invalid) + " non-finite or unbounded flow pixel(s)");
    }
    if(illegal_status > 0) {
        ok = fail(name, std::to_string(illegal_status) + " pixel(s) with an unknown flow status");
    }
    if(nonzero_sentinel > 0) {
        ok = fail(name, std::to_string(nonzero_sentinel) + " MISS/BEHIND pixel(s) carry nonzero flow");
    }
    return ok;
}

template <typename CASE>
bool write_flow_direction(const std::string& output_dir, const std::string& output, const FlowDirection& direction) {
    using SPEC = typename CASE::SPEC;
    bool ok = true;
    const std::string status_output = output + "-status";
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
        const float* camera_flow = direction.flow->data() + (size_t)camera_i * SPEC::CAM_PIXELS * 2;
        const uint32_t* camera_status = direction.status->data() + (size_t)camera_i * SPEC::CAM_PIXELS;
        ok &= golden::write_multi_camera_float_bin(golden_bin(output_dir, camera_i, CASE::STATE, output), camera_flow, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 2);
        ok &= golden::write_multi_camera_uint32_bin(golden_bin(output_dir, camera_i, CASE::STATE, status_output), camera_status, 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        // review images: the flow one normalises per image (advisory), the status one is the pinned false color
        rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(camera_flow, golden_png(output_dir, camera_i, CASE::STATE, output).c_str());
        std::vector<uint32_t> status_encoded(SPEC::CAM_PIXELS);
        golden::colorize_flow_status(camera_status, SPEC::CAM_PIXELS, status_encoded.data());
        ok &= golden::write_camera_png(golden_png(output_dir, camera_i, CASE::STATE, status_output), status_encoded.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
    }
    return ok;
}

template <typename CASE>
bool write_flow(const std::string& output_dir, const Rendered& rendered) {
    using SPEC = typename CASE::SPEC;
    static_assert(SPEC::HAS_FLOW && SPEC::HAS_FLOW_FORWARD, "the flow goldens cover both directions");
    const std::string name = std::string(CASE::STATE) + "/flow";
    const FlowDirection backward{"", &rendered.flow, &rendered.flow_status};
    const FlowDirection forward{"-forward", &rendered.flow_forward, &rendered.flow_forward_status};
    bool ok = validate_flow_direction<CASE>(name, backward);
    ok = validate_flow_direction<CASE>(name + forward.suffix, forward) && ok;
    if(!ok) {
        return false;
    }
    ok = write_flow_direction<CASE>(output_dir, "flow", backward) && write_flow_direction<CASE>(output_dir, std::string("flow") + forward.suffix, forward);
    return ok || fail(name, "failed to write outputs");
}

template <typename CASE>
bool run_case(DEVICE& device, const std::string& output_dir) {
    using SPEC = typename CASE::SPEC;
    const std::string name = CASES::name<CASE>();
    std::cout << LOG_PREFIX << "rendering " << name << std::endl;
    Rendered rendered;
    if(!golden::render_case<CASE, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered)) {
        return fail(name, "failed to load scene: " + SCENE_PATH);
    }
    bool ok = true;
    if constexpr(SPEC::HAS_RGB) {
        ok = write_rgb<CASE>(output_dir, rendered) && ok;
    }
    if constexpr(CASE::PROBES) {
        ok = write_probes<CASE>(output_dir, rendered) && ok;
    }
    if constexpr(SPEC::HAS_DEPTH) {
        ok = write_depth<CASE>(output_dir, rendered) && ok;
    }
    if constexpr(SPEC::HAS_NORMALS) {
        ok = write_geometry<CASE>(output_dir, rendered) && ok;
    }
    if constexpr(SPEC::HAS_FLOW) {
        ok = write_flow<CASE>(output_dir, rendered) && ok;
    }
    return ok;
}

template <typename... CASE>
bool run_cases(CASES::List<CASE...>, DEVICE& device, const std::string& output_dir) {
    bool ok = true;
    ((ok = run_case<CASE>(device, output_dir) && ok), ...);
    return ok;
}

bool write_views(const std::string& output_dir) {
    bool ok = true;
    for(TI camera_i = 0; camera_i < CASES::NUM_CAMERAS; camera_i++) {
        const golden::Pose<T>& pose = CASES::POSES[camera_i];
        const std::string view_directory = golden::layout::join(output_dir, pose.id);
        std::filesystem::create_directories(view_directory);
        ok &= golden::write_view(golden::layout::view_path(view_directory), pose.position, pose.look_at, pose.up, CASES::FOV, (int)CASES::CAM_WIDTH, (int)CASES::CAM_HEIGHT);
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
    if(std::string(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME) != golden::REFERENCE_BACKEND
       && std::filesystem::weakly_canonical(output_dir) == std::filesystem::weakly_canonical(DEFAULT_OUTPUT_DIR)) {
        std::cerr << "Only the CPU reference generator may write the canonical corpus; use --output-dir for a candidate" << std::endl;
        return 2;
    }

    std::filesystem::create_directories(output_dir);
    std::filesystem::remove(golden::layout::manifest_path(output_dir));

    DEVICE device;
    rlt::init(device);

    bool ok = write_views(output_dir);
    ok = run_cases(CASES::ALL{}, device, output_dir) && ok;

    if(ok) {
        ok = golden::write_json(golden::layout::manifest_path(output_dir), golden::procthor_manifest(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME, golden::generation_environment()));
    }

    if(!ok) {
        std::cerr << LOG_PREFIX << "FAILED" << std::endl;
        return 1;
    }
    std::cout << LOG_PREFIX << "done: " << output_dir << std::endl;
    return 0;
}
