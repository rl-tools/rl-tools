// Renders the golden cases (tests/src/rendering/raytracing/golden_cases.h) and compares against
// the OptiX golden renderings in tests/data/rendering_raytracing_golden (see AGENTS.md
// "Raytracing Golden Renderings"). Two targets are compiled from this file:
// - test_rendering_raytracing_golden_comparison_cpu: pinned to the generic (CPU) backend (included
//   directly, not via the mux) so the CPU raytracer is tested in every build configuration.
// - test_rendering_raytracing_golden_comparison_<metal|optix|vulkan>: uses the mux, i.e. the
//   backend the build is configured for (target name matches, e.g. _metal on macOS).
#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/raytracing/save_cpu.h>

#if !defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_CPU
#define RL_TOOLS_GOLDEN_BACKEND_NAME "generic"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_METAL
#define RL_TOOLS_GOLDEN_BACKEND_NAME "metal"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_OPTIX
#define RL_TOOLS_GOLDEN_BACKEND_NAME "optix"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_VULKAN
#define RL_TOOLS_GOLDEN_BACKEND_NAME "vulkan"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_WEBGPU
#define RL_TOOLS_GOLDEN_BACKEND_NAME "webgpu"
#else
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_GENERIC
#define RL_TOOLS_GOLDEN_BACKEND_NAME "generic"
#endif

#include "golden_cases.h"
#include "golden_io.h"
#include "golden_manifest.h"
#include "golden_render.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// falls back to the CWD-relative convention (run from the repo root) like other targets
#ifdef RL_TOOLS_TEST_DATA_PATH
#define RL_TOOLS_GOLDEN_TEST_DATA_PATH RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)
#else
#define RL_TOOLS_GOLDEN_TEST_DATA_PATH "tests/data"
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

// The reference backend (OptiX) rendered the corpus and reproduces it bit-exactly, so every metric
// is held to zero for it: a nonzero deviation is a behavioral change that needs a GOLDEN_LEDGER.md
// row. The other backends get bounds at roughly three times the worst floor measured against
// corpus v2 on generic, Vulkan (NVIDIA and lavapipe) and WebGPU: RGB MAD 0.16, RGB outliers 0.31%,
// normals MAD 0.0013, flow status 0.0015%, probe distances 5.3e-7, everything else zero. Silhouette
// pixels landing on different surfaces across backends make the outlier fractions necessary next
// to the means, and the means alone would hide a localized shading change (one re-materialised
// object covers ~3% of a pose).
static constexpr bool REFERENCE_BACKEND = std::string_view(RL_TOOLS_GOLDEN_BACKEND_NAME) == golden::REFERENCE_BACKEND;
static constexpr double RGB_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.4;              // mean abs diff in 8-bit levels
static constexpr int RGB_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double RGB_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.01;          // pixels with a channel delta above RGB_OUTLIER_CHANNEL_DELTA
static constexpr double DEPTH_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 1e-5;           // mean abs diff normalized by max_depth
static constexpr double DEPTH_OUTLIER_REL = 1e-3;                                      // per-pixel relative outlier threshold
static constexpr double DEPTH_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.001;       // max fraction of outlier pixels
static constexpr int PROBE_HIT_MISMATCH_MAX = REFERENCE_BACKEND ? 0 : 2;              // out of NUM_CAMERAS * NUM_PROBES = 768
static constexpr double PROBE_DISTANCE_REL = REFERENCE_BACKEND ? 0 : 1e-5;
static constexpr double NORMALS_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.1;          // encoded normals compare like RGB
static constexpr int NORMALS_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double NORMALS_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.002;
static constexpr double FLOW_MEAN_EPE_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.01;       // px, over pixels VALID in both target and current
static constexpr double FLOW_OUTLIER_EPE = 0.5;                                        // px
static constexpr double FLOW_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.001;
static constexpr double FLOW_STATUS_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.001;
static constexpr double SEGMENTATION_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.002;
static constexpr double SEGMENTATION_BACKGROUND_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.001;

static const std::string SCENE_PATH = RL_TOOLS_GOLDEN_TEST_DATA_PATH "/ProcTHOR-Train-1.glb";
static const std::string GOLDEN_ROOT = RL_TOOLS_GOLDEN_TEST_DATA_PATH "/rendering_raytracing_golden";
static const std::string GOLDEN_DIR = golden::layout::procthor_static_scene_directory(GOLDEN_ROOT);
static const std::string MANIFEST_PATH = GOLDEN_DIR + "/manifest.json";

#ifndef RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT
#define RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT "build/raytracing_golden_artifacts"
#endif

static const std::string BACKEND_OUTPUT_DIR = std::string(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT)
                                                    + "/" + RL_TOOLS_GOLDEN_BACKEND_NAME + "/procthor_static_scene";

namespace {
    bool goldens_available(){
        return std::filesystem::exists(MANIFEST_PATH);
    }

    struct RGBStats{
        double mad = 0;
        double outlier_fraction = 0;
        int channel_max[3] = {0, 0, 0};
    };

    RGBStats compare_rgb(const uint32_t* ours, const uint32_t* golden, size_t count){
        RGBStats stats;
        const auto* ours_bytes = (const unsigned char*)ours;
        const auto* golden_bytes = (const unsigned char*)golden;
        double total = 0;
        size_t outliers = 0;
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            bool outlier = false;
            for(int channel = 0; channel < 3; channel++){
                const int diff = (int)ours_bytes[pixel_i * 4 + channel] - (int)golden_bytes[pixel_i * 4 + channel];
                const int abs_diff = diff < 0 ? -diff : diff;
                if(abs_diff > stats.channel_max[channel]) stats.channel_max[channel] = abs_diff;
                total += abs_diff;
                outlier = outlier || abs_diff > RGB_OUTLIER_CHANNEL_DELTA;
            }
            outliers += outlier;
        }
        stats.mad = total / ((double)count * 3);
        stats.outlier_fraction = (double)outliers / (double)count;
        return stats;
    }

    template <typename SPEC>
    void write_backend_frames(const char* name, const Rendered& rendered){
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const std::string directory = BACKEND_OUTPUT_DIR + "/" + CASES::POSES[camera_i].id;
            std::filesystem::create_directories(directory);
            golden::write_camera_png(directory + "/" + name + "_current.png", rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            if(!rendered.depth_buffer.empty()){
                golden::write_camera_depth_png(directory + "/" + name + "_depth_current.png", rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            }
        }
    }

    template <typename SPEC>
    void expect_rgb_matches_golden(const char* name, const Rendered& rendered){
        double worst_mad = 0;
        const char* worst_id = CASES::POSES[0].id;
        int max_channel[3] = {0, 0, 0};
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(GOLDEN_DIR + "/" + id + "/" + name + ".png", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/" << name;
            const RGBStats stats = compare_rgb(rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, golden_pixels.data(), SPEC::CAM_PIXELS);
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                golden::write_camera_png(directory + "/" + name + "_target.png", golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                golden::write_camera_diff_png(directory + "/" + name + "_diff.png", rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            EXPECT_LE(stats.mad, RGB_MAD_THRESHOLD)
                << name << " pose " << id << ": RGB mean abs diff " << stats.mad
                << " (max R=" << stats.channel_max[0] << " G=" << stats.channel_max[1] << " B=" << stats.channel_max[2] << ")";
            EXPECT_LE(stats.outlier_fraction, RGB_OUTLIER_FRACTION) << name << " pose " << id << ": RGB outlier fraction " << stats.outlier_fraction;
            if(stats.mad > worst_mad){
                worst_mad = stats.mad;
                worst_id = id;
            }
            for(int channel = 0; channel < 3; channel++){
                if(stats.channel_max[channel] > max_channel[channel]) max_channel[channel] = stats.channel_max[channel];
            }
        }
        std::printf("[golden] %s: worst pose %s rgb mad=%.4f max=(%d,%d,%d) over %d poses\n", name, worst_id, worst_mad, max_channel[0], max_channel[1], max_channel[2], (int)SPEC::NUM_CAMERAS);
    }

    template <typename SPEC, bool T_CAMERA_MOTION = true>
    void run_rgb_case(const char* name){
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        write_backend_frames<SPEC>(name, rendered);
        expect_rgb_matches_golden<SPEC>(name, rendered);
    }

    template <typename SPEC>
    void run_rgbd_case(const char* name){
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        write_backend_frames<SPEC>(name, rendered);
        expect_rgb_matches_golden<SPEC>(name, rendered);

        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_depth;
            ASSERT_TRUE(golden::load_camera_depth_bin(GOLDEN_DIR + "/" + id + "/" + name + "_depth.bin", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_depth)) << "failed to load golden depth: " << id << "/" << name;
            const float* ours = rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                golden::write_camera_depth_png(directory + "/" + name + "_depth_target.png", golden_depth.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
                golden::write_camera_depth_diff_png(directory + "/" + name + "_depth_diff.png", golden_depth.data(), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            }
            double total_abs_diff = 0;
            size_t outliers = 0;
            for(size_t pixel_i = 0; pixel_i < golden_depth.size(); pixel_i++){
                const double diff = (double)ours[pixel_i] - (double)golden_depth[pixel_i];
                const double abs_diff = diff < 0 ? -diff : diff;
                total_abs_diff += abs_diff;
                const double magnitude = std::max((double)golden_depth[pixel_i], 1e-6);
                if(abs_diff / magnitude > DEPTH_OUTLIER_REL) outliers++;
            }
            const double depth_mad = total_abs_diff / (double)golden_depth.size() / (double)rendered.max_depth;
            const double outlier_fraction = (double)outliers / (double)golden_depth.size();
            EXPECT_LE(depth_mad, DEPTH_MAD_THRESHOLD) << name << " pose " << id << ": normalized depth mean abs diff " << depth_mad;
            EXPECT_LE(outlier_fraction, DEPTH_OUTLIER_FRACTION) << name << " pose " << id << ": depth outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, depth_mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] %s: worst depth mad=%.2e worst outliers=%.4f%%\n", name, worst_mad, worst_outliers * 100);
    }

    void run_normals_case(){
        using SPEC = CASES::GEOMETRY;
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;

        const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::vector<uint32_t> encoded(pixel_count);
        golden::colorize_normals(rendered.normals.data(), pixel_count, encoded.data());

        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(GOLDEN_DIR + "/" + id + "/normals.png", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/normals";
            const uint32_t* ours = encoded.data() + camera_i * SPEC::CAM_PIXELS;
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                golden::write_camera_png(directory + "/normals_current.png", ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                golden::write_camera_png(directory + "/normals_target.png", golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                golden::write_camera_diff_png(directory + "/normals_diff.png", ours, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            double total = 0;
            size_t outlier_pixels = 0;
            const auto* ours_bytes = reinterpret_cast<const uint8_t*>(ours);
            const auto* golden_bytes = reinterpret_cast<const uint8_t*>(golden_pixels.data());
            for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                bool outlier = false;
                for(size_t channel = 0; channel < 3; channel++){
                    const int difference = (int)ours_bytes[pixel_i * 4 + channel] - (int)golden_bytes[pixel_i * 4 + channel];
                    const int absolute = difference < 0 ? -difference : difference;
                    total += absolute;
                    outlier = outlier || absolute > NORMALS_OUTLIER_CHANNEL_DELTA;
                }
                outlier_pixels += outlier;
            }
            const double mad = total / ((double)SPEC::CAM_PIXELS * 3);
            const double outlier_fraction = (double)outlier_pixels / (double)SPEC::CAM_PIXELS;
            EXPECT_LE(mad, NORMALS_MAD_THRESHOLD) << "normals pose " << id << ": encoded mean abs diff " << mad;
            EXPECT_LE(outlier_fraction, NORMALS_OUTLIER_FRACTION) << "normals pose " << id << ": outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] normals: worst mad=%.4f worst outliers=%.4f%% over %d poses\n", worst_mad, worst_outliers * 100, (int)SPEC::NUM_CAMERAS);
    }

    struct FlowDirectionView{
        const char* suffix; // "" (backward, current camera grid) or "_forward" (previous camera grid)
        const std::vector<float>* flow;
        const std::vector<uint32_t>* status;
    };

    template <typename SPEC>
    void expect_flow_direction_matches_golden(const char* name, const FlowDirectionView& direction){
        using FlowStatus = rlt::rendering::raytracing::FlowStatus;
        const std::string base = std::string(name) + direction.suffix;
        ASSERT_EQ(direction.flow->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS * 2) << base;
        ASSERT_EQ(direction.status->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS) << base;
        double worst_epe = 0;
        double worst_outliers = 0;
        double worst_status_mismatch = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_flow;
            std::vector<uint32_t> golden_status;
            ASSERT_TRUE(golden::load_multi_camera_float_bin(GOLDEN_DIR + "/" + id + "/" + base + ".bin", 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_flow, 2)) << "failed to load golden: " << id << "/" << base;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(GOLDEN_DIR + "/" + id + "/" + base + "_status.bin", 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_status)) << "failed to load golden: " << id << "/" << base << "_status";
            const float* ours = direction.flow->data() + (size_t)camera_i * SPEC::CAM_PIXELS * 2;
            const uint32_t* ours_status = direction.status->data() + (size_t)camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: exactly the pinned false color of the status .bin
                std::vector<uint32_t> golden_png;
                ASSERT_TRUE(golden::load_camera_png(GOLDEN_DIR + "/" + id + "/" + base + "_status.png", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_png)) << "failed to load golden: " << id << "/" << base << "_status.png";
                std::vector<uint32_t> expected(golden_status.size());
                golden::colorize_flow_status(golden_status.data(), golden_status.size(), expected.data());
                EXPECT_EQ(golden_png, expected) << base << " pose " << id << ": review PNG does not match " << base << "_status.bin";
            }
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(ours, (directory + "/" + base + "_current.png").c_str());
                rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(golden_flow.data(), (directory + "/" + base + "_target.png").c_str());
                std::vector<float> difference(golden_flow.size());
                for(size_t value_i = 0; value_i < difference.size(); value_i++){
                    difference[value_i] = ours[value_i] - golden_flow[value_i];
                }
                rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(difference.data(), (directory + "/" + base + "_diff.png").c_str());
                std::vector<uint32_t> image(SPEC::CAM_PIXELS);
                golden::colorize_flow_status(golden_status.data(), SPEC::CAM_PIXELS, image.data());
                golden::write_camera_png(directory + "/" + base + "_status_target.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                golden::colorize_flow_status(ours_status, SPEC::CAM_PIXELS, image.data());
                golden::write_camera_png(directory + "/" + base + "_status_current.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                    image[pixel_i] = golden::flow_status_diff_color(golden_status[pixel_i], ours_status[pixel_i]);
                }
                golden::write_camera_png(directory + "/" + base + "_status_diff.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            double epe_total = 0;
            size_t compared = 0;
            size_t outliers = 0;
            size_t status_mismatches = 0;
            for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                status_mismatches += ours_status[pixel_i] != golden_status[pixel_i];
                if(ours_status[pixel_i] != FlowStatus::VALID || golden_status[pixel_i] != FlowStatus::VALID){
                    continue;
                }
                const double du = (double)ours[pixel_i * 2 + 0] - golden_flow[pixel_i * 2 + 0];
                const double dv = (double)ours[pixel_i * 2 + 1] - golden_flow[pixel_i * 2 + 1];
                const double endpoint_error = std::sqrt(du * du + dv * dv);
                epe_total += endpoint_error;
                compared++;
                outliers += endpoint_error > FLOW_OUTLIER_EPE;
            }
            const double mean_epe = epe_total / std::max((double)compared, 1.0);
            const double outlier_fraction = (double)outliers / std::max((double)compared, 1.0);
            const double status_mismatch_fraction = (double)status_mismatches / (double)SPEC::CAM_PIXELS;
            EXPECT_GT(compared, (size_t)0) << base << " pose " << id << ": no pixel is VALID in both target and current";
            EXPECT_LE(mean_epe, FLOW_MEAN_EPE_THRESHOLD) << base << " pose " << id << ": mean endpoint error " << mean_epe;
            EXPECT_LE(outlier_fraction, FLOW_OUTLIER_FRACTION) << base << " pose " << id << ": flow outlier fraction " << outlier_fraction;
            EXPECT_LE(status_mismatch_fraction, FLOW_STATUS_MISMATCH_FRACTION) << base << " pose " << id << ": flow status mismatch fraction " << status_mismatch_fraction;
            worst_epe = std::max(worst_epe, mean_epe);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
            worst_status_mismatch = std::max(worst_status_mismatch, status_mismatch_fraction);
        }
        std::printf("[golden] %s: worst mean epe=%.4f px worst outliers=%.4f%% worst status mismatches=%.4f%% over %d poses\n", base.c_str(), worst_epe, worst_outliers * 100, worst_status_mismatch * 100, (int)SPEC::NUM_CAMERAS);
    }

    template <typename SPEC, bool T_CAMERA_MOTION = true>
    void run_flow_case(const char* name){
        static_assert(SPEC::HAS_FLOW && SPEC::HAS_FLOW_FORWARD, "the flow goldens cover both directions");
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        expect_flow_direction_matches_golden<SPEC>(name, FlowDirectionView{"", &rendered.flow, &rendered.flow_status});
        expect_flow_direction_matches_golden<SPEC>(name, FlowDirectionView{"_forward", &rendered.flow_forward, &rendered.flow_forward_status});
    }

    void run_segmentation_case(){
        using SPEC = CASES::GEOMETRY;
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        ASSERT_EQ(rendered.segmentation.size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);

        double worst_mismatches = 0;
        double worst_background_mismatches = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<uint32_t> golden_segmentation;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(GOLDEN_DIR + "/" + id + "/segmentation.bin", 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_segmentation)) << "failed to load golden: " << id << "/segmentation.bin";
            const uint32_t* ours = rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: it must be exactly the false-color
                // encoding of segmentation.bin (mirrors the overlay corpus validation)
                std::vector<uint32_t> golden_png;
                ASSERT_TRUE(golden::load_camera_png(GOLDEN_DIR + "/" + id + "/segmentation.png", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_png)) << "failed to load golden: " << id << "/segmentation.png";
                std::vector<uint32_t> expected(golden_segmentation.size());
                golden::colorize_segmentation(golden_segmentation.data(), golden_segmentation.size(), expected.data());
                EXPECT_EQ(golden_png, expected) << "segmentation pose " << id << ": review PNG does not match segmentation.bin";
            }
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                std::vector<uint32_t> image(SPEC::CAM_PIXELS);
                golden::colorize_segmentation(ours, SPEC::CAM_PIXELS, image.data());
                golden::write_camera_png(directory + "/segmentation_current.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                golden::colorize_segmentation(golden_segmentation.data(), SPEC::CAM_PIXELS, image.data());
                golden::write_camera_png(directory + "/segmentation_target.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
                for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                    image[pixel_i] = golden::segmentation_diff_color(golden_segmentation[pixel_i], ours[pixel_i]);
                }
                golden::write_camera_png(directory + "/segmentation_diff.png", image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            size_t mismatches = 0;
            size_t background_mismatches = 0;
            for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                mismatches += ours[pixel_i] != golden_segmentation[pixel_i];
                background_mismatches += (ours[pixel_i] == golden::SEGMENTATION_BACKGROUND_ID) != (golden_segmentation[pixel_i] == golden::SEGMENTATION_BACKGROUND_ID);
            }
            const double mismatch_fraction = (double)mismatches / (double)SPEC::CAM_PIXELS;
            const double background_fraction = (double)background_mismatches / (double)SPEC::CAM_PIXELS;
            EXPECT_LE(mismatch_fraction, SEGMENTATION_MISMATCH_FRACTION) << "segmentation pose " << id << ": id mismatch fraction " << mismatch_fraction;
            EXPECT_LE(background_fraction, SEGMENTATION_BACKGROUND_MISMATCH_FRACTION) << "segmentation pose " << id << ": hit/miss mismatch fraction " << background_fraction;
            worst_mismatches = std::max(worst_mismatches, mismatch_fraction);
            worst_background_mismatches = std::max(worst_background_mismatches, background_fraction);
        }
        std::printf("[golden] segmentation: worst mismatches=%.4f%% worst hit/miss mismatches=%.4f%% over %d poses\n", worst_mismatches * 100, worst_background_mismatches * 100, (int)SPEC::NUM_CAMERAS);
    }
}

// a present corpus of another version fails: the fix is a data refresh (GOLDEN_LEDGER.md), not a code change
#define RL_TOOLS_GOLDEN_REQUIRE_VERSION() ASSERT_EQ(golden::manifest_version(MANIFEST_PATH), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION) << "golden corpus version mismatch at " << MANIFEST_PATH << " (see tests/src/rendering/raytracing/GOLDEN_LEDGER.md, run tests/src/rendering/raytracing/download_data.sh)"
#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() ASSERT_TRUE(goldens_available()) << "required golden renderings not found at " << GOLDEN_DIR; RL_TOOLS_GOLDEN_REQUIRE_VERSION()
#else
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() if(!goldens_available()){ GTEST_SKIP() << "golden renderings not found at " << GOLDEN_DIR << " (run tests/src/rendering/raytracing/download_data.sh or generate with test_rendering_raytracing_generate_golden_optix --output-dir " << GOLDEN_DIR << ")"; } RL_TOOLS_GOLDEN_REQUIRE_VERSION()
#endif

TEST(RL_TOOLS_GOLDEN_SUITE, LOW_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::LOW_RGB>("low_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, MEDIUM_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::MEDIUM_RGB>("medium_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB>("high_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, VERY_HIGH_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::VERY_HIGH_RGB>("very_high_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_AA2){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_AA2>("high_rgb_aa2");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_MB4){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4>("high_rgb_mb4");
}
// dynamic-object motion blur: moving camera + spinning/translating overlay
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_MB4_DYNAMIC){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4_DYNAMIC>("high_rgb_mb4_dynamic");
}
// object-only isolation: static camera (shutter open == close), the blur is purely the overlay's
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_MB4_OBJECT){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4_DYNAMIC, false>("high_rgb_mb4_object");
}
TEST(RL_TOOLS_GOLDEN_SUITE, LOW_RGBD){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgbd_case<CASES::LOW_RGBD>("low_rgbd");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGBD){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgbd_case<CASES::HIGH_RGBD>("high_rgbd");
}

TEST(RL_TOOLS_GOLDEN_SUITE, NORMALS){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_normals_case();
}

TEST(RL_TOOLS_GOLDEN_SUITE, FLOW){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_flow_case<CASES::FLOW>("flow");
}

// object-only flow: static camera, the overlay moves from its previous (open) to its current
// (close) pose, so the flow, the disocclusion band and the forward direction are purely the object's
TEST(RL_TOOLS_GOLDEN_SUITE, FLOW_DYNAMIC){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_flow_case<CASES::FLOW_DYNAMIC, false>("flow_dynamic");
}

TEST(RL_TOOLS_GOLDEN_SUITE, SEGMENTATION){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_segmentation_case();
}

TEST(RL_TOOLS_GOLDEN_SUITE, PROBES){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    using SPEC = CASES::LOW_RGB; // probes.bin is written during the low_rgb golden case
    DEVICE device;
    rlt::init(device);
    Rendered rendered;
    ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
    ASSERT_FALSE(rendered.probes.empty());

    int hit_mismatches = 0;
    double max_rel_distance_diff = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const char* id = CASES::POSES[camera_i].id;
        std::vector<rlt::rendering::raytracing::CollisionResult> golden_probes;
        ASSERT_TRUE(golden::load_camera_probes(GOLDEN_DIR + "/" + id + "/probes.bin", SPEC::NUM_PROBES, golden_probes)) << "failed to load golden probes: " << id;
        const auto* ours = rendered.probes.data() + camera_i * SPEC::NUM_PROBES;
        for(size_t probe_i = 0; probe_i < golden_probes.size(); probe_i++){
            if((ours[probe_i].hit != 0) != (golden_probes[probe_i].hit != 0)){
                hit_mismatches++;
                continue;
            }
            if(golden_probes[probe_i].hit){
                const double diff = (double)ours[probe_i].distance - (double)golden_probes[probe_i].distance;
                const double abs_diff = diff < 0 ? -diff : diff;
                const double rel = abs_diff / std::max((double)golden_probes[probe_i].distance, 1e-6);
                if(rel > max_rel_distance_diff) max_rel_distance_diff = rel;
            }
        }
    }
    EXPECT_LE(hit_mismatches, PROBE_HIT_MISMATCH_MAX) << "probe hit flag mismatches: " << hit_mismatches << "/" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    EXPECT_LE(max_rel_distance_diff, PROBE_DISTANCE_REL) << "max relative probe distance diff: " << max_rel_distance_diff;
    std::printf("[golden] probes: hit mismatches=%d/%d max rel dist diff=%.2e\n", hit_mismatches, (int)(SPEC::NUM_CAMERAS * SPEC::NUM_PROBES), max_rel_distance_diff);
}
