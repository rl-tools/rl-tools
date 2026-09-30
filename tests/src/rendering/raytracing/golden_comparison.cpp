// Renders the golden cases (tests/src/rendering/raytracing/golden_cases.h) and compares against
// the OptiX golden renderings in tests/data/rendering/raytracing/goldens/procthor-train-1 (see
// AGENTS.md "Raytracing Golden Renderings" and GOLDEN_LEDGER.md). Two targets are compiled from
// this file:
// - test_rendering_raytracing_golden_comparison_cpu: pinned to the generic (CPU) backend (included
//   directly, not via the mux) so the CPU raytracer is tested in every build configuration.
// - test_rendering_raytracing_golden_comparison_<metal|optix|vulkan|webgpu>: uses the mux, i.e.
//   the backend the build is configured for (target name matches, e.g. _metal on macOS).
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
#include "golden_metrics.h"
#include "golden_render.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
// row. The other backends get bounds at about twice the worst floor measured against corpus v2
// on generic, Vulkan (NVIDIA and lavapipe) and WebGPU: RGB MAD 0.16, RGB outliers 0.31%, normals
// MAD 0.0013, probe distances 5.3e-7, flow status and normals outliers one pixel, everything else
// zero; fractions with a zero floor allow 0.05% (33 pixels of a 256x256 pose). The RGB outliers
// are silhouette pixels landing on different surfaces; the subset not explained by the other
// image's neighborhood (golden_metrics.h) is a surface shaded differently, floor two pixels per
// pose, while a re-materialised object covers ~3% of a pose.
static constexpr bool REFERENCE_BACKEND = std::string_view(RL_TOOLS_GOLDEN_BACKEND_NAME) == golden::REFERENCE_BACKEND;
static constexpr double RGB_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.35;             // mean abs diff in 8-bit levels
static constexpr int RGB_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double RGB_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.007;         // pixels with a channel delta above RGB_OUTLIER_CHANNEL_DELTA
static constexpr double RGB_SHADING_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0002; // outliers not explained by the neighborhood (13 pixels)
static constexpr double DEPTH_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 1e-7;           // mean abs diff normalized by max_depth
static constexpr double DEPTH_OUTLIER_REL = 1e-3;                                      // per-pixel relative outlier threshold
static constexpr double DEPTH_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;      // max fraction of outlier pixels
static constexpr int PROBE_HIT_MISMATCH_MAX = REFERENCE_BACKEND ? 0 : 1;              // out of NUM_CAMERAS * NUM_PROBES = 768
static constexpr double PROBE_DISTANCE_REL = REFERENCE_BACKEND ? 0 : 2e-6;
static constexpr double NORMALS_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.005;        // encoded normals compare like RGB
static constexpr int NORMALS_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double NORMALS_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double FLOW_MEAN_EPE_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.001;      // px, over pixels VALID in both target and current
static constexpr double FLOW_OUTLIER_EPE = 0.5;                                        // px
static constexpr double FLOW_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double FLOW_STATUS_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double SEGMENTATION_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double SEGMENTATION_BACKGROUND_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;

static const std::string SCENE_PATH = RL_TOOLS_GOLDEN_TEST_DATA_PATH "/ProcTHOR-Train-1.glb";
static const std::string GOLDEN_ROOT = golden::layout::goldens_root(RL_TOOLS_GOLDEN_TEST_DATA_PATH);
static const std::string CORPUS_DIRECTORY = golden::layout::procthor_directory(GOLDEN_ROOT);
static const std::string MANIFEST_PATH = golden::layout::manifest_path(CORPUS_DIRECTORY);

namespace {
    bool goldens_available(){
        return std::filesystem::exists(MANIFEST_PATH);
    }

    // Vulkan on lavapipe is named apart from Vulkan on a GPU because its floors differ
    std::string backend_tag(){
        const char* driver_files = std::getenv("VK_DRIVER_FILES");
        const bool lavapipe = driver_files != nullptr && std::string(driver_files).find("lvp") != std::string::npos;
        return std::string(RL_TOOLS_GOLDEN_BACKEND_NAME) + (lavapipe ? "-lavapipe" : "");
    }

    std::string golden_directory(TI camera_i, const std::string& modality){
        return golden::layout::modality_directory(golden::layout::procthor_view_directory(GOLDEN_ROOT, CASES::POSES[camera_i].id), modality);
    }

    // render-<backend>.png and versus-*.png live next to the goldens
    std::string review_directory(TI camera_i, const std::string& modality){
        const std::string directory = golden_directory(camera_i, modality);
        std::filesystem::create_directories(directory);
        return directory;
    }

    std::string render_path(TI camera_i, const std::string& modality){
        return golden::layout::render_png(review_directory(camera_i, modality), backend_tag());
    }

    std::string versus_path(TI camera_i, const std::string& modality, const std::string& name = "render-" + backend_tag()){
        return golden::layout::versus_png(review_directory(camera_i, modality), name);
    }

    // the earlier generations that exist for this modality (golden-vNN.png or data/golden-vNN.bin)
    std::vector<uint32_t> historic_versions(const std::string& golden_directory, bool binary){
        std::vector<uint32_t> versions;
        for(uint32_t version = 0; version < golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION; version++){
            if(std::filesystem::exists(binary ? golden::layout::golden_bin(golden_directory, version) : golden::layout::golden_png(golden_directory, version))){
                versions.push_back(version);
            }
        }
        return versions;
    }

    // metra series raytracing/golden/procthor/<backend>/<modality>/<metric> with the bound as
    // <metric>_bound and the corpus version once per process
    std::string metric_prefix(){
        return "raytracing/golden/procthor/" + backend_tag() + "/";
    }

    void log_metric(const std::string& name, double value, double bound){
        static const bool version_logged = metra::log(metric_prefix() + "version", golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION);
        (void)version_logged;
        metra::log(metric_prefix() + name, value);
        metra::log(metric_prefix() + name + "_bound", bound);
    }

    struct RGBStats{
        double mad = 0;
        double outlier_fraction = 0;
        double shading_outlier_fraction = 0;
        int channel_max[3] = {0, 0, 0};
    };

    RGBStats compare_rgb(const uint32_t* ours, const uint32_t* golden, int width, int height){
        RGBStats stats;
        const auto* ours_bytes = (const unsigned char*)ours;
        const auto* golden_bytes = (const unsigned char*)golden;
        const size_t count = (size_t)width * height;
        double total = 0;
        size_t outliers = 0;
        size_t shading_outliers = 0;
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
            shading_outliers += outlier && !golden::neighborhood_explains(golden, ours, width, height, (int)(pixel_i % width), (int)(pixel_i / width), RGB_OUTLIER_CHANNEL_DELTA);
        }
        stats.mad = total / ((double)count * 3);
        stats.outlier_fraction = (double)outliers / (double)count;
        stats.shading_outlier_fraction = (double)shading_outliers / (double)count;
        return stats;
    }

    // the current golden against ours and against every earlier generation, as heatmaps next to it
    template <typename SPEC>
    void write_png_reviews(TI camera_i, const std::string& modality, const uint32_t* ours, const std::vector<uint32_t>& golden_pixels){
        golden::write_camera_png(render_path(camera_i, modality), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        golden::write_camera_diff_png(versus_path(camera_i, modality), ours, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        for(const uint32_t version : historic_versions(golden_directory(camera_i, modality), false)){
            std::vector<uint32_t> historic;
            if(golden::load_camera_png(golden::layout::golden_png(golden_directory(camera_i, modality), version), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, historic)){
                golden::write_camera_diff_png(versus_path(camera_i, modality, golden::layout::golden_name(version)), golden_pixels.data(), historic.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
        }
    }

    template <typename SPEC>
    void expect_rgb_matches_golden(const char* modality, const Rendered& rendered){
        double worst_mad = 0;
        double worst_outliers = 0;
        double worst_shading_outliers = 0;
        const char* worst_id = CASES::POSES[0].id;
        int max_channel[3] = {0, 0, 0};
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            const uint32_t* ours = rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(golden::layout::golden_png(golden_directory(camera_i, modality), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/" << modality;
            const RGBStats stats = compare_rgb(ours, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            write_png_reviews<SPEC>(camera_i, modality, ours, golden_pixels);
            EXPECT_LE(stats.mad, RGB_MAD_THRESHOLD)
                << modality << " pose " << id << ": RGB mean abs diff " << stats.mad
                << " (max R=" << stats.channel_max[0] << " G=" << stats.channel_max[1] << " B=" << stats.channel_max[2] << ")";
            EXPECT_LE(stats.outlier_fraction, RGB_OUTLIER_FRACTION) << modality << " pose " << id << ": RGB outlier fraction " << stats.outlier_fraction;
            EXPECT_LE(stats.shading_outlier_fraction, RGB_SHADING_OUTLIER_FRACTION) << modality << " pose " << id << ": RGB shading outlier fraction " << stats.shading_outlier_fraction;
            worst_outliers = std::max(worst_outliers, stats.outlier_fraction);
            worst_shading_outliers = std::max(worst_shading_outliers, stats.shading_outlier_fraction);
            if(stats.mad > worst_mad){
                worst_mad = stats.mad;
                worst_id = id;
            }
            for(int channel = 0; channel < 3; channel++){
                if(stats.channel_max[channel] > max_channel[channel]) max_channel[channel] = stats.channel_max[channel];
            }
        }
        std::printf("[golden] %s: worst_pose=%s rgb_mad=%.4f rgb_outliers=%.4f%% shading_outliers=%.4f%% max=(%d,%d,%d) over %d poses\n", modality, worst_id, worst_mad, worst_outliers * 100, worst_shading_outliers * 100, max_channel[0], max_channel[1], max_channel[2], (int)SPEC::NUM_CAMERAS);
        log_metric(std::string(modality) + "/rgb_mad", worst_mad, RGB_MAD_THRESHOLD);
        log_metric(std::string(modality) + "/rgb_outliers", worst_outliers, RGB_OUTLIER_FRACTION);
        log_metric(std::string(modality) + "/shading_outliers", worst_shading_outliers, RGB_SHADING_OUTLIER_FRACTION);
    }

    template <typename SPEC, bool T_CAMERA_MOTION = true>
    void run_rgb_case(const char* modality){
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        expect_rgb_matches_golden<SPEC>(modality, rendered);
    }

    void run_depth_case(){
        using SPEC = CASES::LOW_RGBD;
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;

        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            const std::string directory = golden_directory(camera_i, "depth");
            std::vector<float> golden_depth;
            ASSERT_TRUE(golden::load_multi_camera_float_bin(golden::layout::golden_bin(directory, golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_depth)) << "failed to load golden depth: " << id;
            const float* ours = rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            golden::write_camera_depth_png(render_path(camera_i, "depth"), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            golden::write_camera_depth_diff_png(versus_path(camera_i, "depth"), golden_depth.data(), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            for(const uint32_t version : historic_versions(directory, true)){
                std::vector<float> historic;
                if(golden::load_multi_camera_float_bin(golden::layout::golden_bin(directory, version), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, historic)){
                    golden::write_camera_depth_diff_png(versus_path(camera_i, "depth", golden::layout::golden_name(version)), historic.data(), golden_depth.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
                }
            }
            double total_abs_diff = 0;
            size_t outliers = 0;
            for(size_t pixel_i = 0; pixel_i < golden_depth.size(); pixel_i++){
                const double diff = (double)ours[pixel_i] - (double)golden_depth[pixel_i];
                const double abs_diff = diff < 0 ? -diff : diff;
                total_abs_diff += abs_diff;
                const double magnitude = std::max((double)golden_depth[pixel_i], 1e-6);
                if(abs_diff / magnitude > DEPTH_OUTLIER_REL){
                    outliers++;
                }
            }
            const double depth_mad = total_abs_diff / (double)golden_depth.size() / (double)rendered.max_depth;
            const double outlier_fraction = (double)outliers / (double)golden_depth.size();
            EXPECT_LE(depth_mad, DEPTH_MAD_THRESHOLD) << "depth pose " << id << ": normalized depth mean abs diff " << depth_mad;
            EXPECT_LE(outlier_fraction, DEPTH_OUTLIER_FRACTION) << "depth pose " << id << ": depth outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, depth_mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] depth: depth_mad=%.2e depth_outliers=%.4f%%\n", worst_mad, worst_outliers * 100);
        log_metric("depth/depth_mad", worst_mad, DEPTH_MAD_THRESHOLD);
        log_metric("depth/depth_outliers", worst_outliers, DEPTH_OUTLIER_FRACTION);
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
            ASSERT_TRUE(golden::load_camera_png(golden::layout::golden_png(golden_directory(camera_i, "normals"), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/normals";
            const uint32_t* ours = encoded.data() + camera_i * SPEC::CAM_PIXELS;
            write_png_reviews<SPEC>(camera_i, "normals", ours, golden_pixels);
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
        std::printf("[golden] normals: normals_mad=%.4f normals_outliers=%.4f%% over %d poses\n", worst_mad, worst_outliers * 100, (int)SPEC::NUM_CAMERAS);
        log_metric("normals/normals_mad", worst_mad, NORMALS_MAD_THRESHOLD);
        log_metric("normals/normals_outliers", worst_outliers, NORMALS_OUTLIER_FRACTION);
    }

    struct FlowDirectionView{
        const char* suffix; // "" (backward, current camera grid) or "-forward" (previous camera grid)
        const std::vector<float>* flow;
        const std::vector<uint32_t>* status;
    };

    void write_flow_status_reviews(TI camera_i, const std::string& modality, const uint32_t* ours_status, const std::vector<uint32_t>& golden_status){
        std::vector<uint32_t> image(golden_status.size());
        golden::colorize_flow_status(ours_status, image.size(), image.data());
        golden::write_camera_png(render_path(camera_i, modality), image.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        const auto write_versus = [&](const std::string& name, const uint32_t* reference, const uint32_t* other){
            for(size_t pixel_i = 0; pixel_i < image.size(); pixel_i++){
                image[pixel_i] = golden::flow_status_diff_color(reference[pixel_i], other[pixel_i]);
            }
            golden::write_camera_png(versus_path(camera_i, modality, name), image.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        };
        write_versus("render-" + backend_tag(), golden_status.data(), ours_status);
        const std::string directory = golden_directory(camera_i, modality);
        for(const uint32_t version : historic_versions(directory, true)){
            std::vector<uint32_t> historic;
            if(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(directory, version), 1, CASES::CAM_WIDTH, CASES::CAM_HEIGHT, historic)){
                write_versus(golden::layout::golden_name(version), historic.data(), golden_status.data());
            }
        }
    }

    void write_flow_reviews(TI camera_i, const std::string& modality, const float* ours, const std::vector<float>& golden_flow){
        rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(ours, render_path(camera_i, modality).c_str());
        std::vector<float> difference(golden_flow.size());
        const auto write_versus = [&](const std::string& name, const float* reference, const float* other){
            for(size_t value_i = 0; value_i < difference.size(); value_i++){
                difference[value_i] = other[value_i] - reference[value_i];
            }
            rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(difference.data(), versus_path(camera_i, modality, name).c_str());
        };
        write_versus("render-" + backend_tag(), golden_flow.data(), ours);
        const std::string directory = golden_directory(camera_i, modality);
        for(const uint32_t version : historic_versions(directory, true)){
            std::vector<float> historic;
            if(golden::load_multi_camera_float_bin(golden::layout::golden_bin(directory, version), 1, CASES::CAM_WIDTH, CASES::CAM_HEIGHT, historic, 2)){
                write_versus(golden::layout::golden_name(version), historic.data(), golden_flow.data());
            }
        }
    }

    template <typename SPEC>
    void expect_flow_direction_matches_golden(const char* name, const FlowDirectionView& direction){
        using FlowStatus = rlt::rendering::raytracing::FlowStatus;
        const std::string modality = std::string(name) + direction.suffix;
        const std::string status_modality = modality + "-status";
        ASSERT_EQ(direction.flow->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS * 2) << modality;
        ASSERT_EQ(direction.status->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS) << modality;
        double worst_epe = 0;
        double worst_outliers = 0;
        double worst_status_mismatch = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_flow;
            std::vector<uint32_t> golden_status;
            ASSERT_TRUE(golden::load_multi_camera_float_bin(golden::layout::golden_bin(golden_directory(camera_i, modality), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_flow, 2)) << "failed to load golden: " << id << "/" << modality;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(golden_directory(camera_i, status_modality), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_status)) << "failed to load golden: " << id << "/" << status_modality;
            const float* ours = direction.flow->data() + (size_t)camera_i * SPEC::CAM_PIXELS * 2;
            const uint32_t* ours_status = direction.status->data() + (size_t)camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: exactly the pinned false color of the status binary
                std::vector<uint32_t> golden_png;
                ASSERT_TRUE(golden::load_camera_png(golden::layout::golden_png(golden_directory(camera_i, status_modality), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_png)) << "failed to load golden: " << id << "/" << status_modality << " review image";
                std::vector<uint32_t> expected(golden_status.size());
                golden::colorize_flow_status(golden_status.data(), golden_status.size(), expected.data());
                EXPECT_EQ(golden_png, expected) << status_modality << " pose " << id << ": review PNG does not match the binary";
            }
            write_flow_reviews(camera_i, modality, ours, golden_flow);
            write_flow_status_reviews(camera_i, status_modality, ours_status, golden_status);
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
            EXPECT_GT(compared, (size_t)0) << modality << " pose " << id << ": no pixel is VALID in both target and current";
            EXPECT_LE(mean_epe, FLOW_MEAN_EPE_THRESHOLD) << modality << " pose " << id << ": mean endpoint error " << mean_epe;
            EXPECT_LE(outlier_fraction, FLOW_OUTLIER_FRACTION) << modality << " pose " << id << ": flow outlier fraction " << outlier_fraction;
            EXPECT_LE(status_mismatch_fraction, FLOW_STATUS_MISMATCH_FRACTION) << modality << " pose " << id << ": flow status mismatch fraction " << status_mismatch_fraction;
            worst_epe = std::max(worst_epe, mean_epe);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
            worst_status_mismatch = std::max(worst_status_mismatch, status_mismatch_fraction);
        }
        std::printf("[golden] %s: flow_epe=%.4f flow_outliers=%.4f%% status_mismatch=%.4f%% over %d poses\n", modality.c_str(), worst_epe, worst_outliers * 100, worst_status_mismatch * 100, (int)SPEC::NUM_CAMERAS);
        log_metric(modality + "/flow_epe", worst_epe, FLOW_MEAN_EPE_THRESHOLD);
        log_metric(modality + "/flow_outliers", worst_outliers, FLOW_OUTLIER_FRACTION);
        log_metric(modality + "/status_mismatch", worst_status_mismatch, FLOW_STATUS_MISMATCH_FRACTION);
    }

    template <typename SPEC, bool T_CAMERA_MOTION = true>
    void run_flow_case(const char* name){
        static_assert(SPEC::HAS_FLOW && SPEC::HAS_FLOW_FORWARD, "the flow goldens cover both directions");
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES, T_CAMERA_MOTION>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
        expect_flow_direction_matches_golden<SPEC>(name, FlowDirectionView{"", &rendered.flow, &rendered.flow_status});
        expect_flow_direction_matches_golden<SPEC>(name, FlowDirectionView{"-forward", &rendered.flow_forward, &rendered.flow_forward_status});
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
            const std::string directory = golden_directory(camera_i, "segmentation");
            std::vector<uint32_t> golden_segmentation;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(directory, golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_segmentation)) << "failed to load golden: " << id << "/segmentation";
            const uint32_t* ours = rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: it must be exactly the false-color
                // encoding of the binary (mirrors the overlay corpus validation)
                std::vector<uint32_t> golden_png;
                ASSERT_TRUE(golden::load_camera_png(golden::layout::golden_png(directory, golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_png)) << "failed to load golden: " << id << "/segmentation review image";
                std::vector<uint32_t> expected(golden_segmentation.size());
                golden::colorize_segmentation(golden_segmentation.data(), golden_segmentation.size(), expected.data());
                EXPECT_EQ(golden_png, expected) << "segmentation pose " << id << ": review PNG does not match the binary";
            }
            std::vector<uint32_t> image(SPEC::CAM_PIXELS);
            golden::colorize_segmentation(ours, SPEC::CAM_PIXELS, image.data());
            golden::write_camera_png(render_path(camera_i, "segmentation"), image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            const auto write_versus = [&](const std::string& name, const uint32_t* reference, const uint32_t* other){
                for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                    image[pixel_i] = golden::segmentation_diff_color(reference[pixel_i], other[pixel_i]);
                }
                golden::write_camera_png(versus_path(camera_i, "segmentation", name), image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            };
            write_versus("render-" + backend_tag(), golden_segmentation.data(), ours);
            for(const uint32_t version : historic_versions(directory, true)){
                std::vector<uint32_t> historic;
                if(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(directory, version), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, historic)){
                    write_versus(golden::layout::golden_name(version), historic.data(), golden_segmentation.data());
                }
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
        std::printf("[golden] segmentation: segmentation_mismatch=%.4f%% background_mismatch=%.4f%% over %d poses\n", worst_mismatches * 100, worst_background_mismatches * 100, (int)SPEC::NUM_CAMERAS);
        log_metric("segmentation/segmentation_mismatch", worst_mismatches, SEGMENTATION_MISMATCH_FRACTION);
        log_metric("segmentation/background_mismatch", worst_background_mismatches, SEGMENTATION_BACKGROUND_MISMATCH_FRACTION);
    }
}

// a present corpus of another version fails: the fix is a data refresh (GOLDEN_LEDGER.md), not a code change
#define RL_TOOLS_GOLDEN_REQUIRE_VERSION() ASSERT_EQ(golden::manifest_version(MANIFEST_PATH), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION) << "golden corpus version mismatch at " << MANIFEST_PATH << " (see tests/src/rendering/raytracing/GOLDEN_LEDGER.md, run tests/src/rendering/raytracing/download_data.sh)"
#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() ASSERT_TRUE(goldens_available()) << "required golden renderings not found at " << CORPUS_DIRECTORY; RL_TOOLS_GOLDEN_REQUIRE_VERSION()
#else
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() if(!goldens_available()){ GTEST_SKIP() << "golden renderings not found at " << CORPUS_DIRECTORY << " (run tests/src/rendering/raytracing/download_data.sh or generate with test_rendering_raytracing_generate_golden_optix)"; } RL_TOOLS_GOLDEN_REQUIRE_VERSION()
#endif

TEST(RL_TOOLS_GOLDEN_SUITE, LOW){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::LOW_RGB>("low");
}
TEST(RL_TOOLS_GOLDEN_SUITE, MEDIUM){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::MEDIUM_RGB>("medium");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB>("high");
}
TEST(RL_TOOLS_GOLDEN_SUITE, VERY_HIGH){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::VERY_HIGH_RGB>("very-high");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_AA2){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_AA2>("high-aa2");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_MB4){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4>("high-mb4");
}
// dynamic-object motion blur: moving camera + spinning/translating overlay
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_MB4_DYNAMIC){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4_DYNAMIC>("high-mb4-dynamic");
}
// object-only isolation: static camera (shutter open == close), the blur is purely the overlay's
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_MB4_OBJECT){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4_DYNAMIC, false>("high-mb4-object");
}
TEST(RL_TOOLS_GOLDEN_SUITE, DEPTH){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_depth_case();
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
    run_flow_case<CASES::FLOW_DYNAMIC, false>("flow-dynamic");
}

TEST(RL_TOOLS_GOLDEN_SUITE, SEGMENTATION){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_segmentation_case();
}

TEST(RL_TOOLS_GOLDEN_SUITE, PROBES){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    using SPEC = CASES::LOW_RGB; // the probes are written during the low case
    DEVICE device;
    rlt::init(device);
    Rendered rendered;
    ASSERT_TRUE((golden::render_case<SPEC, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
    ASSERT_FALSE(rendered.probes.empty());

    int hit_mismatches = 0;
    double max_rel_distance_diff = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const char* id = CASES::POSES[camera_i].id;
        std::vector<float> golden_probes; // NUM_PROBES x [distance, hit]
        ASSERT_TRUE(golden::load_multi_camera_float_bin(golden::layout::golden_bin(golden_directory(camera_i, "probes"), golden::PROCTHOR_STATIC_SCENE_GOLDEN_VERSION), 1, SPEC::NUM_PROBES, 1, golden_probes, 2)) << "failed to load golden probes: " << id;
        const auto* ours = rendered.probes.data() + camera_i * SPEC::NUM_PROBES;
        for(size_t probe_i = 0; probe_i < (size_t)SPEC::NUM_PROBES; probe_i++){
            const float golden_distance = golden_probes[probe_i * 2 + 0];
            const bool golden_hit = golden_probes[probe_i * 2 + 1] != 0;
            if((ours[probe_i].hit != 0) != golden_hit){
                hit_mismatches++;
                continue;
            }
            if(golden_hit){
                const double diff = (double)ours[probe_i].distance - (double)golden_distance;
                const double abs_diff = diff < 0 ? -diff : diff;
                const double rel = abs_diff / std::max((double)golden_distance, 1e-6);
                if(rel > max_rel_distance_diff) max_rel_distance_diff = rel;
            }
        }
    }
    EXPECT_LE(hit_mismatches, PROBE_HIT_MISMATCH_MAX) << "probe hit flag mismatches: " << hit_mismatches << "/" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    EXPECT_LE(max_rel_distance_diff, PROBE_DISTANCE_REL) << "max relative probe distance diff: " << max_rel_distance_diff;
    std::printf("[golden] probes: hit_mismatches=%d/%d distance_rel=%.2e\n", hit_mismatches, (int)(SPEC::NUM_CAMERAS * SPEC::NUM_PROBES), max_rel_distance_diff);
    log_metric("probes/hit_mismatches", hit_mismatches, PROBE_HIT_MISMATCH_MAX);
    log_metric("probes/distance_rel", max_rel_distance_diff, PROBE_DISTANCE_REL);
}
