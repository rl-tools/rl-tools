// Renders every case of tests/src/rendering/raytracing/golden_cases.h and compares against the
// CPU golden renderings in tests/data/rendering/raytracing/goldens/procthor-train-1 (see
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
using rlt::is_miss;

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

// CPU reproduction is exact in the manifest's generation environment; other CPU toolchains
// must be qualified explicitly. Non-reference bounds retain the OptiX v2 calibration: RGB MAD
// 0.16, outliers 0.31%, shading outliers two pixels, normals MAD 0.0013, probes 5.3e-7,
// forward flow EPE 0.0005 px and one outlier/status pixel; zero-floor fractions allow 0.05%.
// Against CPU v3, Metal measures RGB MAD 0.0973, outliers 0.3906%, shading outliers three pixels,
// normals MAD 0.0015, depth MAD 4.42e-9, probes 8.48e-7, flow EPE 0.0003 px and one outlier/status
// pixel.
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
// CPU v4 on M5 Ultra / AppleClang 21 / Assimp 6.0.4: Metal instance IDs differ on 0.5234%
// static / 0.4974% overlay pixels due to coplanar wall/floor/ceiling overlaps; allow ~2x that floor.
static constexpr double METAL_INSTANCE_SEGMENTATION_MISMATCH_FRACTION = 0.011;
static constexpr double SEGMENTATION_BACKGROUND_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;

static const std::string SCENE_PATH = golden::PROCTHOR_TRAIN_1_SCENE;
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

    std::string golden_directory(TI camera_i, const char* state, const std::string& output){
        return golden::layout::output_directory(golden::layout::procthor_view_directory(GOLDEN_ROOT, CASES::POSES[camera_i].id), state, output);
    }

    std::string golden_png(TI camera_i, const char* state, const std::string& output){
        return golden::layout::golden_png(golden_directory(camera_i, state, output), golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION);
    }

    std::string golden_bin(TI camera_i, const char* state, const std::string& output){
        return golden::layout::golden_bin(golden_directory(camera_i, state, output), golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION);
    }

    // render-<backend>.png and diff-*.png live next to the goldens
    std::string review_directory(TI camera_i, const char* state, const std::string& output){
        const std::string directory = golden_directory(camera_i, state, output);
        std::filesystem::create_directories(directory);
        return directory;
    }

    std::string render_path(TI camera_i, const char* state, const std::string& output){
        return golden::layout::render_png(review_directory(camera_i, state, output), backend_tag());
    }

    std::string diff_path(TI camera_i, const char* state, const std::string& output, const std::string& name = "render-" + backend_tag()){
        return golden::layout::diff_png(review_directory(camera_i, state, output), golden::layout::golden_name(golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION), name);
    }

    // the earlier generations that exist for this output (golden-vNN.png or data/golden-vNN.bin)
    std::vector<uint32_t> historic_versions(const std::string& golden_directory, bool binary){
        std::vector<uint32_t> versions;
        for(uint32_t version = 0; version < golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION; version++){
            if(std::filesystem::exists(binary ? golden::layout::golden_bin(golden_directory, version) : golden::layout::golden_png(golden_directory, version))){
                versions.push_back(version);
            }
        }
        return versions;
    }

    // metra series raytracing/golden/procthor/<backend>/<state>/<output>/<metric> with the bound
    // as <metric>_bound and the corpus version once per process
    std::string metric_prefix(){
        return "raytracing/golden/procthor/" + backend_tag() + "/";
    }

    void log_metric(const std::string& name, double value, double bound){
        static const bool version_logged = metra::log(metric_prefix() + "version", golden::PROCTHOR_TRAIN_1_GOLDEN_VERSION);
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
                if(abs_diff > stats.channel_max[channel]){
                    stats.channel_max[channel] = abs_diff;
                }
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
    void write_png_reviews(TI camera_i, const char* state, const std::string& output, const uint32_t* ours, const std::vector<uint32_t>& golden_pixels){
        golden::write_camera_png(render_path(camera_i, state, output), ours, CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        golden::write_camera_diff_png(diff_path(camera_i, state, output), ours, golden_pixels.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        const std::string directory = golden_directory(camera_i, state, output);
        for(const uint32_t version : historic_versions(directory, false)){
            std::vector<uint32_t> historic;
            if(golden::load_camera_png(golden::layout::golden_png(directory, version), CASES::CAM_WIDTH, CASES::CAM_HEIGHT, historic)){
                golden::write_camera_diff_png(diff_path(camera_i, state, output, golden::layout::golden_name(version)), golden_pixels.data(), historic.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
            }
        }
    }

    template <typename CASE>
    void expect_rgb(const Rendered& rendered){
        using SPEC = typename CASE::SPEC;
        const std::string output = CASES::rgb_output<SPEC>();
        const std::string name = std::string(CASE::STATE) + "/" + output;
        double worst_mad = 0;
        double worst_outliers = 0;
        double worst_shading_outliers = 0;
        const char* worst_id = CASES::POSES[0].id;
        int max_channel[3] = {0, 0, 0};
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            const uint32_t* ours = rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(golden_png(camera_i, CASE::STATE, output), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/" << name;
            const RGBStats stats = compare_rgb(ours, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            write_png_reviews(camera_i, CASE::STATE, output, ours, golden_pixels);
            EXPECT_LE(stats.mad, RGB_MAD_THRESHOLD)
                << name << " pose " << id << ": RGB mean abs diff " << stats.mad
                << " (max R=" << stats.channel_max[0] << " G=" << stats.channel_max[1] << " B=" << stats.channel_max[2] << ")";
            EXPECT_LE(stats.outlier_fraction, RGB_OUTLIER_FRACTION) << name << " pose " << id << ": RGB outlier fraction " << stats.outlier_fraction;
            EXPECT_LE(stats.shading_outlier_fraction, RGB_SHADING_OUTLIER_FRACTION) << name << " pose " << id << ": RGB shading outlier fraction " << stats.shading_outlier_fraction;
            worst_outliers = std::max(worst_outliers, stats.outlier_fraction);
            worst_shading_outliers = std::max(worst_shading_outliers, stats.shading_outlier_fraction);
            if(stats.mad > worst_mad){
                worst_mad = stats.mad;
                worst_id = id;
            }
            for(int channel = 0; channel < 3; channel++){
                max_channel[channel] = std::max(max_channel[channel], stats.channel_max[channel]);
            }
        }
        std::printf("[golden] %s: worst_pose=%s rgb_mad=%.4f rgb_outliers=%.4f%% shading_outliers=%.4f%% max=(%d,%d,%d) over %d poses\n", name.c_str(), worst_id, worst_mad, worst_outliers * 100, worst_shading_outliers * 100, max_channel[0], max_channel[1], max_channel[2], (int)SPEC::NUM_CAMERAS);
        log_metric(name + "/rgb_mad", worst_mad, RGB_MAD_THRESHOLD);
        log_metric(name + "/rgb_outliers", worst_outliers, RGB_OUTLIER_FRACTION);
        log_metric(name + "/shading_outliers", worst_shading_outliers, RGB_SHADING_OUTLIER_FRACTION);
    }

    template <typename CASE>
    void expect_depth(const Rendered& rendered){
        using SPEC = typename CASE::SPEC;
        const std::string name = std::string(CASE::STATE) + "/depth";
        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            const std::string directory = golden_directory(camera_i, CASE::STATE, "depth");
            std::vector<float> golden_depth;
            ASSERT_TRUE(golden::load_multi_camera_float_bin(golden_bin(camera_i, CASE::STATE, "depth"), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_depth)) << "failed to load golden: " << id << "/" << name;
            const float* ours = rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            golden::write_camera_depth_png(render_path(camera_i, CASE::STATE, "depth"), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            golden::write_camera_depth_diff_png(diff_path(camera_i, CASE::STATE, "depth"), golden_depth.data(), ours, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            for(const uint32_t version : historic_versions(directory, true)){
                std::vector<float> historic;
                if(golden::load_multi_camera_float_bin(golden::layout::golden_bin(directory, version), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, historic)){
                    golden::write_camera_depth_diff_png(diff_path(camera_i, CASE::STATE, "depth", golden::layout::golden_name(version)), historic.data(), golden_depth.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
                }
            }
            double total_abs_diff = 0;
            size_t outliers = 0;
            size_t comparable_pixels = golden_depth.size();
            for(size_t pixel_i = 0; pixel_i < golden_depth.size(); pixel_i++){
                const bool ours_miss = is_miss(ours[pixel_i]);
                const bool golden_miss = is_miss(golden_depth[pixel_i]);
                if(ours_miss || golden_miss){
                    outliers += ours_miss != golden_miss;
                    comparable_pixels -= ours_miss != golden_miss;
                    continue;
                }
                const double diff = (double)ours[pixel_i] - (double)golden_depth[pixel_i];
                const double abs_diff = diff < 0 ? -diff : diff;
                total_abs_diff += abs_diff;
                const double magnitude = std::max((double)golden_depth[pixel_i], 1e-6);
                outliers += abs_diff / magnitude > DEPTH_OUTLIER_REL;
            }
            const double depth_mad = comparable_pixels > 0 ? total_abs_diff / (double)comparable_pixels / (double)rendered.max_depth : 0;
            const double outlier_fraction = (double)outliers / (double)golden_depth.size();
            EXPECT_LE(depth_mad, DEPTH_MAD_THRESHOLD) << name << " pose " << id << ": normalized depth mean abs diff " << depth_mad;
            EXPECT_LE(outlier_fraction, DEPTH_OUTLIER_FRACTION) << name << " pose " << id << ": depth outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, depth_mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] %s: depth_mad=%.2e depth_outliers=%.4f%%\n", name.c_str(), worst_mad, worst_outliers * 100);
        log_metric(name + "/depth_mad", worst_mad, DEPTH_MAD_THRESHOLD);
        log_metric(name + "/depth_outliers", worst_outliers, DEPTH_OUTLIER_FRACTION);
    }

    template <typename CASE>
    void expect_normals(const Rendered& rendered){
        using SPEC = typename CASE::SPEC;
        const std::string name = std::string(CASE::STATE) + "/normals";
        const size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        ASSERT_EQ(rendered.normals.size(), pixel_count * 3);
        std::vector<uint32_t> encoded(pixel_count);
        golden::colorize_normals(rendered.normals.data(), pixel_count, encoded.data());

        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(golden_png(camera_i, CASE::STATE, "normals"), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/" << name;
            const uint32_t* ours = encoded.data() + camera_i * SPEC::CAM_PIXELS;
            write_png_reviews(camera_i, CASE::STATE, "normals", ours, golden_pixels);
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
            EXPECT_LE(mad, NORMALS_MAD_THRESHOLD) << name << " pose " << id << ": encoded mean abs diff " << mad;
            EXPECT_LE(outlier_fraction, NORMALS_OUTLIER_FRACTION) << name << " pose " << id << ": outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] %s: normals_mad=%.4f normals_outliers=%.4f%% over %d poses\n", name.c_str(), worst_mad, worst_outliers * 100, (int)SPEC::NUM_CAMERAS);
        log_metric(name + "/normals_mad", worst_mad, NORMALS_MAD_THRESHOLD);
        log_metric(name + "/normals_outliers", worst_outliers, NORMALS_OUTLIER_FRACTION);
    }

    template <typename CASE>
    void expect_segmentation(const Rendered& rendered){
        using SPEC = typename CASE::SPEC;
        constexpr double mismatch_fraction_bound = !REFERENCE_BACKEND && !SPEC::HAS_SEMANTIC_SEGMENTATION
            && std::string_view(RL_TOOLS_GOLDEN_BACKEND_NAME) == "metal"
            ? METAL_INSTANCE_SEGMENTATION_MISMATCH_FRACTION : SEGMENTATION_MISMATCH_FRACTION;
        const char* output = SPEC::HAS_SEMANTIC_SEGMENTATION ? "segmentation-semantic" : "segmentation";
        const std::string name = std::string(CASE::STATE) + "/" + output;
        ASSERT_EQ(rendered.segmentation.size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);

        double worst_mismatches = 0;
        double worst_background_mismatches = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            const std::string directory = golden_directory(camera_i, CASE::STATE, output);
            std::vector<uint32_t> golden_segmentation;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(golden_bin(camera_i, CASE::STATE, output), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_segmentation)) << "failed to load golden: " << id << "/" << name;
            const uint32_t* ours = rendered.segmentation.data() + camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: it must be exactly the false-color encoding of the binary
                std::vector<uint32_t> golden_review;
                ASSERT_TRUE(golden::load_camera_png(golden_png(camera_i, CASE::STATE, output), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_review)) << "failed to load golden: " << id << "/" << name << " review image";
                std::vector<uint32_t> expected(golden_segmentation.size());
                golden::colorize_segmentation(golden_segmentation.data(), golden_segmentation.size(), expected.data());
                EXPECT_EQ(golden_review, expected) << name << " pose " << id << ": review PNG does not match the binary";
            }
            std::vector<uint32_t> image(SPEC::CAM_PIXELS);
            golden::colorize_segmentation(ours, SPEC::CAM_PIXELS, image.data());
            golden::write_camera_png(render_path(camera_i, CASE::STATE, output), image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            const auto write_diff = [&](const std::string& diff_name, const uint32_t* reference, const uint32_t* other){
                for(size_t pixel_i = 0; pixel_i < (size_t)SPEC::CAM_PIXELS; pixel_i++){
                    image[pixel_i] = golden::segmentation_diff_color(reference[pixel_i], other[pixel_i]);
                }
                golden::write_camera_png(diff_path(camera_i, CASE::STATE, output, diff_name), image.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            };
            write_diff("render-" + backend_tag(), golden_segmentation.data(), ours);
            for(const uint32_t version : historic_versions(directory, true)){
                std::vector<uint32_t> historic;
                if(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(directory, version), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, historic)){
                    write_diff(golden::layout::golden_name(version), historic.data(), golden_segmentation.data());
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
            EXPECT_LE(mismatch_fraction, mismatch_fraction_bound) << name << " pose " << id << ": id mismatch fraction " << mismatch_fraction;
            EXPECT_LE(background_fraction, SEGMENTATION_BACKGROUND_MISMATCH_FRACTION) << name << " pose " << id << ": hit/miss mismatch fraction " << background_fraction;
            worst_mismatches = std::max(worst_mismatches, mismatch_fraction);
            worst_background_mismatches = std::max(worst_background_mismatches, background_fraction);
        }
        std::printf("[golden] %s: segmentation_mismatch=%.4f%% background_mismatch=%.4f%% over %d poses\n", name.c_str(), worst_mismatches * 100, worst_background_mismatches * 100, (int)SPEC::NUM_CAMERAS);
        log_metric(name + "/segmentation_mismatch", worst_mismatches, mismatch_fraction_bound);
        log_metric(name + "/background_mismatch", worst_background_mismatches, SEGMENTATION_BACKGROUND_MISMATCH_FRACTION);
    }

    struct FlowDirectionView{
        const char* suffix; // "" (backward, current camera grid) or "-forward" (previous camera grid)
        const std::vector<float>* flow;
        const std::vector<uint32_t>* status;
    };

    void write_flow_status_reviews(TI camera_i, const char* state, const std::string& output, const uint32_t* ours_status, const std::vector<uint32_t>& golden_status){
        std::vector<uint32_t> image(golden_status.size());
        golden::colorize_flow_status(ours_status, image.size(), image.data());
        golden::write_camera_png(render_path(camera_i, state, output), image.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        const auto write_diff = [&](const std::string& name, const uint32_t* reference, const uint32_t* other){
            for(size_t pixel_i = 0; pixel_i < image.size(); pixel_i++){
                image[pixel_i] = golden::flow_status_diff_color(reference[pixel_i], other[pixel_i]);
            }
            golden::write_camera_png(diff_path(camera_i, state, output, name), image.data(), CASES::CAM_WIDTH, CASES::CAM_HEIGHT);
        };
        write_diff("render-" + backend_tag(), golden_status.data(), ours_status);
        const std::string directory = golden_directory(camera_i, state, output);
        for(const uint32_t version : historic_versions(directory, true)){
            std::vector<uint32_t> historic;
            if(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(directory, version), 1, CASES::CAM_WIDTH, CASES::CAM_HEIGHT, historic)){
                write_diff(golden::layout::golden_name(version), historic.data(), golden_status.data());
            }
        }
    }

    void write_flow_reviews(DEVICE& device, TI camera_i, const char* state, const std::string& output, const float* ours, const std::vector<float>& golden_flow){
        rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(device, ours, render_path(camera_i, state, output).c_str());
        std::vector<float> difference(golden_flow.size());
        const auto write_diff = [&](const std::string& name, const float* reference, const float* other){
            for(size_t value_i = 0; value_i < difference.size(); value_i++){
                difference[value_i] = other[value_i] - reference[value_i];
            }
            rlt::rendering::raytracing::detail::write_flow_grid_png<CASES::SingleCamera>(device, difference.data(), diff_path(camera_i, state, output, name).c_str());
        };
        write_diff("render-" + backend_tag(), golden_flow.data(), ours);
        const std::string directory = golden_directory(camera_i, state, output);
        for(const uint32_t version : historic_versions(directory, true)){
            std::vector<float> historic;
            if(golden::load_multi_camera_float_bin(golden::layout::golden_bin(directory, version), 1, CASES::CAM_WIDTH, CASES::CAM_HEIGHT, historic, 2)){
                write_diff(golden::layout::golden_name(version), historic.data(), golden_flow.data());
            }
        }
    }

    template <typename CASE>
    void expect_flow_direction(DEVICE& device, const FlowDirectionView& direction){
        using SPEC = typename CASE::SPEC;
        using FlowStatus = rlt::rendering::raytracing::FlowStatus;
        const std::string output = std::string("flow") + direction.suffix;
        const std::string status_output = output + "-status";
        const std::string name = std::string(CASE::STATE) + "/" + output;
        ASSERT_EQ(direction.flow->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS * 2) << name;
        ASSERT_EQ(direction.status->size(), (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS) << name;
        double worst_epe = 0;
        double worst_outliers = 0;
        double worst_status_mismatch = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_flow;
            std::vector<uint32_t> golden_status;
            ASSERT_TRUE(golden::load_multi_camera_float_bin(golden_bin(camera_i, CASE::STATE, output), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_flow, 2)) << "failed to load golden: " << id << "/" << name;
            ASSERT_TRUE(golden::load_multi_camera_uint32_bin(golden_bin(camera_i, CASE::STATE, status_output), 1, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_status)) << "failed to load golden: " << id << "/" << name << "-status";
            const float* ours = direction.flow->data() + (size_t)camera_i * SPEC::CAM_PIXELS * 2;
            const uint32_t* ours_status = direction.status->data() + (size_t)camera_i * SPEC::CAM_PIXELS;
            {
                // the review PNG is corpus surface too: exactly the pinned false color of the status binary
                std::vector<uint32_t> golden_review;
                ASSERT_TRUE(golden::load_camera_png(golden_png(camera_i, CASE::STATE, status_output), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_review)) << "failed to load golden: " << id << "/" << name << "-status review image";
                std::vector<uint32_t> expected(golden_status.size());
                golden::colorize_flow_status(golden_status.data(), golden_status.size(), expected.data());
                EXPECT_EQ(golden_review, expected) << name << "-status pose " << id << ": review PNG does not match the binary";
            }
            write_flow_reviews(device, camera_i, CASE::STATE, output, ours, golden_flow);
            write_flow_status_reviews(camera_i, CASE::STATE, status_output, ours_status, golden_status);
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
            EXPECT_GT(compared, (size_t)0) << name << " pose " << id << ": no pixel is VALID in both target and current";
            EXPECT_LE(mean_epe, FLOW_MEAN_EPE_THRESHOLD) << name << " pose " << id << ": mean endpoint error " << mean_epe;
            EXPECT_LE(outlier_fraction, FLOW_OUTLIER_FRACTION) << name << " pose " << id << ": flow outlier fraction " << outlier_fraction;
            EXPECT_LE(status_mismatch_fraction, FLOW_STATUS_MISMATCH_FRACTION) << name << " pose " << id << ": flow status mismatch fraction " << status_mismatch_fraction;
            worst_epe = std::max(worst_epe, mean_epe);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
            worst_status_mismatch = std::max(worst_status_mismatch, status_mismatch_fraction);
        }
        std::printf("[golden] %s: flow_epe=%.4f flow_outliers=%.4f%% status_mismatch=%.4f%% over %d poses\n", name.c_str(), worst_epe, worst_outliers * 100, worst_status_mismatch * 100, (int)SPEC::NUM_CAMERAS);
        log_metric(name + "/flow_epe", worst_epe, FLOW_MEAN_EPE_THRESHOLD);
        log_metric(name + "/flow_outliers", worst_outliers, FLOW_OUTLIER_FRACTION);
        log_metric(name + "/status_mismatch", worst_status_mismatch, FLOW_STATUS_MISMATCH_FRACTION);
    }

    template <typename CASE>
    void expect_probes(const Rendered& rendered){
        using SPEC = typename CASE::SPEC;
        const std::string name = std::string(CASE::STATE) + "/probes";
        ASSERT_EQ(rendered.probes.size(), (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
        int hit_mismatches = 0;
        double max_rel_distance_diff = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_probes; // NUM_PROBES x [distance, hit]
            ASSERT_TRUE(golden::load_multi_camera_float_bin(golden_bin(camera_i, CASE::STATE, "probes"), 1, SPEC::NUM_PROBES, 1, golden_probes, 2)) << "failed to load golden: " << id << "/" << name;
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
                    max_rel_distance_diff = std::max(max_rel_distance_diff, abs_diff / std::max((double)golden_distance, 1e-6));
                }
            }
        }
        EXPECT_LE(hit_mismatches, PROBE_HIT_MISMATCH_MAX) << name << ": probe hit flag mismatches " << hit_mismatches << "/" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
        EXPECT_LE(max_rel_distance_diff, PROBE_DISTANCE_REL) << name << ": max relative probe distance diff " << max_rel_distance_diff;
        std::printf("[golden] %s: hit_mismatches=%d/%d distance_rel=%.2e\n", name.c_str(), hit_mismatches, (int)(SPEC::NUM_CAMERAS * SPEC::NUM_PROBES), max_rel_distance_diff);
        log_metric(name + "/hit_mismatches", hit_mismatches, PROBE_HIT_MISMATCH_MAX);
        log_metric(name + "/distance_rel", max_rel_distance_diff, PROBE_DISTANCE_REL);
    }
}

// a present corpus of another version fails: the fix is a data refresh (GOLDEN_LEDGER.md), not a code change
#define RL_TOOLS_GOLDEN_REQUIRE_MANIFEST() ASSERT_EQ(golden::manifest_error(MANIFEST_PATH), "") << "invalid golden corpus at " << MANIFEST_PATH << " (see tests/src/rendering/raytracing/GOLDEN_LEDGER.md)"
#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() ASSERT_TRUE(goldens_available()) << "required golden renderings not found at " << CORPUS_DIRECTORY; RL_TOOLS_GOLDEN_REQUIRE_MANIFEST()
#else
#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() if(!goldens_available()){ GTEST_SKIP() << "golden renderings not found at " << CORPUS_DIRECTORY << " (run tests/src/rendering/raytracing/download_data.sh or generate with test_rendering_raytracing_generate_golden_cpu)"; } RL_TOOLS_GOLDEN_REQUIRE_MANIFEST()
#endif

// one test per entry of the case table, registered under a readable name (static/high-mb4 ->
// static_high_mb4) instead of gtest's typed-test spelling of the whole specification; gtest
// requires one fixture class per suite, so the case enters through the body pointer
template <typename CASE>
void run_case(){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    using SPEC = typename CASE::SPEC;
    DEVICE device;
    rlt::init(device);
    Rendered rendered;
    ASSERT_TRUE((golden::render_case<CASE, BACKEND, DEVICE, CASES>(device, SCENE_PATH, rendered))) << "failed to load scene: " << SCENE_PATH;
    if constexpr(SPEC::HAS_RGB){
        expect_rgb<CASE>(rendered);
    }
    if constexpr(CASE::PROBES){
        expect_probes<CASE>(rendered);
    }
    if constexpr(SPEC::HAS_DEPTH){
        expect_depth<CASE>(rendered);
    }
    if constexpr(SPEC::HAS_NORMALS){
        expect_normals<CASE>(rendered);
    }
    if constexpr(SPEC::HAS_SEGMENTATION){
        expect_segmentation<CASE>(rendered);
    }
    if constexpr(SPEC::HAS_FLOW){
        expect_flow_direction<CASE>(device, FlowDirectionView{"", &rendered.flow, &rendered.flow_status});
        expect_flow_direction<CASE>(device, FlowDirectionView{"-forward", &rendered.flow_forward, &rendered.flow_forward_status});
    }
}

class GoldenCase: public ::testing::Test{
public:
    explicit GoldenCase(void (*body)()): body(body){}
    void TestBody() override {
        body();
    }
private:
    void (*body)();
};

template <typename... CASE>
bool register_cases(CASES::List<CASE...>){
    const auto register_case = [](const std::string& name, void (*body)()){
        std::string test_name = name;
        std::replace(test_name.begin(), test_name.end(), '/', '_');
        std::replace(test_name.begin(), test_name.end(), '-', '_');
        ::testing::RegisterTest(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_GOLDEN_SUITE), test_name.c_str(), nullptr, nullptr, __FILE__, __LINE__, [body](){ return new GoldenCase(body); });
    };
    (register_case(CASES::name<CASE>(), &run_case<CASE>), ...);
    return true;
}

static const bool CASES_REGISTERED = register_cases(CASES::ALL{});
