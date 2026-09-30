#include <rl_tools/operations/cpu.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include "overlay_golden_frames.h"
#include "overlay_golden_manifest.h"
#include "golden_metrics.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#ifndef RL_TOOLS_OVERLAY_GOLDEN_SUITE
#define RL_TOOLS_OVERLAY_GOLDEN_SUITE RENDERING_RAYTRACING_OVERLAY_GOLDEN_GENERIC
#endif

#ifndef RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME
#define RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME "generic"
#endif

#ifndef RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT
#define RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT "build/raytracing_golden_artifacts"
#endif

#ifdef RL_TOOLS_TEST_DATA_PATH
#define RL_TOOLS_OVERLAY_GOLDEN_TEST_DATA_PATH RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)
#else
#define RL_TOOLS_OVERLAY_GOLDEN_TEST_DATA_PATH "tests/data"
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using SPEC = overlay_scenarios::OverlaySpecification<T, TI>;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND)
using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif
using RENDERER = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;
using overlay_goldens::Frame;

// The reference backend (OptiX) rendered the corpus and reproduces it bit-exactly, so every metric
// is held to zero for it (see golden_comparison.cpp). The other backends get bounds at about
// twice the worst floor measured against overlay corpus v1: Vulkan (NVIDIA and lavapipe) is
// exact, the generic and WebGPU silhouette strays reach RGB MAD 0.25, normals MAD 0.125 and 0.29%
// of pixels for outliers, segmentation, background and flow status, id area 0.003, everything
// else zero; fractions with a zero floor allow 0.05% (8 pixels of a 4x64x64 view).
static constexpr bool REFERENCE_BACKEND = std::string_view(RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME) == golden::REFERENCE_BACKEND;
static constexpr double RGB_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.5;
static constexpr double RGB_ID_OVERLAP_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.05;
static constexpr int RGB_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double RGB_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.006;
static constexpr double RGB_SHADING_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0002; // outliers not explained by the neighborhood (golden_metrics.h), floor zero
static constexpr double DEPTH_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 1e-7;
static constexpr double DEPTH_OUTLIER_REL = 1e-3;
static constexpr double DEPTH_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double SEGMENTATION_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.006;
static constexpr double BACKGROUND_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.006;
static constexpr double ID_AREA_RELATIVE_TOLERANCE = REFERENCE_BACKEND ? 0 : 0.01;
// encoded normals compare like RGB, but silhouette/internal-edge bands (a different triangle
// winning the same pixel across backends) produce large per-pixel deltas, so the mean budget is
// wider while the outlier band stays bounded like segmentation's
static constexpr double NORMALS_MAD_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.25;
static constexpr int NORMALS_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double NORMALS_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.006;
// endpoint error on pixels VALID in both frames (backward additionally segmentation-gated):
// geometry-derived flow matches to float precision across backends except in the silhouette
// band the gates exclude; the status buffers may disagree only in that band
static constexpr double FLOW_MEAN_EPE_THRESHOLD = REFERENCE_BACKEND ? 0 : 0.001; // px
static constexpr double FLOW_OUTLIER_EPE = 0.5;                                   // px
static constexpr double FLOW_OUTLIER_FRACTION = REFERENCE_BACKEND ? 0 : 0.0005;
static constexpr double FLOW_STATUS_MISMATCH_FRACTION = REFERENCE_BACKEND ? 0 : 0.006;

static const std::string GOLDEN_ROOT = RL_TOOLS_OVERLAY_GOLDEN_TEST_DATA_PATH "/rendering_raytracing_golden";
static const std::string ARTIFACT_ROOT = RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT;

namespace {
    bool manifest_available(){
        return std::filesystem::is_regular_file(golden::layout::overlay_manifest_path(GOLDEN_ROOT));
    }

    // worst value per metric over the views and capture states of one scenario, logged to metra as
    // raytracing/golden/overlay/<backend>/<scenario>/<metric> with the bound as <metric>_bound and the
    // corpus version once per process; Vulkan on lavapipe is tagged because its floors differ
    struct ScenarioMetrics {
        static constexpr std::pair<const char*, double> BOUNDS[] = {
            {"rgb_mad", RGB_MAD_THRESHOLD}, {"rgb_outliers", RGB_OUTLIER_FRACTION}, {"shading_outliers", RGB_SHADING_OUTLIER_FRACTION},
            {"segmentation_mismatch", SEGMENTATION_MISMATCH_FRACTION}, {"background_mismatch", BACKGROUND_MISMATCH_FRACTION}, {"depth_outliers", DEPTH_OUTLIER_FRACTION},
            {"normals_mad", NORMALS_MAD_THRESHOLD}, {"normals_outliers", NORMALS_OUTLIER_FRACTION},
            {"flow_epe", FLOW_MEAN_EPE_THRESHOLD}, {"flow_outliers", FLOW_OUTLIER_FRACTION}, {"status_mismatch", FLOW_STATUS_MISMATCH_FRACTION},
            {"flow_forward_epe", FLOW_MEAN_EPE_THRESHOLD}, {"flow_forward_outliers", FLOW_OUTLIER_FRACTION}, {"flow_forward_status_mismatch", FLOW_STATUS_MISMATCH_FRACTION}
        };
        std::map<std::string, double> worst;
        void record(const char* name, double value){
            worst[name] = std::max(worst[name], value);
        }
        void log(const char* scenario) const {
            const char* driver_files = std::getenv("VK_DRIVER_FILES");
            const bool lavapipe = driver_files != nullptr && std::string(driver_files).find("lvp") != std::string::npos;
            const std::string prefix = std::string("raytracing/golden/overlay/") + RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME + (lavapipe ? "-lavapipe" : "") + "/";
            static const bool version_logged = metra::log(prefix + "version", golden::OVERLAY_GOLDEN_VERSION);
            (void)version_logged;
            for(const auto& [name, bound] : BOUNDS){
                metra::log(prefix + scenario + "/" + name, worst.count(name) ? worst.at(name) : 0.0);
                metra::log(prefix + scenario + "/" + name + "_bound", bound);
            }
        }
    };

// a present corpus of another version fails: the fix is a data refresh (GOLDEN_LEDGER.md), not a code change
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_VERSION() ASSERT_EQ(golden::manifest_version(golden::layout::overlay_manifest_path(GOLDEN_ROOT)), golden::OVERLAY_GOLDEN_VERSION) << "overlay golden corpus version mismatch (see tests/src/rendering/raytracing/GOLDEN_LEDGER.md, run tests/src/rendering/raytracing/download_data.sh)"
#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() ASSERT_TRUE(manifest_available()) << "required overlay goldens are missing at " << golden::layout::overlay_directory(GOLDEN_ROOT); RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_VERSION()
#else
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() if(!manifest_available()){ GTEST_SKIP() << "overlay goldens not found at " << golden::layout::overlay_directory(GOLDEN_ROOT) << " (run tests/src/rendering/raytracing/download_data.sh)"; } RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_VERSION()
#endif

    void expect_update_scope(overlay_scenarios::Scenario scenario, const Frame& initial, const Frame& updated, const char* view){
        const auto scope = overlay_goldens::update_scope<SPEC>(scenario, initial, updated);
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            if(scope[camera].should_change){
                EXPECT_GT(scope[camera].difference.rgb, overlay_goldens::MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " RGB";
                EXPECT_GT(scope[camera].difference.depth, overlay_goldens::MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " depth";
                EXPECT_GT(scope[camera].difference.segmentation, overlay_goldens::MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " segmentation";
            }
            else{
                EXPECT_EQ(scope[camera].difference.rgb, (size_t)0) << view << " camera " << camera << " RGB";
                EXPECT_EQ(scope[camera].difference.depth, (size_t)0) << view << " camera " << camera << " depth";
                EXPECT_EQ(scope[camera].difference.segmentation, (size_t)0) << view << " camera " << camera << " segmentation";
                EXPECT_EQ(scope[camera].difference.normals, (size_t)0) << view << " camera " << camera << " normals";
                EXPECT_EQ(scope[camera].difference.flow, (size_t)0) << view << " camera " << camera << " flow";
            }
        }
    }

    bool mask_has_id_near(
        const std::vector<uint32_t>& segmentation,
        size_t camera_offset,
        TI row,
        TI column,
        uint32_t id
    ){
        for(int row_delta = -1; row_delta <= 1; row_delta++){
            const int candidate_row = (int)row + row_delta;
            if(candidate_row < 0 || candidate_row >= (int)SPEC::CAM_HEIGHT){
                continue;
            }
            for(int column_delta = -1; column_delta <= 1; column_delta++){
                const int candidate_column = (int)column + column_delta;
                if(candidate_column < 0 || candidate_column >= (int)SPEC::CAM_WIDTH){
                    continue;
                }
                const size_t index = camera_offset
                    + (size_t)candidate_row * SPEC::CAM_WIDTH
                    + (size_t)candidate_column;
                if(segmentation[index] == id){
                    return true;
                }
            }
        }
        return false;
    }

    // segmentation masks may only differ within a one-pixel band: every hit must have a matching
    // hit in the other frame's 3x3 neighborhood
    size_t distant_mask_mismatches(const Frame& target, const Frame& current, TI camera, uint32_t id){
        const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
        size_t mismatches = 0;
        for(TI row = 0; row < SPEC::CAM_HEIGHT; row++){
            for(TI column = 0; column < SPEC::CAM_WIDTH; column++){
                const size_t index = offset + (size_t)row * SPEC::CAM_WIDTH + column;
                mismatches += target.segmentation[index] == id && !mask_has_id_near(current.segmentation, offset, row, column, id);
                mismatches += current.segmentation[index] == id && !mask_has_id_near(target.segmentation, offset, row, column, id);
            }
        }
        return mismatches;
    }

    void expect_camera_topology_and_objects(
        overlay_scenarios::Scenario scenario,
        const Frame& target,
        const Frame& current,
        const char* view
    ){
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const auto ids = overlay_goldens::expected_ids(scenario, camera);
            const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
            size_t unexpected_target = 0;
            size_t unexpected_current = 0;
            for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                const uint32_t target_id = target.segmentation[offset + pixel];
                const uint32_t current_id = current.segmentation[offset + pixel];
                unexpected_target += target_id != golden::SEGMENTATION_BACKGROUND_ID
                    && std::find(ids.begin(), ids.end(), target_id) == ids.end();
                unexpected_current += current_id != golden::SEGMENTATION_BACKGROUND_ID
                    && std::find(ids.begin(), ids.end(), current_id) == ids.end();
            }
            EXPECT_EQ(unexpected_target, (size_t)0) << view << " camera " << camera << " target";
            EXPECT_EQ(unexpected_current, (size_t)0) << view << " camera " << camera << " current";

            for(const uint32_t id : ids){
                const size_t target_count = overlay_goldens::camera_id_count<SPEC>(target, camera, id);
                const size_t current_count = overlay_goldens::camera_id_count<SPEC>(current, camera, id);
                EXPECT_GT(target_count, overlay_goldens::MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " target ID " << id;
                EXPECT_GT(current_count, overlay_goldens::MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " current ID " << id;
                const size_t area_difference = target_count > current_count
                    ? target_count - current_count
                    : current_count - target_count;
                EXPECT_LE((double)area_difference / std::max((double)target_count, 1.0), ID_AREA_RELATIVE_TOLERANCE)
                    << view << " camera " << camera << " area ID " << id;

                EXPECT_EQ(distant_mask_mismatches(target, current, camera, id), (size_t)0)
                    << view << " camera " << camera << " mask ID " << id;

                size_t overlap = 0;
                double rgb_total = 0;
                double depth_total = 0;
                for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                    const size_t index = offset + pixel;
                    if(target.segmentation[index] != id || current.segmentation[index] != id){
                        continue;
                    }
                    overlap++;
                    const auto* target_bytes = reinterpret_cast<const uint8_t*>(target.rgb.data() + index);
                    const auto* current_bytes = reinterpret_cast<const uint8_t*>(current.rgb.data() + index);
                    for(size_t channel = 0; channel < 3; channel++){
                        rgb_total += std::abs((int)target_bytes[channel] - (int)current_bytes[channel]);
                    }
                    depth_total += std::abs((double)target.depth[index] - current.depth[index]);
                }
                EXPECT_GT(overlap, overlay_goldens::MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " overlap ID " << id;
                EXPECT_LE(rgb_total / std::max((double)overlap * 3.0, 1.0), RGB_ID_OVERLAP_MAD_THRESHOLD)
                    << view << " camera " << camera << " RGB overlap ID " << id;
                EXPECT_LE(depth_total / std::max((double)overlap, 1.0) / current.max_depth, DEPTH_MAD_THRESHOLD)
                    << view << " camera " << camera << " depth overlap ID " << id;
            }
        }
    }

    void compare_frame(
        overlay_scenarios::Scenario scenario,
        overlay_goldens::CaptureState state,
        const overlay_goldens::View& view,
        const Frame& target,
        const Frame& current,
        ScenarioMetrics& metrics
    ){
        expect_camera_topology_and_objects(scenario, target, current, view.id);

        const auto review_paths = golden::layout::scenario_review_paths(
            ARTIFACT_ROOT,
            RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME,
            overlay_scenarios::scenario_id(scenario),
            overlay_goldens::capture_state_id(state),
            view.id
        );
        std::filesystem::create_directories(review_paths.directory);
        ASSERT_TRUE(golden::write_rgb_review_grid_pngs(
            review_paths.rgb_target_png, review_paths.rgb_current_png, review_paths.rgb_diff_png,
            target.rgb.data(), current.rgb.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));
        ASSERT_TRUE(golden::write_depth_review_grid_pngs(
            review_paths.depth_target_png, review_paths.depth_current_png, review_paths.depth_diff_png,
            target.depth.data(), current.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, current.max_depth
        ));
        ASSERT_TRUE(golden::write_segmentation_review_grid_pngs(
            review_paths.segmentation_target_png, review_paths.segmentation_current_png, review_paths.segmentation_diff_png,
            target.segmentation.data(), current.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));
        ASSERT_TRUE(golden::write_rgb_review_grid_pngs(
            review_paths.normals_target_png, review_paths.normals_current_png, review_paths.normals_diff_png,
            target.normals.data(), current.normals.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));
        const auto write_flow_review = [](const std::vector<float>& target_flow, const std::vector<float>& current_flow, const std::string& target_png, const std::string& current_png, const std::string& diff_png){
            rlt::rendering::raytracing::detail::write_flow_grid_png<SPEC>(target_flow.data(), target_png.c_str());
            rlt::rendering::raytracing::detail::write_flow_grid_png<SPEC>(current_flow.data(), current_png.c_str());
            std::vector<float> flow_difference(target_flow.size());
            for(size_t value_i = 0; value_i < flow_difference.size(); value_i++){
                flow_difference[value_i] = current_flow[value_i] - target_flow[value_i];
            }
            rlt::rendering::raytracing::detail::write_flow_grid_png<SPEC>(flow_difference.data(), diff_png.c_str());
        };
        write_flow_review(target.flow, current.flow, review_paths.flow_target_png, review_paths.flow_current_png, review_paths.flow_diff_png);
        write_flow_review(target.flow_forward, current.flow_forward, review_paths.flow_forward_target_png, review_paths.flow_forward_current_png, review_paths.flow_forward_diff_png);
        ASSERT_TRUE(golden::write_flow_status_review_grid_pngs(
            review_paths.flow_status_target_png, review_paths.flow_status_current_png, review_paths.flow_status_diff_png,
            target.flow_status.data(), current.flow_status.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));
        ASSERT_TRUE(golden::write_flow_status_review_grid_pngs(
            review_paths.flow_forward_status_target_png, review_paths.flow_forward_status_current_png, review_paths.flow_forward_status_diff_png,
            target.flow_forward_status.data(), current.flow_forward_status.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));

        double rgb_total = 0;
        size_t rgb_channels = 0;
        size_t rgb_outlier_pixels = 0;
        size_t rgb_shading_outlier_pixels = 0;
        size_t invalid_target_alpha = 0;
        size_t invalid_current_alpha = 0;
        size_t segmentation_mismatches = 0;
        size_t background_mismatches = 0;
        size_t depth_pixels = 0;
        size_t depth_outliers = 0;
        double normals_total = 0;
        size_t normals_outlier_pixels = 0;
        double flow_epe_total = 0;
        size_t flow_pixels = 0;
        size_t flow_outliers = 0;
        size_t flow_status_mismatches = 0;
        double flow_forward_epe_total = 0;
        size_t flow_forward_pixels = 0;
        size_t flow_forward_outliers = 0;
        size_t flow_forward_status_mismatches = 0;
        using FlowStatus = rlt::rendering::raytracing::FlowStatus;

        const size_t count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        for(size_t index = 0; index < count; index++){
            const auto* target_bytes = reinterpret_cast<const uint8_t*>(target.rgb.data() + index);
            const auto* current_bytes = reinterpret_cast<const uint8_t*>(current.rgb.data() + index);
            invalid_target_alpha += target_bytes[3] != 0xFFu;
            invalid_current_alpha += current_bytes[3] != 0xFFu;
            bool rgb_outlier = false;
            for(size_t channel = 0; channel < 3; channel++){
                const int difference = (int)current_bytes[channel] - (int)target_bytes[channel];
                const int absolute = difference < 0 ? -difference : difference;
                rgb_total += absolute;
                rgb_channels++;
                rgb_outlier = rgb_outlier || absolute > RGB_OUTLIER_CHANNEL_DELTA;
            }
            rgb_outlier_pixels += rgb_outlier;
            if(rgb_outlier){
                const size_t camera_offset = index - index % SPEC::CAM_PIXELS;
                const int local = (int)(index % SPEC::CAM_PIXELS);
                rgb_shading_outlier_pixels += !golden::neighborhood_explains(target.rgb.data() + camera_offset, current.rgb.data() + camera_offset, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, local % SPEC::CAM_WIDTH, local / SPEC::CAM_WIDTH, RGB_OUTLIER_CHANNEL_DELTA);
            }

            const auto* target_normal_bytes = reinterpret_cast<const uint8_t*>(target.normals.data() + index);
            const auto* current_normal_bytes = reinterpret_cast<const uint8_t*>(current.normals.data() + index);
            bool normals_outlier = false;
            for(size_t channel = 0; channel < 3; channel++){
                const int difference = (int)current_normal_bytes[channel] - (int)target_normal_bytes[channel];
                const int absolute = difference < 0 ? -difference : difference;
                normals_total += absolute;
                normals_outlier = normals_outlier || absolute > NORMALS_OUTLIER_CHANNEL_DELTA;
            }
            normals_outlier_pixels += normals_outlier;

            const uint32_t target_id = target.segmentation[index];
            segmentation_mismatches += target_id != current.segmentation[index];
            background_mismatches += (target_id == golden::SEGMENTATION_BACKGROUND_ID) != (current.segmentation[index] == golden::SEGMENTATION_BACKGROUND_ID);
            if(target_id == current.segmentation[index] && target_id != golden::SEGMENTATION_BACKGROUND_ID){
                const double difference = (double)current.depth[index] - target.depth[index];
                const double absolute = difference < 0 ? -difference : difference;
                const double magnitude = std::max((double)target.depth[index], 1e-6);
                depth_pixels++;
                depth_outliers += absolute / magnitude > DEPTH_OUTLIER_REL;

                if(target.flow_status[index] == FlowStatus::VALID && current.flow_status[index] == FlowStatus::VALID){
                    const double flow_du = (double)current.flow[index * 2 + 0] - target.flow[index * 2 + 0];
                    const double flow_dv = (double)current.flow[index * 2 + 1] - target.flow[index * 2 + 1];
                    const double endpoint_error = std::sqrt(flow_du * flow_du + flow_dv * flow_dv);
                    flow_epe_total += endpoint_error;
                    flow_pixels++;
                    flow_outliers += endpoint_error > FLOW_OUTLIER_EPE;
                }
            }
            flow_status_mismatches += target.flow_status[index] != current.flow_status[index];
            flow_forward_status_mismatches += target.flow_forward_status[index] != current.flow_forward_status[index];
            if(target.flow_forward_status[index] == FlowStatus::VALID && current.flow_forward_status[index] == FlowStatus::VALID){
                const double flow_du = (double)current.flow_forward[index * 2 + 0] - target.flow_forward[index * 2 + 0];
                const double flow_dv = (double)current.flow_forward[index * 2 + 1] - target.flow_forward[index * 2 + 1];
                const double endpoint_error = std::sqrt(flow_du * flow_du + flow_dv * flow_dv);
                flow_forward_epe_total += endpoint_error;
                flow_forward_pixels++;
                flow_forward_outliers += endpoint_error > FLOW_OUTLIER_EPE;
            }
        }

        const double rgb_mad = rgb_total / std::max((double)rgb_channels, 1.0);
        const double rgb_outliers = (double)rgb_outlier_pixels / count;
        const double shading_outliers = (double)rgb_shading_outlier_pixels / count;
        const double segmentation_mismatch = (double)segmentation_mismatches / count;
        const double background_mismatch = (double)background_mismatches / count;
        const double depth_outlier_fraction = (double)depth_outliers / std::max((double)depth_pixels, 1.0);
        const double normals_mad = normals_total / std::max((double)rgb_channels, 1.0);
        const double normals_outliers = (double)normals_outlier_pixels / count;
        const double flow_epe = flow_epe_total / std::max((double)flow_pixels, 1.0);
        const double flow_outlier_fraction = (double)flow_outliers / std::max((double)flow_pixels, 1.0);
        const double status_mismatch = (double)flow_status_mismatches / count;
        const double flow_forward_epe = flow_forward_epe_total / std::max((double)flow_forward_pixels, 1.0);
        const double flow_forward_outlier_fraction = (double)flow_forward_outliers / std::max((double)flow_forward_pixels, 1.0);
        const double flow_forward_status_mismatch = (double)flow_forward_status_mismatches / count;

        EXPECT_LE(rgb_mad, RGB_MAD_THRESHOLD) << view.id;
        EXPECT_LE(rgb_outliers, RGB_OUTLIER_FRACTION) << view.id;
        EXPECT_LE(shading_outliers, RGB_SHADING_OUTLIER_FRACTION) << view.id << " shading";
        EXPECT_EQ(invalid_target_alpha, (size_t)0) << view.id;
        EXPECT_EQ(invalid_current_alpha, (size_t)0) << view.id;
        EXPECT_LE(segmentation_mismatch, SEGMENTATION_MISMATCH_FRACTION) << view.id;
        EXPECT_LE(background_mismatch, BACKGROUND_MISMATCH_FRACTION) << view.id << " background";
        EXPECT_LE(depth_outlier_fraction, DEPTH_OUTLIER_FRACTION) << view.id;
        EXPECT_LE(normals_mad, NORMALS_MAD_THRESHOLD) << view.id << " normals";
        EXPECT_LE(normals_outliers, NORMALS_OUTLIER_FRACTION) << view.id << " normals";
        EXPECT_GT(flow_pixels, (size_t)0) << view.id << " flow";
        EXPECT_LE(flow_epe, FLOW_MEAN_EPE_THRESHOLD) << view.id << " flow";
        EXPECT_LE(flow_outlier_fraction, FLOW_OUTLIER_FRACTION) << view.id << " flow";
        EXPECT_LE(status_mismatch, FLOW_STATUS_MISMATCH_FRACTION) << view.id << " flow status";
        EXPECT_GT(flow_forward_pixels, (size_t)0) << view.id << " flow forward";
        EXPECT_LE(flow_forward_epe, FLOW_MEAN_EPE_THRESHOLD) << view.id << " flow forward";
        EXPECT_LE(flow_forward_outlier_fraction, FLOW_OUTLIER_FRACTION) << view.id << " flow forward";
        EXPECT_LE(flow_forward_status_mismatch, FLOW_STATUS_MISMATCH_FRACTION) << view.id << " flow forward status";

        metrics.record("rgb_mad", rgb_mad);
        metrics.record("rgb_outliers", rgb_outliers);
        metrics.record("shading_outliers", shading_outliers);
        metrics.record("segmentation_mismatch", segmentation_mismatch);
        metrics.record("background_mismatch", background_mismatch);
        metrics.record("depth_outliers", depth_outlier_fraction);
        metrics.record("normals_mad", normals_mad);
        metrics.record("normals_outliers", normals_outliers);
        metrics.record("flow_epe", flow_epe);
        metrics.record("flow_outliers", flow_outlier_fraction);
        metrics.record("status_mismatch", status_mismatch);
        metrics.record("flow_forward_epe", flow_forward_epe);
        metrics.record("flow_forward_outliers", flow_forward_outlier_fraction);
        metrics.record("flow_forward_status_mismatch", flow_forward_status_mismatch);
    }

    void run_scenario(overlay_scenarios::Scenario scenario){
        RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA();
        DEVICE device;
        rlt::init(device);
        auto scenario_state = overlay_scenarios::prepare(device, scenario);
        RENDERER renderer;
        rlt::malloc(device, renderer);
        rlt::generate_probe_directions(device, renderer);
        rlt::init(device, renderer, scenario_state.bundle, scenario_state.pool);
        overlay_scenarios::build_initial(device, renderer, scenario_state);
        rlt::update(device, renderer);

        ScenarioMetrics metrics;
        std::array<Frame, overlay_goldens::VIEWS.size()> initial_frames;
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            overlay_goldens::set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
            initial_frames[view_i] = overlay_goldens::capture(device, renderer);
            Frame target;
            ASSERT_TRUE(overlay_goldens::load_target_frame<SPEC>(
                GOLDEN_ROOT,
                scenario,
                overlay_goldens::CaptureState::INITIAL,
                overlay_goldens::VIEWS[view_i],
                target
            )) << overlay_scenarios::scenario_id(scenario) << " initial " << overlay_goldens::VIEWS[view_i].id;
            compare_frame(scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i], target, initial_frames[view_i], metrics);
        }

        if(overlay_goldens::capture_state_count(scenario) == 2){
            overlay_scenarios::apply_update(device, renderer, scenario_state);
            rlt::update(device, renderer);
            for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
                overlay_goldens::set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
                const auto updated = overlay_goldens::capture(device, renderer);
                Frame target;
                ASSERT_TRUE(overlay_goldens::load_target_frame<SPEC>(
                    GOLDEN_ROOT,
                    scenario,
                    overlay_goldens::CaptureState::UPDATED,
                    overlay_goldens::VIEWS[view_i],
                    target
                )) << overlay_scenarios::scenario_id(scenario) << " updated " << overlay_goldens::VIEWS[view_i].id;
                expect_update_scope(scenario, initial_frames[view_i], updated, overlay_goldens::VIEWS[view_i].id);
                compare_frame(scenario, overlay_goldens::CaptureState::UPDATED, overlay_goldens::VIEWS[view_i], target, updated, metrics);
            }
        }
        metrics.log(overlay_scenarios::scenario_id(scenario));

        rlt::free(device, renderer);
    }
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, SHARED_SCENE_NO_DYNAMIC){
    run_scenario(overlay_scenarios::Scenario::SHARED_SCENE_NO_DYNAMIC);
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, ALL_SHARED_MESH_TRANSFORM){
    run_scenario(overlay_scenarios::Scenario::ALL_SHARED_MESH_TRANSFORM);
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, PARTIALLY_SHARED_MESH_TRANSFORM){
    run_scenario(overlay_scenarios::Scenario::PARTIALLY_SHARED_MESH_TRANSFORM);
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, SHARED_MESH_INDIVIDUAL_TRANSFORM){
    run_scenario(overlay_scenarios::Scenario::SHARED_MESH_INDIVIDUAL_TRANSFORM);
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, DISJOINT){
    run_scenario(overlay_scenarios::Scenario::DISJOINT);
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, MIXED){
    run_scenario(overlay_scenarios::Scenario::MIXED);
}
