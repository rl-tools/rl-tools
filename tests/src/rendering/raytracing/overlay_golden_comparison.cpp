#include <rl_tools/operations/cpu.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#include "overlay_golden_frames.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
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

static constexpr double RGB_MAD_THRESHOLD = 1.0;
static constexpr double RGB_ID_OVERLAP_MAD_THRESHOLD = 1.0;
static constexpr int RGB_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double RGB_OUTLIER_FRACTION = 0.02;
static constexpr double DEPTH_MAD_THRESHOLD = 1e-3;
static constexpr double DEPTH_OUTLIER_REL = 1e-3;
static constexpr double DEPTH_OUTLIER_FRACTION = 0.005;
static constexpr double SEGMENTATION_MISMATCH_FRACTION = 0.02;
static constexpr double BACKGROUND_MISMATCH_FRACTION = 0.005;
static constexpr double ID_AREA_RELATIVE_TOLERANCE = 0.35;
// encoded normals compare like RGB, but silhouette/internal-edge bands (a different triangle
// winning the same pixel across backends) produce large per-pixel deltas, so the mean budget is
// wider while the outlier band stays bounded like segmentation's
static constexpr double NORMALS_MAD_THRESHOLD = 2.0;
static constexpr int NORMALS_OUTLIER_CHANNEL_DELTA = 8;
static constexpr double NORMALS_OUTLIER_FRACTION = 0.02;

static const std::string GOLDEN_ROOT = RL_TOOLS_OVERLAY_GOLDEN_TEST_DATA_PATH "/rendering_raytracing_golden";
static const std::string ARTIFACT_ROOT = RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT;

namespace {
    bool manifest_available(){
        return std::filesystem::is_regular_file(golden::layout::overlay_manifest_path(GOLDEN_ROOT));
    }

#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() ASSERT_TRUE(manifest_available()) << "required overlay goldens are missing at " << golden::layout::overlay_directory(GOLDEN_ROOT)
#else
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() if(!manifest_available()){ GTEST_SKIP() << "overlay goldens not found at " << golden::layout::overlay_directory(GOLDEN_ROOT); }
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
        const Frame& current
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

        double rgb_total = 0;
        size_t rgb_channels = 0;
        size_t rgb_outlier_pixels = 0;
        size_t invalid_target_alpha = 0;
        size_t invalid_current_alpha = 0;
        size_t segmentation_mismatches = 0;
        size_t background_mismatches = 0;
        size_t depth_pixels = 0;
        size_t depth_outliers = 0;
        double normals_total = 0;
        size_t normals_outlier_pixels = 0;

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
            }
        }

        const double rgb_mad = rgb_total / std::max((double)rgb_channels, 1.0);
        const double rgb_outliers = (double)rgb_outlier_pixels / count;
        const double segmentation_mismatch = (double)segmentation_mismatches / count;
        const double depth_outlier_fraction = (double)depth_outliers / std::max((double)depth_pixels, 1.0);

        EXPECT_LE(rgb_mad, RGB_MAD_THRESHOLD) << view.id;
        EXPECT_LE(rgb_outliers, RGB_OUTLIER_FRACTION) << view.id;
        EXPECT_EQ(invalid_target_alpha, (size_t)0) << view.id;
        EXPECT_EQ(invalid_current_alpha, (size_t)0) << view.id;
        EXPECT_LE(segmentation_mismatch, SEGMENTATION_MISMATCH_FRACTION) << view.id;
        EXPECT_LE((double)background_mismatches / count, BACKGROUND_MISMATCH_FRACTION) << view.id << " background";
        EXPECT_LE(depth_outlier_fraction, DEPTH_OUTLIER_FRACTION) << view.id;
        EXPECT_LE(normals_total / std::max((double)rgb_channels, 1.0), NORMALS_MAD_THRESHOLD) << view.id << " normals";
        EXPECT_LE((double)normals_outlier_pixels / count, NORMALS_OUTLIER_FRACTION) << view.id << " normals";
    }

    void run_scenario(overlay_scenarios::Scenario scenario){
        RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA();
        DEVICE device;
        rlt::init(device);
        auto scenario_state = overlay_scenarios::prepare(device, scenario);
        RENDERER renderer;
        rlt::malloc(device, renderer);
        rlt::generate_probe_directions(device, renderer);
        rlt::init(device, renderer, scenario_state.scene, scenario_state.pool);
        overlay_scenarios::build_initial(device, renderer, scenario_state);
        rlt::update(device, renderer);

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
            compare_frame(scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i], target, initial_frames[view_i]);
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
                compare_frame(scenario, overlay_goldens::CaptureState::UPDATED, overlay_goldens::VIEWS[view_i], target, updated);
            }
        }

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
