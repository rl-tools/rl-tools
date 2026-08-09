#include <rl_tools/operations/cpu.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#include "golden_io.h"
#include "overlay_golden_cases.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

static constexpr double RGB_MAD_THRESHOLD = 1.0;
static constexpr double RGB_INTERIOR_MAD_THRESHOLD = 1.0;
static constexpr double RGB_OUTLIER_FRACTION = 0.02;
static constexpr double DEPTH_MAD_THRESHOLD = 1e-3;
static constexpr double DEPTH_OUTLIER_REL = 1e-3;
static constexpr double DEPTH_OUTLIER_FRACTION = 0.005;
static constexpr double SEGMENTATION_MISMATCH_FRACTION = 0.02;
static constexpr double SEGMENTATION_INTERIOR_MISMATCH_FRACTION = 0.005;
static constexpr double ID_AREA_RELATIVE_TOLERANCE = 0.35;
static constexpr size_t MIN_VISIBLE_ID_PIXELS = 4;
static constexpr size_t MIN_UPDATE_CHANGED_PIXELS = 8;
static constexpr size_t MIN_POSE_CHANGED_PIXELS = 32;
static constexpr size_t MIN_POSE_SEGMENTATION_CHANGED_PIXELS = 8;

static const std::string GOLDEN_ROOT = RL_TOOLS_OVERLAY_GOLDEN_TEST_DATA_PATH "/rendering_raytracing_golden";
static const std::string ARTIFACT_ROOT = RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT;

namespace {
    struct Frame {
        std::vector<uint32_t> rgb;
        std::vector<float> depth;
        std::vector<uint32_t> segmentation;
        float max_depth = 0;
    };

    struct DifferenceCounts {
        size_t rgb = 0;
        size_t depth = 0;
        size_t segmentation = 0;
    };

    template <typename ELEMENT, typename TENSOR>
    void read_output(const TENSOR& tensor, ELEMENT* destination, size_t count){
#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND) && defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        cudaMemcpy(destination, rlt::data(tensor), count * sizeof(ELEMENT), cudaMemcpyDeviceToHost);
#else
        std::memcpy(destination, rlt::data(tensor), count * sizeof(ELEMENT));
#endif
    }

    void set_view(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, const overlay_goldens::View& view){
        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        const auto camera = rlt::make_camera_data(view.position, view.look_at, view.up, SPEC::COS_FOVY, aspect);
        std::array<rlt::rendering::raytracing::Camera<T>, SPEC::NUM_CAMERAS> cameras;
        cameras.fill(camera);
#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND) && defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        cudaMemcpy(rlt::data(rlt::cameras(device, renderer)), cameras.data(), sizeof(cameras), cudaMemcpyHostToDevice);
#else
        std::memcpy(rlt::data(rlt::cameras(device, renderer)), cameras.data(), sizeof(cameras));
#endif
    }

    Frame capture(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer){
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        const size_t count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        Frame frame;
        frame.rgb.resize(count);
        frame.depth.resize(count);
        frame.segmentation.resize(count);
        read_output(rlt::frame_buffer(device, renderer), frame.rgb.data(), count);
        read_output(rlt::depth_buffer(device, renderer), frame.depth.data(), count);
        read_output(rlt::segmentation_buffer(device, renderer), frame.segmentation.data(), count);
        frame.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        return frame;
    }

#if !defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
    bool manifest_available(){
        return std::filesystem::is_regular_file(golden::layout::overlay_manifest_path(GOLDEN_ROOT));
    }
#endif

    bool manifest_valid(){
        std::ifstream input(golden::layout::overlay_manifest_path(GOLDEN_ROOT));
        if(!input){
            return false;
        }
        try{
            nlohmann::json manifest;
            input >> manifest;
            std::vector<std::string> expected_scenarios;
            nlohmann::json expected_states = nlohmann::json::object();
            for(const auto scenario : overlay_scenarios::SCENARIOS){
                const std::string id = overlay_scenarios::scenario_id(scenario);
                expected_scenarios.push_back(id);
                expected_states[id] = overlay_goldens::capture_state_count(scenario) == 1
                    ? nlohmann::json::array({"initial"})
                    : nlohmann::json::array({"initial", "updated"});
            }
            std::vector<std::string> expected_views;
            for(const auto& view : overlay_goldens::VIEWS){
                expected_views.emplace_back(view.id);
            }
            return manifest.at("schema_version").get<uint32_t>() == 1
                && manifest.at("binary_format_version").get<uint32_t>() == golden::MULTI_CAMERA_BINARY_VERSION
                && manifest.at("reference_backend").get<std::string>() == "optix"
                && manifest.at("num_cameras").get<size_t>() == SPEC::NUM_CAMERAS
                && manifest.at("width").get<size_t>() == SPEC::CAM_WIDTH
                && manifest.at("height").get<size_t>() == SPEC::CAM_HEIGHT
                && manifest.at("camera_grid").get<std::string>() == "2x2 row-major logical cameras 0,1,2,3"
                && manifest.at("views").get<std::vector<std::string>>() == expected_views
                && manifest.at("scenarios").get<std::vector<std::string>>() == expected_scenarios
                && manifest.at("capture_states") == expected_states;
        }
        catch(const nlohmann::json::exception&){
            return false;
        }
    }

#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() ASSERT_TRUE(manifest_valid()) << "required overlay goldens are missing or invalid at " << golden::layout::overlay_directory(GOLDEN_ROOT)
#else
#define RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA() if(!manifest_available()){ GTEST_SKIP() << "overlay goldens not found at " << golden::layout::overlay_directory(GOLDEN_ROOT); } ASSERT_TRUE(manifest_valid())
#endif

    bool load_target_frame(
        overlay_scenarios::Scenario scenario,
        overlay_goldens::CaptureState state,
        const overlay_goldens::View& view,
        Frame& target
    ){
        const auto paths = golden::layout::scenario_target_paths(
            GOLDEN_ROOT,
            overlay_scenarios::scenario_id(scenario),
            overlay_goldens::capture_state_id(state),
            view.id
        );
        return golden::load_camera_grid_png(paths.rgb_png, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.rgb)
            && golden::load_multi_camera_float_bin(paths.depth_bin, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.depth)
            && golden::load_multi_camera_uint32_bin(paths.segmentation_bin, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.segmentation);
    }

    DifferenceCounts camera_difference(const Frame& first, const Frame& second, TI camera){
        DifferenceCounts difference;
        const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
        for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
            const size_t index = offset + pixel;
            difference.rgb += first.rgb[index] != second.rgb[index];
            difference.depth += first.depth[index] != second.depth[index];
            difference.segmentation += first.segmentation[index] != second.segmentation[index];
        }
        return difference;
    }

    DifferenceCounts frame_difference(const Frame& first, const Frame& second){
        DifferenceCounts difference;
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const auto camera_counts = camera_difference(first, second, camera);
            difference.rgb += camera_counts.rgb;
            difference.depth += camera_counts.depth;
            difference.segmentation += camera_counts.segmentation;
        }
        return difference;
    }

    void expect_update_scope(overlay_scenarios::Scenario scenario, const Frame& initial, const Frame& updated, const char* view){
        const auto affected = overlay_scenarios::definition(scenario).affected_camera_mask;
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const bool should_change = (affected & overlay_scenarios::camera_bit(camera)) != 0;
            const auto difference = camera_difference(initial, updated, camera);
            if(should_change){
                EXPECT_GT(difference.rgb, MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " RGB";
                EXPECT_GT(difference.depth, MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " depth";
                EXPECT_GT(difference.segmentation, MIN_UPDATE_CHANGED_PIXELS) << view << " camera " << camera << " segmentation";
            }
            else{
                EXPECT_EQ(difference.rgb, (size_t)0) << view << " camera " << camera << " RGB";
                EXPECT_EQ(difference.depth, (size_t)0) << view << " camera " << camera << " depth";
                EXPECT_EQ(difference.segmentation, (size_t)0) << view << " camera " << camera << " segmentation";
            }
        }
    }

    std::vector<uint32_t> expected_ids(overlay_scenarios::Scenario scenario, TI camera){
        std::vector<uint32_t> ids = {0};
        for(const auto& placement : overlay_scenarios::definition(scenario).placements){
            if((placement.cameras & overlay_scenarios::camera_bit(camera)) != 0){
                ids.push_back(placement.expected_id);
            }
        }
        return ids;
    }

    size_t camera_id_count(const std::vector<uint32_t>& segmentation, TI camera, uint32_t id){
        const auto begin = segmentation.begin() + (size_t)camera * SPEC::CAM_PIXELS;
        return static_cast<size_t>(std::count(begin, begin + SPEC::CAM_PIXELS, id));
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

    void expect_camera_topology_and_objects(
        overlay_scenarios::Scenario scenario,
        const Frame& target,
        const Frame& current,
        const char* view
    ){
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const auto ids = expected_ids(scenario, camera);
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
                const size_t target_count = camera_id_count(target.segmentation, camera, id);
                const size_t current_count = camera_id_count(current.segmentation, camera, id);
                EXPECT_GT(target_count, MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " target ID " << id;
                EXPECT_GT(current_count, MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " current ID " << id;
                const size_t area_difference = target_count > current_count
                    ? target_count - current_count
                    : current_count - target_count;
                EXPECT_LE((double)area_difference / std::max((double)target_count, 1.0), ID_AREA_RELATIVE_TOLERANCE)
                    << view << " camera " << camera << " ID " << id;

                size_t distant_mask_mismatches = 0;
                size_t overlap = 0;
                double rgb_total = 0;
                double depth_total = 0;
                for(TI row = 0; row < SPEC::CAM_HEIGHT; row++){
                    for(TI column = 0; column < SPEC::CAM_WIDTH; column++){
                        const size_t index = offset + (size_t)row * SPEC::CAM_WIDTH + column;
                        const bool target_hit = target.segmentation[index] == id;
                        const bool current_hit = current.segmentation[index] == id;
                        distant_mask_mismatches += target_hit && !mask_has_id_near(current.segmentation, offset, row, column, id);
                        distant_mask_mismatches += current_hit && !mask_has_id_near(target.segmentation, offset, row, column, id);
                        if(target_hit && current_hit){
                            overlap++;
                            const auto* target_bytes = reinterpret_cast<const uint8_t*>(target.rgb.data() + index);
                            const auto* current_bytes = reinterpret_cast<const uint8_t*>(current.rgb.data() + index);
                            for(size_t channel = 0; channel < 3; channel++){
                                rgb_total += std::abs((int)target_bytes[channel] - (int)current_bytes[channel]);
                            }
                            depth_total += std::abs((double)target.depth[index] - current.depth[index]);
                        }
                    }
                }
                EXPECT_EQ(distant_mask_mismatches, (size_t)0) << view << " camera " << camera << " ID " << id;
                EXPECT_GT(overlap, MIN_VISIBLE_ID_PIXELS) << view << " camera " << camera << " overlap ID " << id;
                EXPECT_LE(rgb_total / std::max((double)overlap * 3.0, 1.0), RGB_INTERIOR_MAD_THRESHOLD)
                    << view << " camera " << camera << " ID " << id;
                EXPECT_LE(depth_total / std::max((double)overlap, 1.0) / current.max_depth, DEPTH_MAD_THRESHOLD)
                    << view << " camera " << camera << " ID " << id;
            }
        }
    }

    bool pixel_is_target_interior(const std::vector<uint32_t>& segmentation, size_t camera_offset, TI row, TI column){
        if(row == 0 || column == 0 || row + 1 == SPEC::CAM_HEIGHT || column + 1 == SPEC::CAM_WIDTH){
            return false;
        }
        const uint32_t id = segmentation[camera_offset + (size_t)row * SPEC::CAM_WIDTH + column];
        for(int row_delta = -1; row_delta <= 1; row_delta++){
            for(int column_delta = -1; column_delta <= 1; column_delta++){
                const size_t index = camera_offset
                    + (size_t)((int)row + row_delta) * SPEC::CAM_WIDTH
                    + (size_t)((int)column + column_delta);
                if(segmentation[index] != id){
                    return false;
                }
            }
        }
        return true;
    }

    void compare_frame(
        overlay_scenarios::Scenario scenario,
        overlay_goldens::CaptureState state,
        const overlay_goldens::View& view,
        const Frame& current
    ){
        const auto target_paths = golden::layout::scenario_target_paths(
            GOLDEN_ROOT,
            overlay_scenarios::scenario_id(scenario),
            overlay_goldens::capture_state_id(state),
            view.id
        );
        Frame target;
        ASSERT_TRUE(load_target_frame(scenario, state, view, target)) << target_paths.directory;
        const auto& target_rgb = target.rgb;
        const auto& target_depth = target.depth;
        const auto& target_segmentation = target.segmentation;
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
            target_rgb.data(), current.rgb.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));
        ASSERT_TRUE(golden::write_depth_review_grid_pngs(
            review_paths.depth_target_png, review_paths.depth_current_png, review_paths.depth_diff_png,
            target_depth.data(), current.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, current.max_depth
        ));
        ASSERT_TRUE(golden::write_segmentation_review_grid_pngs(
            review_paths.segmentation_target_png, review_paths.segmentation_current_png, review_paths.segmentation_diff_png,
            target_segmentation.data(), current.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT
        ));

        double rgb_total = 0;
        double rgb_interior_total = 0;
        size_t rgb_channels = 0;
        size_t rgb_interior_channels = 0;
        size_t rgb_outlier_pixels = 0;
        size_t invalid_target_alpha = 0;
        size_t invalid_current_alpha = 0;
        size_t segmentation_mismatches = 0;
        size_t segmentation_interior_mismatches = 0;
        size_t segmentation_interior_pixels = 0;
        double depth_total = 0;
        size_t depth_pixels = 0;
        size_t depth_outliers = 0;

        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const size_t camera_offset = (size_t)camera * SPEC::CAM_PIXELS;
            for(TI row = 0; row < SPEC::CAM_HEIGHT; row++){
                for(TI column = 0; column < SPEC::CAM_WIDTH; column++){
                    const size_t index = camera_offset + (size_t)row * SPEC::CAM_WIDTH + column;
                    const auto* target_bytes = reinterpret_cast<const uint8_t*>(target_rgb.data() + index);
                    const auto* current_bytes = reinterpret_cast<const uint8_t*>(current.rgb.data() + index);
                    invalid_target_alpha += target_bytes[3] != 0xFFu;
                    invalid_current_alpha += current_bytes[3] != 0xFFu;
                    bool rgb_outlier = false;
                    const bool interior = pixel_is_target_interior(target_segmentation, camera_offset, row, column);
                    segmentation_interior_pixels += interior;
                    for(size_t channel = 0; channel < 3; channel++){
                        const int difference = (int)current_bytes[channel] - (int)target_bytes[channel];
                        const int absolute = difference < 0 ? -difference : difference;
                        rgb_total += absolute;
                        rgb_channels++;
                        rgb_outlier = rgb_outlier || absolute > 8;
                        if(interior){
                            rgb_interior_total += absolute;
                            rgb_interior_channels++;
                        }
                    }
                    rgb_outlier_pixels += rgb_outlier;

                    if(target_segmentation[index] != current.segmentation[index]){
                        segmentation_mismatches++;
                        segmentation_interior_mismatches += interior;
                    }
                    const uint32_t target_id = target_segmentation[index];
                    if(target_id == current.segmentation[index] && target_id != 0xFFFFFFFFu){
                        const double difference = (double)current.depth[index] - target_depth[index];
                        const double absolute = difference < 0 ? -difference : difference;
                        const double magnitude = std::max((double)target_depth[index], 1e-6);
                        depth_total += absolute;
                        depth_pixels++;
                        depth_outliers += absolute / magnitude > DEPTH_OUTLIER_REL;
                    }
                }
            }
        }

        const double rgb_mad = rgb_total / std::max((double)rgb_channels, 1.0);
        const double rgb_interior_mad = rgb_interior_total / std::max((double)rgb_interior_channels, 1.0);
        const double rgb_outliers = (double)rgb_outlier_pixels / current.rgb.size();
        const double segmentation_mismatch = (double)segmentation_mismatches / current.segmentation.size();
        const double segmentation_interior_mismatch = (double)segmentation_interior_mismatches / std::max((double)segmentation_interior_pixels, 1.0);
        const double depth_mad = depth_total / std::max((double)depth_pixels, 1.0) / current.max_depth;
        const double depth_outlier_fraction = (double)depth_outliers / std::max((double)depth_pixels, 1.0);

        EXPECT_LE(rgb_mad, RGB_MAD_THRESHOLD) << view.id;
        EXPECT_LE(rgb_interior_mad, RGB_INTERIOR_MAD_THRESHOLD) << view.id;
        EXPECT_LE(rgb_outliers, RGB_OUTLIER_FRACTION) << view.id;
        EXPECT_EQ(invalid_target_alpha, (size_t)0) << view.id;
        EXPECT_EQ(invalid_current_alpha, (size_t)0) << view.id;
        EXPECT_LE(segmentation_mismatch, SEGMENTATION_MISMATCH_FRACTION) << view.id;
        EXPECT_LE(segmentation_interior_mismatch, SEGMENTATION_INTERIOR_MISMATCH_FRACTION) << view.id;
        EXPECT_LE(depth_mad, DEPTH_MAD_THRESHOLD) << view.id;
        EXPECT_LE(depth_outlier_fraction, DEPTH_OUTLIER_FRACTION) << view.id;
    }

    void run_scenario(overlay_scenarios::Scenario scenario){
        RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA();
        DEVICE device;
        rlt::init(device);
        auto scenario_state = overlay_scenarios::prepare(device, scenario);
        rlt::rendering::raytracing::Renderer<SPEC> renderer;
        rlt::malloc(device, renderer);
        rlt::generate_probe_directions(device, renderer);
        rlt::init(device, renderer, scenario_state.scene, scenario_state.pool);
        overlay_scenarios::build_initial(device, renderer, scenario_state);
        rlt::update(device, renderer);

        std::array<Frame, overlay_goldens::VIEWS.size()> initial_frames;
        std::array<Frame, overlay_goldens::VIEWS.size()> target_initial_frames;
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
            initial_frames[view_i] = capture(device, renderer);
            ASSERT_TRUE(load_target_frame(
                scenario,
                overlay_goldens::CaptureState::INITIAL,
                overlay_goldens::VIEWS[view_i],
                target_initial_frames[view_i]
            ));
            compare_frame(scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i], initial_frames[view_i]);
        }

        if(overlay_goldens::capture_state_count(scenario) == 2){
            overlay_scenarios::apply_update(device, renderer, scenario_state);
            rlt::update(device, renderer);
            for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
                set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
                const auto updated = capture(device, renderer);
                Frame target_updated;
                ASSERT_TRUE(load_target_frame(
                    scenario,
                    overlay_goldens::CaptureState::UPDATED,
                    overlay_goldens::VIEWS[view_i],
                    target_updated
                ));
                expect_update_scope(scenario, target_initial_frames[view_i], target_updated, overlay_goldens::VIEWS[view_i].id);
                expect_update_scope(scenario, initial_frames[view_i], updated, overlay_goldens::VIEWS[view_i].id);
                compare_frame(scenario, overlay_goldens::CaptureState::UPDATED, overlay_goldens::VIEWS[view_i], updated);
            }
        }

        rlt::free(device, renderer);
    }

    void expect_targets_visibly_distinct(){
        RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA();
        std::array<std::array<Frame, overlay_goldens::VIEWS.size()>, overlay_scenarios::SCENARIOS.size()> targets;
        for(size_t scenario_i = 0; scenario_i < overlay_scenarios::SCENARIOS.size(); scenario_i++){
            const auto scenario = overlay_scenarios::SCENARIOS[scenario_i];
            for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
                ASSERT_TRUE(load_target_frame(
                    scenario,
                    overlay_goldens::CaptureState::INITIAL,
                    overlay_goldens::VIEWS[view_i],
                    targets[scenario_i][view_i]
                ));
            }
            for(size_t first = 0; first < overlay_goldens::VIEWS.size(); first++){
                for(size_t second = first + 1; second < overlay_goldens::VIEWS.size(); second++){
                    const auto difference = frame_difference(targets[scenario_i][first], targets[scenario_i][second]);
                    EXPECT_GT(difference.rgb, MIN_POSE_CHANGED_PIXELS)
                        << overlay_scenarios::scenario_id(scenario) << " "
                        << overlay_goldens::VIEWS[first].id << " vs " << overlay_goldens::VIEWS[second].id;
                    EXPECT_GT(difference.depth, MIN_POSE_CHANGED_PIXELS)
                        << overlay_scenarios::scenario_id(scenario) << " "
                        << overlay_goldens::VIEWS[first].id << " vs " << overlay_goldens::VIEWS[second].id;
                    if(scenario != overlay_scenarios::Scenario::SHARED_SCENE_NO_DYNAMIC){
                        EXPECT_GT(difference.segmentation, MIN_POSE_SEGMENTATION_CHANGED_PIXELS)
                            << overlay_scenarios::scenario_id(scenario) << " "
                            << overlay_goldens::VIEWS[first].id << " vs " << overlay_goldens::VIEWS[second].id;
                    }
                }
            }
        }
        for(size_t first = 0; first < targets.size(); first++){
            for(size_t second = first + 1; second < targets.size(); second++){
                const auto difference = frame_difference(targets[first][0], targets[second][0]);
                EXPECT_GT(difference.rgb, (size_t)16)
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[first]) << " vs "
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[second]);
                EXPECT_GT(difference.depth, (size_t)16)
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[first]) << " vs "
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[second]);
                EXPECT_GT(difference.segmentation, (size_t)16)
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[first]) << " vs "
                    << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[second]);
            }
        }
    }
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, MANIFEST){
    RL_TOOLS_OVERLAY_GOLDEN_REQUIRE_DATA();
}

TEST(RL_TOOLS_OVERLAY_GOLDEN_SUITE, TARGETS_ARE_VISUALLY_DISTINCT){
    expect_targets_visibly_distinct();
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
