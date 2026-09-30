// Validates the published overlay golden corpus itself, independent of any rendering backend:
// manifest content, cross-view/cross-scenario distinctness, update scope between the published
// initial/updated directories (the mixed-revision detector), and review-PNG consistency.
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>

#include "overlay_golden_frames.h"
#include "overlay_golden_manifest.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef RL_TOOLS_TEST_DATA_PATH
#define RL_TOOLS_OVERLAY_CORPUS_TEST_DATA_PATH RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)
#else
#define RL_TOOLS_OVERLAY_CORPUS_TEST_DATA_PATH "tests/data"
#endif

namespace rlt = rl_tools;

using T = float;
using TI = typename rlt::devices::DefaultCPU::index_t;
using SPEC = overlay_scenarios::OverlaySpecification<T, TI>;
using overlay_goldens::Frame;

static const std::string GOLDEN_ROOT = golden::layout::goldens_root(RL_TOOLS_OVERLAY_CORPUS_TEST_DATA_PATH);
static const std::string MANIFEST_PATH = golden::layout::manifest_path(golden::layout::overlay_directory(GOLDEN_ROOT));

namespace {
    bool manifest_available(){
        return std::filesystem::is_regular_file(MANIFEST_PATH);
    }

#if defined(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
#define RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA() ASSERT_TRUE(manifest_available()) << "required overlay goldens are missing at " << golden::layout::overlay_directory(GOLDEN_ROOT)
#else
#define RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA() if(!manifest_available()){ GTEST_SKIP() << "overlay goldens not found at " << golden::layout::overlay_directory(GOLDEN_ROOT); }
#endif

    Frame load_target(overlay_scenarios::Scenario scenario, overlay_goldens::CaptureState state, const overlay_goldens::View& view){
        Frame target;
        EXPECT_TRUE(overlay_goldens::load_target_frame<SPEC>(GOLDEN_ROOT, scenario, state, view, target))
            << overlay_scenarios::scenario_id(scenario) << " " << overlay_goldens::capture_state_id(state) << " " << view.id;
        return target;
    }
}

TEST(RENDERING_RAYTRACING_OVERLAY_GOLDEN_CORPUS, MANIFEST){
    RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA();
    std::ifstream input(MANIFEST_PATH);
    ASSERT_TRUE(input.is_open());
    nlohmann::json manifest;
    ASSERT_NO_THROW(input >> manifest);
    EXPECT_EQ(manifest, overlay_goldens::expected_manifest<SPEC>());
}

TEST(RENDERING_RAYTRACING_OVERLAY_GOLDEN_CORPUS, TARGETS_ARE_VISUALLY_DISTINCT){
    RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA();
    std::array<std::array<Frame, overlay_goldens::VIEWS.size()>, overlay_scenarios::SCENARIOS.size()> targets;
    for(size_t scenario_i = 0; scenario_i < overlay_scenarios::SCENARIOS.size(); scenario_i++){
        const auto scenario = overlay_scenarios::SCENARIOS[scenario_i];
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            targets[scenario_i][view_i] = load_target(scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i]);
        }
        if(::testing::Test::HasFailure()){
            return;
        }
        for(size_t first = 0; first < overlay_goldens::VIEWS.size(); first++){
            for(size_t second = first + 1; second < overlay_goldens::VIEWS.size(); second++){
                EXPECT_TRUE(overlay_goldens::views_distinct<SPEC>(scenario, targets[scenario_i][first], targets[scenario_i][second]))
                    << overlay_scenarios::scenario_id(scenario) << " "
                    << overlay_goldens::VIEWS[first].id << " vs " << overlay_goldens::VIEWS[second].id;
            }
        }
    }
    for(size_t first = 0; first < targets.size(); first++){
        for(size_t second = first + 1; second < targets.size(); second++){
            EXPECT_TRUE(overlay_goldens::scenarios_distinct<SPEC>(targets[first][0], targets[second][0]))
                << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[first]) << " vs "
                << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[second]);
        }
    }
}

TEST(RENDERING_RAYTRACING_OVERLAY_GOLDEN_CORPUS, TARGET_UPDATE_SCOPE){
    RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA();
    for(const auto scenario : overlay_scenarios::SCENARIOS){
        if(overlay_goldens::capture_state_count(scenario) != 2){
            continue;
        }
        for(const auto& view : overlay_goldens::VIEWS){
            const Frame initial = load_target(scenario, overlay_goldens::CaptureState::INITIAL, view);
            const Frame updated = load_target(scenario, overlay_goldens::CaptureState::UPDATED, view);
            if(::testing::Test::HasFailure()){
                return;
            }
            const auto scope = overlay_goldens::update_scope<SPEC>(scenario, initial, updated);
            for(size_t camera = 0; camera < overlay_scenarios::NUM_CAMERAS; camera++){
                EXPECT_TRUE(overlay_goldens::update_scope_ok(scope[camera]))
                    << overlay_scenarios::scenario_id(scenario) << " " << view.id << " camera " << camera
                    << (scope[camera].should_change ? " (expected changes)" : " (expected bitwise-identical goldens)");
            }
        }
    }
}

TEST(RENDERING_RAYTRACING_OVERLAY_GOLDEN_CORPUS, SEGMENTATION_REVIEW_PNG_MATCHES_BIN){
    RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA();
    for(const auto scenario : overlay_scenarios::SCENARIOS){
        for(std::size_t state_i = 0; state_i < overlay_goldens::capture_state_count(scenario); state_i++){
            const auto state = state_i == 0 ? overlay_goldens::CaptureState::INITIAL : overlay_goldens::CaptureState::UPDATED;
            for(const auto& view : overlay_goldens::VIEWS){
                const std::string modality = golden::layout::modality_directory(golden::layout::overlay_view_directory(GOLDEN_ROOT, overlay_scenarios::scenario_id(scenario), overlay_goldens::capture_state_id(state), view.id), "segmentation");
                std::vector<uint32_t> segmentation;
                ASSERT_TRUE(golden::load_multi_camera_uint32_bin(golden::layout::golden_bin(modality), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, segmentation)) << modality;
                std::vector<uint32_t> png_pixels;
                ASSERT_TRUE(golden::load_camera_grid_png(golden::layout::golden_png(modality), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, png_pixels)) << modality;
                std::vector<uint32_t> expected(segmentation.size());
                golden::colorize_segmentation(segmentation.data(), segmentation.size(), expected.data());
                EXPECT_EQ(png_pixels, expected)
                    << overlay_scenarios::scenario_id(scenario) << " " << overlay_goldens::capture_state_id(state) << " " << view.id;
            }
        }
    }
}

// the flow status review images must be exactly the pinned false color of their binaries
TEST(RENDERING_RAYTRACING_OVERLAY_GOLDEN_CORPUS, FLOW_STATUS_REVIEW_PNG_MATCHES_BIN){
    RL_TOOLS_OVERLAY_CORPUS_REQUIRE_DATA();
    for(const auto scenario : overlay_scenarios::SCENARIOS){
        for(std::size_t state_i = 0; state_i < overlay_goldens::capture_state_count(scenario); state_i++){
            const auto state = state_i == 0 ? overlay_goldens::CaptureState::INITIAL : overlay_goldens::CaptureState::UPDATED;
            for(const auto& view : overlay_goldens::VIEWS){
                const std::string view_directory = golden::layout::overlay_view_directory(GOLDEN_ROOT, overlay_scenarios::scenario_id(scenario), overlay_goldens::capture_state_id(state), view.id);
                for(const char* name : {"flow-status", "flow-forward-status"}){
                    const std::string modality = golden::layout::modality_directory(view_directory, name);
                    const std::string bin_path = golden::layout::golden_bin(modality);
                    const std::string png_path = golden::layout::golden_png(modality);
                    std::vector<uint32_t> status;
                    ASSERT_TRUE(golden::load_multi_camera_uint32_bin(bin_path, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, status)) << bin_path;
                    std::vector<uint32_t> png_pixels;
                    ASSERT_TRUE(golden::load_camera_grid_png(png_path, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, png_pixels)) << png_path;
                    std::vector<uint32_t> expected(status.size());
                    golden::colorize_flow_status(status.data(), status.size(), expected.data());
                    EXPECT_EQ(png_pixels, expected) << png_path;
                }
            }
        }
    }
}
