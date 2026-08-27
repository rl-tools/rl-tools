#ifndef TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_FRAMES_H
#define TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_FRAMES_H

// Shared harness for the overlay golden generator, the per-backend comparators, and the corpus
// validation test: frame capture plumbing plus the invariant constants and predicates that
// generation and comparison must agree on.

#include "golden_io.h"
#include "golden_layout.h"
#include "overlay_golden_cases.h"
#include "render_copy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace overlay_goldens {
    static constexpr std::size_t MIN_VISIBLE_ID_PIXELS = 4;
    static constexpr std::size_t MIN_UPDATE_CHANGED_PIXELS = 8;
    static constexpr std::size_t MIN_POSE_CHANGED_PIXELS = 32;
    static constexpr std::size_t MIN_POSE_SEGMENTATION_CHANGED_PIXELS = 8;
    static constexpr std::size_t MIN_SCENARIO_DISTINCT_PIXELS = 16;
    // fixed shutter-open camera offset for the flow modality: every capture renders with the
    // close camera at the view pose and the open camera translated by this world-frame delta,
    // so the flow goldens carry nonzero ego-motion flow. Corpus surface — changing it changes
    // every flow golden.
    static constexpr float FLOW_PAIR_OFFSET[3] = {0.05f, 0.08f, 0.03f};

    struct Frame {
        std::vector<std::uint32_t> rgb;
        std::vector<float> depth;
        std::vector<std::uint32_t> segmentation;
        std::vector<std::uint32_t> normals; // golden::normal_rgba-encoded, reuses the rgb grid/diff machinery
        std::vector<float> flow;            // 2 per pixel, raw float backward flow (machine target: flow.bin)
        float max_depth = 0;
    };

    struct DifferenceCounts {
        std::size_t rgb = 0;
        std::size_t depth = 0;
        std::size_t segmentation = 0;
        std::size_t normals = 0;
        std::size_t flow = 0;
    };

    template <typename DEVICE, typename RENDERER>
    void set_view(DEVICE& device, RENDERER& renderer, const View& view){
        using SPEC = typename RENDERER::SPEC;
        using T = typename SPEC::T;
        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        const auto camera = rl_tools::make_camera_data(view.position, view.look_at, view.up, SPEC::CONFIG::FOV, aspect);
        std::array<rl_tools::rendering::raytracing::Camera<T>, SPEC::NUM_CAMERAS> cameras;
        cameras.fill(camera);
        golden::copy_in(device, renderer.device, cameras.data(), rl_tools::cameras(device, renderer));
        if constexpr (SPEC::HAS_CAMERA_PAIR){
            T position_open[3], look_at_open[3];
            for(int dim_i = 0; dim_i < 3; dim_i++){
                position_open[dim_i] = view.position[dim_i] + (T)FLOW_PAIR_OFFSET[dim_i];
                look_at_open[dim_i] = view.look_at[dim_i] + (T)FLOW_PAIR_OFFSET[dim_i];
            }
            const auto camera_open = rl_tools::make_camera_data(position_open, look_at_open, view.up, SPEC::CONFIG::FOV, aspect);
            cameras.fill(camera_open);
            golden::copy_in(device, renderer.device, cameras.data(), rl_tools::cameras_open(device, renderer));
        }
    }

    template <typename DEVICE, typename RENDERER>
    Frame capture(DEVICE& device, RENDERER& renderer){
        using SPEC = typename RENDERER::SPEC;
        rl_tools::render(device, renderer);
        rl_tools::synchronize(device, renderer);
        const std::size_t count = (std::size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        Frame frame;
        golden::copy_out(renderer.device, device, rl_tools::frame_buffer(device, renderer), frame.rgb);
        golden::copy_out(renderer.device, device, rl_tools::depth_buffer(device, renderer), frame.depth);
        golden::copy_out(renderer.device, device, rl_tools::segmentation_buffer(device, renderer), frame.segmentation);
        std::vector<float> normals_raw;
        golden::copy_out(renderer.device, device, rl_tools::normals_buffer(device, renderer), normals_raw);
        frame.normals.resize(count);
        golden::colorize_normals(normals_raw.data(), count, frame.normals.data());
        golden::copy_out(renderer.device, device, rl_tools::flow_buffer(device, renderer), frame.flow);
        frame.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        return frame;
    }

    template <typename SPEC>
    DifferenceCounts camera_difference(const Frame& first, const Frame& second, std::size_t camera){
        DifferenceCounts difference;
        const std::size_t offset = camera * SPEC::CAM_PIXELS;
        for(std::size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
            const std::size_t index = offset + pixel;
            difference.rgb += first.rgb[index] != second.rgb[index];
            difference.depth += first.depth[index] != second.depth[index];
            difference.segmentation += first.segmentation[index] != second.segmentation[index];
            difference.normals += first.normals[index] != second.normals[index];
            difference.flow += first.flow[index * 2 + 0] != second.flow[index * 2 + 0]
                            || first.flow[index * 2 + 1] != second.flow[index * 2 + 1];
        }
        return difference;
    }

    template <typename SPEC>
    DifferenceCounts frame_difference(const Frame& first, const Frame& second){
        DifferenceCounts difference;
        for(std::size_t camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const auto camera_counts = camera_difference<SPEC>(first, second, camera);
            difference.rgb += camera_counts.rgb;
            difference.depth += camera_counts.depth;
            difference.segmentation += camera_counts.segmentation;
            difference.normals += camera_counts.normals;
            difference.flow += camera_counts.flow;
        }
        return difference;
    }

    template <typename SPEC>
    std::size_t camera_id_count(const Frame& frame, std::size_t camera, std::uint32_t id){
        const std::size_t offset = camera * SPEC::CAM_PIXELS;
        std::size_t count = 0;
        for(std::size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
            count += frame.segmentation[offset + pixel] == id;
        }
        return count;
    }

    inline std::vector<std::uint32_t> expected_ids(overlay_scenarios::Scenario scenario, std::size_t camera){
        std::vector<std::uint32_t> ids = {0};
        for(const auto& placement : overlay_scenarios::definition(scenario).placements){
            if((placement.cameras & overlay_scenarios::camera_bit(camera)) != 0){
                ids.push_back(placement.expected_id);
            }
        }
        return ids;
    }

    inline bool id_expected_in_camera(overlay_scenarios::Scenario scenario, std::size_t camera, std::uint32_t id){
        if(id == 0){
            return true;
        }
        for(const auto& placement : overlay_scenarios::definition(scenario).placements){
            if(placement.expected_id == id && (placement.cameras & overlay_scenarios::camera_bit(camera)) != 0){
                return true;
            }
        }
        return false;
    }

    struct CameraUpdateScope {
        bool should_change = false;
        DifferenceCounts difference;
    };

    template <typename SPEC>
    std::array<CameraUpdateScope, overlay_scenarios::NUM_CAMERAS> update_scope(overlay_scenarios::Scenario scenario, const Frame& initial, const Frame& updated){
        const auto affected = overlay_scenarios::affected_camera_mask(overlay_scenarios::definition(scenario));
        std::array<CameraUpdateScope, overlay_scenarios::NUM_CAMERAS> scope;
        for(std::size_t camera = 0; camera < overlay_scenarios::NUM_CAMERAS; camera++){
            scope[camera].should_change = (affected & overlay_scenarios::camera_bit(camera)) != 0;
            scope[camera].difference = camera_difference<SPEC>(initial, updated, camera);
        }
        return scope;
    }

    // normals and flow only participate in the bitwise-identical direction: normals is
    // piecewise constant per face and flow depends on the camera pair, so an affected camera
    // need not change them, but an unaffected camera must reproduce them exactly
    inline bool update_scope_ok(const CameraUpdateScope& scope){
        return scope.should_change
            ? scope.difference.rgb > MIN_UPDATE_CHANGED_PIXELS
                && scope.difference.depth > MIN_UPDATE_CHANGED_PIXELS
                && scope.difference.segmentation > MIN_UPDATE_CHANGED_PIXELS
            : scope.difference.rgb == 0
                && scope.difference.depth == 0
                && scope.difference.segmentation == 0
                && scope.difference.normals == 0
                && scope.difference.flow == 0;
    }

    template <typename SPEC>
    bool views_distinct(overlay_scenarios::Scenario scenario, const Frame& first, const Frame& second){
        const auto difference = frame_difference<SPEC>(first, second);
        const bool dynamic = !overlay_scenarios::definition(scenario).placements.empty();
        return difference.rgb > MIN_POSE_CHANGED_PIXELS
            && difference.depth > MIN_POSE_CHANGED_PIXELS
            && (!dynamic || difference.segmentation > MIN_POSE_SEGMENTATION_CHANGED_PIXELS);
    }

    template <typename SPEC>
    bool scenarios_distinct(const Frame& first, const Frame& second){
        const auto difference = frame_difference<SPEC>(first, second);
        return difference.rgb > MIN_SCENARIO_DISTINCT_PIXELS
            && difference.depth > MIN_SCENARIO_DISTINCT_PIXELS
            && difference.segmentation > MIN_SCENARIO_DISTINCT_PIXELS;
    }

    template <typename SPEC>
    bool load_target_frame(
        const std::string& golden_root,
        overlay_scenarios::Scenario scenario,
        CaptureState state,
        const View& view,
        Frame& target
    ){
        const auto paths = golden::layout::scenario_target_paths(
            golden_root,
            overlay_scenarios::scenario_id(scenario),
            capture_state_id(state),
            view.id
        );
        return golden::load_camera_grid_png(paths.rgb_png, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.rgb)
            && golden::load_multi_camera_float_bin(paths.depth_bin, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.depth)
            && golden::load_multi_camera_uint32_bin(paths.segmentation_bin, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.segmentation)
            && golden::load_camera_grid_png(paths.normals_png, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.normals)
            && golden::load_multi_camera_float_bin(paths.flow_bin, SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, target.flow, 2);
    }
}

#endif
