#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>

#include "golden_io.h"
#include "overlay_golden_cases.h"
#include "../../utils/utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef RL_TOOLS_TEST_DATA_PATH
#error "RL_TOOLS_TEST_DATA_PATH is required"
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using SPEC = overlay_scenarios::OverlaySpecification<T, TI>;

static const std::string GOLDEN_ROOT = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/rendering_raytracing_golden";

namespace {
    static constexpr size_t MIN_VISIBLE_PIXELS = 4;
    static constexpr size_t MIN_UPDATE_CHANGED_PIXELS = 8;
    static constexpr size_t MIN_POSE_CHANGED_PIXELS = 32;
    static constexpr size_t MIN_POSE_SEGMENTATION_CHANGED_PIXELS = 8;

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

    void set_view(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer, const overlay_goldens::View& view){
        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        const auto camera = rlt::make_camera_data(view.position, view.look_at, view.up, SPEC::COS_FOVY, aspect);
        std::array<rlt::rendering::raytracing::Camera<T>, SPEC::NUM_CAMERAS> cameras;
        cameras.fill(camera);
        cudaMemcpy(rlt::data(rlt::cameras(device, renderer)), cameras.data(), sizeof(cameras), cudaMemcpyHostToDevice);
    }

    Frame capture(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC>& renderer){
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        const size_t count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        Frame frame;
        frame.rgb.resize(count);
        frame.depth.resize(count);
        frame.segmentation.resize(count);
        cudaMemcpy(frame.rgb.data(), rlt::data(rlt::frame_buffer(device, renderer)), count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(frame.depth.data(), rlt::data(rlt::depth_buffer(device, renderer)), count * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(frame.segmentation.data(), rlt::data(rlt::segmentation_buffer(device, renderer)), count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        frame.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        return frame;
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

    size_t camera_id_count(const Frame& frame, TI camera, uint32_t id){
        const auto begin = frame.segmentation.begin() + (size_t)camera * SPEC::CAM_PIXELS;
        return static_cast<size_t>(std::count(begin, begin + SPEC::CAM_PIXELS, id));
    }

    bool id_expected_in_camera(overlay_scenarios::Scenario scenario, TI camera, uint32_t id){
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

    bool valid_topology(overlay_scenarios::Scenario scenario, const Frame& frame){
        const auto& definition = overlay_scenarios::definition(scenario);
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            if(camera_id_count(frame, camera, 0) <= SPEC::CAM_PIXELS / 2){
                return false;
            }
            const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
            for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                const uint32_t id = frame.segmentation[offset + pixel];
                if(id != golden::SEGMENTATION_BACKGROUND_ID && !id_expected_in_camera(scenario, camera, id)){
                    return false;
                }
            }
            for(const auto& placement : definition.placements){
                const bool expected = (placement.cameras & overlay_scenarios::camera_bit(camera)) != 0;
                const size_t count = camera_id_count(frame, camera, placement.expected_id);
                if((expected && count <= MIN_VISIBLE_PIXELS) || (!expected && count != 0)){
                    return false;
                }
            }
        }
        return true;
    }

    bool valid_shared_instance_pixels(overlay_scenarios::Scenario scenario, const Frame& frame){
        for(const auto& placement : overlay_scenarios::definition(scenario).placements){
            TI reference_camera = SPEC::NUM_CAMERAS;
            for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
                if((placement.cameras & overlay_scenarios::camera_bit(camera)) == 0){
                    continue;
                }
                if(reference_camera == SPEC::NUM_CAMERAS){
                    reference_camera = camera;
                    continue;
                }
                const size_t reference_offset = (size_t)reference_camera * SPEC::CAM_PIXELS;
                const size_t camera_offset = (size_t)camera * SPEC::CAM_PIXELS;
                for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                    const bool reference_hit = frame.segmentation[reference_offset + pixel] == placement.expected_id;
                    const bool camera_hit = frame.segmentation[camera_offset + pixel] == placement.expected_id;
                    if(reference_hit != camera_hit){
                        return false;
                    }
                    if(reference_hit && (frame.rgb[reference_offset + pixel] != frame.rgb[camera_offset + pixel]
                        || frame.depth[reference_offset + pixel] != frame.depth[camera_offset + pixel])){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool valid_placement_materials(overlay_scenarios::Scenario scenario, const Frame& frame){
        const auto& definition = overlay_scenarios::definition(scenario);
        for(const auto& placement : definition.placements){
            const auto& asset = definition.assets[placement.asset];
            for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
                if((placement.cameras & overlay_scenarios::camera_bit(camera)) == 0){
                    continue;
                }
                const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
                uint32_t material_rgb = 0;
                bool found = false;
                for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                    if(frame.segmentation[offset + pixel] != placement.expected_id){
                        continue;
                    }
                    if(!found){
                        material_rgb = frame.rgb[offset + pixel];
                        found = true;
                    }
                    else if(frame.rgb[offset + pixel] != material_rgb){
                        return false;
                    }
                }
                if(!found){
                    return false;
                }
                uint32_t channels[3];
                for(size_t channel = 0; channel < 3; channel++){
                    channels[channel] = (material_rgb >> (8 * channel)) & 0xFFu;
                    const int expected = (int)(asset.color[channel] * 255.0f);
                    if(std::abs((int)channels[channel] - expected) > 2){
                        return false;
                    }
                }
                const uint32_t dominant = channels[asset.dominant_channel];
                const uint32_t other_a = channels[(asset.dominant_channel + 1) % 3];
                const uint32_t other_b = channels[(asset.dominant_channel + 2) % 3];
                if(dominant <= other_a + 64u || dominant <= other_b + 64u){
                    return false;
                }
            }
        }
        return true;
    }

    bool valid_frame(overlay_scenarios::Scenario scenario, const Frame& frame){
        const bool rgb_varies = std::any_of(frame.rgb.begin() + 1, frame.rgb.end(), [&](uint32_t pixel){ return pixel != frame.rgb[0]; });
        const bool valid_alpha = std::all_of(frame.rgb.begin(), frame.rgb.end(), [](uint32_t pixel){ return (pixel >> 24) == 0xFFu; });
        const bool valid_depth = std::all_of(frame.depth.begin(), frame.depth.end(), [](float depth){ return std::isfinite(depth) && depth > 0; });
        return rgb_varies
            && valid_alpha
            && valid_depth
            && valid_topology(scenario, frame)
            && valid_shared_instance_pixels(scenario, frame)
            && valid_placement_materials(scenario, frame);
    }

    bool write_frame(
        overlay_scenarios::Scenario scenario,
        overlay_goldens::CaptureState state,
        const overlay_goldens::View& view,
        const Frame& frame
    ){
        const auto paths = golden::layout::scenario_target_paths(
            GOLDEN_ROOT,
            overlay_scenarios::scenario_id(scenario),
            overlay_goldens::capture_state_id(state),
            view.id
        );
        std::filesystem::create_directories(paths.directory);
        return golden::write_camera_grid_png(paths.rgb_png, frame.rgb.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_multi_camera_float_bin(paths.depth_bin, frame.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_depth_grid_png(paths.depth_png, frame.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, frame.max_depth)
            && golden::write_multi_camera_uint32_bin(paths.segmentation_bin, frame.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_segmentation_grid_png(paths.segmentation_png, frame.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
    }

    bool validate_update_scope(overlay_scenarios::Scenario scenario, const Frame& initial, const Frame& updated){
        const auto affected = overlay_scenarios::definition(scenario).affected_camera_mask;
        bool ok = true;
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const bool should_change = (affected & overlay_scenarios::camera_bit(camera)) != 0;
            const auto difference = camera_difference(initial, updated, camera);
            if(should_change){
                ok = ok
                  && difference.rgb > MIN_UPDATE_CHANGED_PIXELS
                  && difference.depth > MIN_UPDATE_CHANGED_PIXELS
                  && difference.segmentation > MIN_UPDATE_CHANGED_PIXELS;
            }
            else{
                ok = ok && difference.rgb == 0 && difference.depth == 0 && difference.segmentation == 0;
            }
        }
        return ok;
    }

    bool run_scenario(DEVICE& device, overlay_scenarios::Scenario scenario, Frame& centered_initial){
        std::cout << "[overlay-golden] rendering " << overlay_scenarios::scenario_id(scenario) << std::endl;
        auto state = overlay_scenarios::prepare(device, scenario);
        rlt::rendering::raytracing::Renderer<SPEC> renderer;
        rlt::malloc(device, renderer);
        rlt::generate_probe_directions(device, renderer);
        rlt::init(device, renderer, state.scene, state.pool);
        overlay_scenarios::build_initial(device, renderer, state);
        rlt::update(device, renderer);

        bool ok = true;
        std::array<Frame, overlay_goldens::VIEWS.size()> initial_frames;
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
            initial_frames[view_i] = capture(device, renderer);
            ok = valid_frame(scenario, initial_frames[view_i])
              && write_frame(scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i], initial_frames[view_i])
              && ok;
        }
        for(size_t first = 0; first < initial_frames.size(); first++){
            for(size_t second = first + 1; second < initial_frames.size(); second++){
                const auto difference = frame_difference(initial_frames[first], initial_frames[second]);
                const bool dynamic = scenario != overlay_scenarios::Scenario::SHARED_SCENE_NO_DYNAMIC;
                ok = difference.rgb > MIN_POSE_CHANGED_PIXELS
                  && difference.depth > MIN_POSE_CHANGED_PIXELS
                  && (!dynamic || difference.segmentation > MIN_POSE_SEGMENTATION_CHANGED_PIXELS)
                  && ok;
            }
        }
        centered_initial = initial_frames[0];

        if(overlay_goldens::capture_state_count(scenario) == 2){
            overlay_scenarios::apply_update(device, renderer, state);
            rlt::update(device, renderer);
            for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
                set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
                const auto updated = capture(device, renderer);
                ok = valid_frame(scenario, updated)
                  && validate_update_scope(scenario, initial_frames[view_i], updated)
                  && write_frame(scenario, overlay_goldens::CaptureState::UPDATED, overlay_goldens::VIEWS[view_i], updated)
                  && ok;
            }
        }

        rlt::free(device, renderer);
        return ok;
    }

    bool visibly_distinct(const Frame& first, const Frame& second){
        const auto difference = frame_difference(first, second);
        return difference.rgb > 16 && difference.depth > 16 && difference.segmentation > 16;
    }

    bool write_manifest(){
        const std::string path = golden::layout::overlay_manifest_path(GOLDEN_ROOT);
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream output(path);
        if(!output){
            return false;
        }
        output << "{\n"
               << "  \"schema_version\": 1,\n"
               << "  \"binary_format_version\": " << golden::MULTI_CAMERA_BINARY_VERSION << ",\n"
               << "  \"reference_backend\": \"optix\",\n"
               << "  \"num_cameras\": " << SPEC::NUM_CAMERAS << ",\n"
               << "  \"width\": " << SPEC::CAM_WIDTH << ",\n"
               << "  \"height\": " << SPEC::CAM_HEIGHT << ",\n"
               << "  \"camera_grid\": \"2x2 row-major logical cameras 0,1,2,3\",\n"
               << "  \"views\": [";
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            output << (view_i == 0 ? "" : ", ") << "\"" << overlay_goldens::VIEWS[view_i].id << "\"";
        }
        output << "],\n  \"scenarios\": [";
        for(size_t scenario_i = 0; scenario_i < overlay_scenarios::SCENARIOS.size(); scenario_i++){
            output << (scenario_i == 0 ? "" : ", ") << "\"" << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[scenario_i]) << "\"";
        }
        output << "],\n  \"capture_states\": {\n";
        for(size_t scenario_i = 0; scenario_i < overlay_scenarios::SCENARIOS.size(); scenario_i++){
            const auto scenario = overlay_scenarios::SCENARIOS[scenario_i];
            output << "    \"" << overlay_scenarios::scenario_id(scenario) << "\": [\"initial\"";
            if(overlay_goldens::capture_state_count(scenario) == 2){
                output << ", \"updated\"";
            }
            output << "]" << (scenario_i + 1 == overlay_scenarios::SCENARIOS.size() ? "\n" : ",\n");
        }
        output << "  }\n}\n";
        return output.good();
    }
}

int main(){
    std::filesystem::remove(golden::layout::overlay_manifest_path(GOLDEN_ROOT));
    DEVICE device;
    rlt::init(device);

    bool ok = true;
    std::array<Frame, overlay_scenarios::SCENARIOS.size()> centered_initial;
    for(size_t scenario_i = 0; scenario_i < overlay_scenarios::SCENARIOS.size(); scenario_i++){
        ok = run_scenario(device, overlay_scenarios::SCENARIOS[scenario_i], centered_initial[scenario_i]) && ok;
    }
    for(size_t first = 0; first < centered_initial.size(); first++){
        for(size_t second = first + 1; second < centered_initial.size(); second++){
            if(!visibly_distinct(centered_initial[first], centered_initial[second])){
                std::cerr << "[overlay-golden] scenarios are not visibly distinct: "
                          << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[first]) << " and "
                          << overlay_scenarios::scenario_id(overlay_scenarios::SCENARIOS[second]) << std::endl;
                ok = false;
            }
        }
    }
    if(ok){
        ok = write_manifest();
    }
    if(!ok){
        std::cerr << "[overlay-golden] FAILED" << std::endl;
        return 1;
    }
    std::cout << "[overlay-golden] done: " << golden::layout::overlay_directory(GOLDEN_ROOT) << std::endl;
    return 0;
}
