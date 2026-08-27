#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/backends/optix/operations_cuda.h>
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include "overlay_golden_frames.h"
#include "overlay_golden_manifest.h"
#include "../../utils/utils.h"

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
using RENDERER = rlt::rendering::raytracing::Renderer<SPEC>;
using overlay_goldens::Frame;

static const std::string GOLDEN_ROOT = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/rendering_raytracing_golden";

namespace {
    struct FrameContext {
        overlay_scenarios::Scenario scenario;
        overlay_goldens::CaptureState state;
        const char* view;
    };

    bool report(bool ok, const FrameContext& context, const std::string& what){
        if(!ok){
            std::cerr << "[overlay-golden] " << overlay_scenarios::scenario_id(context.scenario)
                      << "/" << overlay_goldens::capture_state_id(context.state)
                      << "/" << context.view << ": " << what << std::endl;
        }
        return ok;
    }

    bool valid_topology(overlay_scenarios::Scenario scenario, const Frame& frame){
        const auto& scenario_definition = overlay_scenarios::definition(scenario);
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            if(overlay_goldens::camera_id_count<SPEC>(frame, camera, 0) <= SPEC::CAM_PIXELS / 2){
                return false;
            }
            const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
            for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                const uint32_t id = frame.segmentation[offset + pixel];
                if(id != golden::SEGMENTATION_BACKGROUND_ID && !overlay_goldens::id_expected_in_camera(scenario, camera, id)){
                    return false;
                }
            }
            for(const auto& placement : scenario_definition.placements){
                const bool expected = (placement.cameras & overlay_scenarios::camera_bit(camera)) != 0;
                const size_t count = overlay_goldens::camera_id_count<SPEC>(frame, camera, placement.expected_id);
                if((expected && count <= overlay_goldens::MIN_VISIBLE_ID_PIXELS) || (!expected && count != 0)){
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
        const auto& scenario_definition = overlay_scenarios::definition(scenario);
        for(const auto& placement : scenario_definition.placements){
            const auto& asset = scenario_definition.assets[placement.asset];
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

    // the encoded miss pixel (128,128,128) is unreachable for unit normals (a unit vector has a
    // component of magnitude >= 1/sqrt(3), i.e. an encoded channel <= 54 or >= 201), so normals
    // must be exactly gray on segmentation background and never gray on a hit
    bool valid_normals(const Frame& frame){
        constexpr float zero_normal[3] = {0.f, 0.f, 0.f};
        const uint32_t miss_pixel = golden::normal_rgba(zero_normal);
        for(size_t index = 0; index < frame.normals.size(); index++){
            const bool miss = frame.segmentation[index] == golden::SEGMENTATION_BACKGROUND_ID;
            if((frame.normals[index] == miss_pixel) != miss || (frame.normals[index] >> 24) != 0xFFu){
                return false;
            }
        }
        return true;
    }

    // hit pixels must carry the ego-motion flow of the fixed FLOW_PAIR_OFFSET camera pair
    // (nonzero, bounded), miss pixels the exact (0, 0) sentinel
    bool valid_flow(const Frame& frame){
        size_t nonzero = 0;
        for(size_t index = 0; index < frame.segmentation.size(); index++){
            const float u = frame.flow[index * 2 + 0];
            const float v = frame.flow[index * 2 + 1];
            if(!std::isfinite(u) || !std::isfinite(v) || std::fabs(u) > (float)SPEC::CAM_WIDTH || std::fabs(v) > (float)SPEC::CAM_HEIGHT){
                return false;
            }
            if(frame.segmentation[index] == golden::SEGMENTATION_BACKGROUND_ID){
                if(u != 0.f || v != 0.f){
                    return false;
                }
            }
            else{
                nonzero += u != 0.f || v != 0.f;
            }
        }
        return nonzero > 0;
    }

    bool validate_frame(const FrameContext& context, const Frame& frame){
        const bool rgb_varies = std::any_of(frame.rgb.begin() + 1, frame.rgb.end(), [&](uint32_t pixel){ return pixel != frame.rgb[0]; });
        const bool valid_alpha = std::all_of(frame.rgb.begin(), frame.rgb.end(), [](uint32_t pixel){ return (pixel >> 24) == 0xFFu; });
        const bool valid_depth = std::all_of(frame.depth.begin(), frame.depth.end(), [](float depth){ return std::isfinite(depth) && depth > 0; });
        bool ok = report(rgb_varies, context, "RGB output is constant");
        ok = report(valid_alpha, context, "RGB output has non-opaque alpha") && ok;
        ok = report(valid_depth, context, "depth output has non-finite or non-positive values") && ok;
        ok = report(valid_normals(frame), context, "normals output does not match the segmentation hit/miss topology") && ok;
        ok = report(valid_flow(frame), context, "flow output violates the hit/miss contract or is unbounded") && ok;
        ok = report(valid_topology(context.scenario, frame), context, "segmentation topology does not match the scenario table") && ok;
        ok = report(valid_shared_instance_pixels(context.scenario, frame), context, "shared placements are not pixel-identical across cameras") && ok;
        ok = report(valid_placement_materials(context.scenario, frame), context, "placement materials do not match the scenario assets") && ok;
        return ok;
    }

    // non-moving placements must be untouched by the update even inside affected cameras
    bool valid_static_placements(const FrameContext& context, const Frame& initial, const Frame& updated){
        const auto& scenario_definition = overlay_scenarios::definition(context.scenario);
        bool ok = true;
        for(size_t placement_i = 0; placement_i < scenario_definition.placements.size(); placement_i++){
            if(overlay_scenarios::moves(scenario_definition, placement_i)){
                continue;
            }
            const auto& placement = scenario_definition.placements[placement_i];
            for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
                if((placement.cameras & overlay_scenarios::camera_bit(camera)) == 0){
                    continue;
                }
                const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
                size_t common = 0;
                bool equal = true;
                for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
                    const size_t index = offset + pixel;
                    if(initial.segmentation[index] == placement.expected_id && updated.segmentation[index] == placement.expected_id){
                        common++;
                        equal = equal
                            && initial.rgb[index] == updated.rgb[index]
                            && initial.depth[index] == updated.depth[index]
                            && initial.normals[index] == updated.normals[index]
                            && initial.flow[index * 2 + 0] == updated.flow[index * 2 + 0]
                            && initial.flow[index * 2 + 1] == updated.flow[index * 2 + 1];
                    }
                }
                ok = report(equal && common > overlay_goldens::MIN_VISIBLE_ID_PIXELS, context,
                            "non-moving placement with ID " + std::to_string(placement.expected_id)
                            + " drifted in camera " + std::to_string(camera)) && ok;
            }
        }
        return ok;
    }

    bool validate_update_scope(const FrameContext& context, const Frame& initial, const Frame& updated){
        const auto scope = overlay_goldens::update_scope<SPEC>(context.scenario, initial, updated);
        bool ok = true;
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            ok = report(overlay_goldens::update_scope_ok(scope[camera]), context,
                        std::string("update scope violated in camera ") + std::to_string(camera)
                        + (scope[camera].should_change ? " (expected changes)" : " (expected bitwise-identical output)")) && ok;
        }
        return ok;
    }

    bool write_frame(const FrameContext& context, const Frame& frame){
        const auto paths = golden::layout::scenario_target_paths(
            GOLDEN_ROOT,
            overlay_scenarios::scenario_id(context.scenario),
            overlay_goldens::capture_state_id(context.state),
            context.view
        );
        std::filesystem::create_directories(paths.directory);
        const bool ok = golden::write_camera_grid_png(paths.rgb_png, frame.rgb.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_multi_camera_float_bin(paths.depth_bin, frame.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_depth_grid_png(paths.depth_png, frame.depth.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, frame.max_depth)
            && golden::write_multi_camera_uint32_bin(paths.segmentation_bin, frame.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_segmentation_grid_png(paths.segmentation_png, frame.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_camera_grid_png(paths.normals_png, frame.normals.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT)
            && golden::write_multi_camera_float_bin(paths.flow_bin, frame.flow.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 2);
        rlt::rendering::raytracing::detail::write_flow_grid_png<SPEC>(frame.flow.data(), paths.flow_png.c_str()); // advisory review image
        return report(ok, context, "failed to write golden files to " + paths.directory);
    }

    bool run_scenario(DEVICE& device, overlay_scenarios::Scenario scenario, Frame& centered_initial){
        std::cout << "[overlay-golden] rendering " << overlay_scenarios::scenario_id(scenario) << std::endl;
        auto state = overlay_scenarios::prepare(device, scenario);
        RENDERER renderer;
        rlt::malloc(device, renderer);
        rlt::generate_probe_directions(device, renderer);
        rlt::init(device, renderer, state.scene, state.pool);
        overlay_scenarios::build_initial(device, renderer, state);
        rlt::update(device, renderer);

        bool ok = true;
        std::array<Frame, overlay_goldens::VIEWS.size()> initial_frames;
        for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
            const FrameContext context{scenario, overlay_goldens::CaptureState::INITIAL, overlay_goldens::VIEWS[view_i].id};
            overlay_goldens::set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
            initial_frames[view_i] = overlay_goldens::capture(device, renderer);
            const bool wrote = write_frame(context, initial_frames[view_i]);
            ok = validate_frame(context, initial_frames[view_i]) && wrote && ok;
        }
        for(size_t first = 0; first < initial_frames.size(); first++){
            for(size_t second = first + 1; second < initial_frames.size(); second++){
                const bool distinct = overlay_goldens::views_distinct<SPEC>(scenario, initial_frames[first], initial_frames[second]);
                if(!distinct){
                    std::cerr << "[overlay-golden] " << overlay_scenarios::scenario_id(scenario) << ": views are not visibly distinct: "
                              << overlay_goldens::VIEWS[first].id << " and " << overlay_goldens::VIEWS[second].id << std::endl;
                }
                ok = distinct && ok;
            }
        }
        centered_initial = initial_frames[0];

        if(overlay_goldens::capture_state_count(scenario) == 2){
            overlay_scenarios::apply_update(device, renderer, state);
            rlt::update(device, renderer);
            for(size_t view_i = 0; view_i < overlay_goldens::VIEWS.size(); view_i++){
                const FrameContext context{scenario, overlay_goldens::CaptureState::UPDATED, overlay_goldens::VIEWS[view_i].id};
                overlay_goldens::set_view(device, renderer, overlay_goldens::VIEWS[view_i]);
                const auto updated = overlay_goldens::capture(device, renderer);
                const bool wrote = write_frame(context, updated);
                ok = validate_frame(context, updated)
                  && validate_update_scope(context, initial_frames[view_i], updated)
                  && valid_static_placements(context, initial_frames[view_i], updated)
                  && wrote
                  && ok;
            }
        }

        rlt::free(device, renderer);
        return ok;
    }

    bool write_manifest(){
        const std::string path = golden::layout::overlay_manifest_path(GOLDEN_ROOT);
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream output(path);
        if(!output){
            return false;
        }
        output << overlay_goldens::expected_manifest<SPEC>().dump(2) << "\n";
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
            if(!overlay_goldens::scenarios_distinct<SPEC>(centered_initial[first], centered_initial[second])){
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
