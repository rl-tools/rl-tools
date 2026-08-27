#include <rl_tools/operations/cpu.h>

#ifdef RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_TEST_ACTIVE_BACKEND
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#include <gtest/gtest.h>

#include "../overlay_scenario_cases.h"
#include "../render_copy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#ifndef RL_TOOLS_OVERLAY_SCENARIOS_SUITE
#define RL_TOOLS_OVERLAY_SCENARIOS_SUITE RENDERING_RAYTRACING_OVERLAY_SCENARIOS_GENERIC
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using rlt::rendering::raytracing::AssetHandle;
using rlt::rendering::raytracing::AssetPool;
using rlt::rendering::raytracing::Scene;

namespace {
    constexpr TI NUM_CAMERAS = static_cast<TI>(overlay_scenarios::NUM_CAMERAS);
    using StaticSpec = overlay_scenarios::StaticSpecification<T, TI>;
    using OverlaySpec = overlay_scenarios::OverlaySpecification<T, TI>;
#ifdef RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_TEST_ACTIVE_BACKEND
    using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
    using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif

    template <typename SPEC>
    struct RendererOwner{
        DEVICE& device;
        rlt::rendering::raytracing::Renderer<SPEC, BACKEND> renderer;

        explicit RendererOwner(DEVICE& device): device(device){
            rlt::malloc(device, renderer);
            rlt::generate_probe_directions(device, renderer);
        }
        ~RendererOwner(){
            rlt::free(device, renderer);
        }
        RendererOwner(const RendererOwner&) = delete;
        RendererOwner& operator=(const RendererOwner&) = delete;
    };

    template <typename SPEC>
    void set_identical_cameras(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        const T position[3] = {0, 0, 0};
        const T look_at[3] = {1, 0, 0};
        const T up[3] = {0, 0, 1};
        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        const auto camera = rlt::make_camera_data(position, look_at, up, SPEC::CONFIG::FOV, aspect);
        std::array<rlt::rendering::raytracing::Camera<T>, SPEC::NUM_CAMERAS> cameras;
        cameras.fill(camera);
        golden::copy_in(device, renderer.device, cameras.data(), rlt::cameras(device, renderer));
    }

    template <typename SPEC>
    struct Frame{
        std::vector<uint32_t> rgb;
        std::vector<float> depth;
        std::vector<uint32_t> segmentation;
    };

    template <typename SPEC>
    Frame<SPEC> capture(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        Frame<SPEC> frame;
        golden::copy_out(renderer.device, device, rlt::frame_buffer(device, renderer), frame.rgb);
        golden::copy_out(renderer.device, device, rlt::depth_buffer(device, renderer), frame.depth);
        golden::copy_out(renderer.device, device, rlt::segmentation_buffer(device, renderer), frame.segmentation);
        return frame;
    }

    template <typename SPEC, typename ELEMENT>
    bool camera_slice_equal(const std::vector<ELEMENT>& lhs, TI lhs_camera, const std::vector<ELEMENT>& rhs, TI rhs_camera){
        const size_t lhs_offset = (size_t)lhs_camera * SPEC::CAM_PIXELS;
        const size_t rhs_offset = (size_t)rhs_camera * SPEC::CAM_PIXELS;
        return std::equal(lhs.begin() + lhs_offset, lhs.begin() + lhs_offset + SPEC::CAM_PIXELS, rhs.begin() + rhs_offset);
    }

    template <typename SPEC>
    bool camera_equal(const Frame<SPEC>& lhs, TI lhs_camera, const Frame<SPEC>& rhs, TI rhs_camera){
        return camera_slice_equal<SPEC>(lhs.rgb, lhs_camera, rhs.rgb, rhs_camera)
            && camera_slice_equal<SPEC>(lhs.depth, lhs_camera, rhs.depth, rhs_camera)
            && camera_slice_equal<SPEC>(lhs.segmentation, lhs_camera, rhs.segmentation, rhs_camera);
    }

    template <typename SPEC>
    void expect_camera_changed(const Frame<SPEC>& before, const Frame<SPEC>& after, TI camera){
        EXPECT_FALSE(camera_slice_equal<SPEC>(before.rgb, camera, after.rgb, camera));
        EXPECT_FALSE(camera_slice_equal<SPEC>(before.depth, camera, after.depth, camera));
        EXPECT_FALSE(camera_slice_equal<SPEC>(before.segmentation, camera, after.segmentation, camera));
    }

    struct InstanceObservation{
        size_t count = 0;
        double column_sum = 0;
        double row_sum = 0;
        double depth_sum = 0;
        uint32_t rgb = 0;
        bool rgb_uniform = true;

        double column() const { return count == 0 ? 0 : column_sum / count; }
        double row() const { return count == 0 ? 0 : row_sum / count; }
        double mean_depth() const { return count == 0 ? 0 : depth_sum / count; }
    };

    template <typename SPEC>
    InstanceObservation observe(const Frame<SPEC>& frame, TI camera, uint32_t id){
        InstanceObservation observation;
        const size_t camera_offset = (size_t)camera * SPEC::CAM_PIXELS;
        for(TI row = 0; row < SPEC::CAM_HEIGHT; row++){
            for(TI column = 0; column < SPEC::CAM_WIDTH; column++){
                const size_t index = camera_offset + (size_t)row * SPEC::CAM_WIDTH + column;
                if(frame.segmentation[index] != id){
                    continue;
                }
                if(observation.count == 0){
                    observation.rgb = frame.rgb[index];
                }
                else{
                    observation.rgb_uniform = observation.rgb_uniform && observation.rgb == frame.rgb[index];
                }
                observation.count++;
                observation.column_sum += column;
                observation.row_sum += row;
                observation.depth_sum += frame.depth[index];
            }
        }
        return observation;
    }

    template <typename SPEC>
    void expect_raw_frame_valid(const Frame<SPEC>& frame){
        size_t invalid_alpha = 0;
        size_t invalid_depth = 0;
        for(size_t index = 0; index < frame.rgb.size(); index++){
            invalid_alpha += (frame.rgb[index] >> 24) != 0xFFu;
            invalid_depth += !std::isfinite(frame.depth[index]) || frame.depth[index] <= 0;
        }
        EXPECT_EQ(invalid_alpha, (size_t)0);
        EXPECT_EQ(invalid_depth, (size_t)0);
    }

    template <typename SPEC>
    InstanceObservation expect_visible(const Frame<SPEC>& frame, TI camera, uint32_t id, int dominant_channel){
        const auto observation = observe(frame, camera, id);
        EXPECT_GT(observation.count, (size_t)4);
        EXPECT_TRUE(observation.rgb_uniform);
        if(observation.count > 0){
            const uint32_t dominant = (observation.rgb >> (8 * dominant_channel)) & 0xFFu;
            const uint32_t other_a = (observation.rgb >> (8 * ((dominant_channel + 1) % 3))) & 0xFFu;
            const uint32_t other_b = (observation.rgb >> (8 * ((dominant_channel + 2) % 3))) & 0xFFu;
            EXPECT_GT(dominant, other_a + 64u);
            EXPECT_GT(dominant, other_b + 64u);
            EXPECT_LT(observation.mean_depth(), 6.0);
        }
        return observation;
    }

    template <typename SPEC>
    void expect_absent(const Frame<SPEC>& frame, TI camera, uint32_t id){
        EXPECT_EQ(observe(frame, camera, id).count, (size_t)0);
    }

    template <typename SPEC>
    bool instance_mask_equal(const Frame<SPEC>& lhs, const Frame<SPEC>& rhs, TI camera, uint32_t id){
        const size_t offset = (size_t)camera * SPEC::CAM_PIXELS;
        for(TI pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
            if((lhs.segmentation[offset + pixel] == id) != (rhs.segmentation[offset + pixel] == id)){
                return false;
            }
        }
        return true;
    }

    template <typename SPEC>
    void expect_instance_moved(const Frame<SPEC>& before, const Frame<SPEC>& after, TI camera, uint32_t id){
        const auto before_observation = observe(before, camera, id);
        const auto after_observation = observe(after, camera, id);
        EXPECT_GT(before_observation.count, (size_t)4);
        EXPECT_GT(after_observation.count, (size_t)4);
        EXPECT_FALSE(instance_mask_equal(before, after, camera, id));
        const double column_delta = before_observation.column() - after_observation.column();
        const double row_delta = before_observation.row() - after_observation.row();
        EXPECT_GT(std::sqrt(column_delta * column_delta + row_delta * row_delta), 2.0);
    }

    template <typename SPEC>
    void expect_same_instance_pixels(const Frame<SPEC>& first_frame, TI first_camera, const Frame<SPEC>& second_frame, TI second_camera, uint32_t id){
        bool equal = true;
        const size_t first_offset = (size_t)first_camera * SPEC::CAM_PIXELS;
        const size_t second_offset = (size_t)second_camera * SPEC::CAM_PIXELS;
        for(TI pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++){
            const bool first_hit = first_frame.segmentation[first_offset + pixel] == id;
            const bool second_hit = second_frame.segmentation[second_offset + pixel] == id;
            equal = equal && first_hit == second_hit;
            if(first_hit && second_hit){
                equal = equal && first_frame.rgb[first_offset + pixel] == second_frame.rgb[second_offset + pixel];
                equal = equal && first_frame.depth[first_offset + pixel] == second_frame.depth[second_offset + pixel];
            }
        }
        EXPECT_TRUE(equal);
    }

    template <typename SPEC>
    void expect_same_instance_pixels(const Frame<SPEC>& frame, TI first_camera, TI second_camera, uint32_t id){
        expect_same_instance_pixels(frame, first_camera, frame, second_camera, id);
    }

    template <typename SPEC>
    void expect_maps_to_asset(DEVICE& device, const Scene& scene, const AssetPool& pool, const rlt::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, uint32_t id, AssetHandle asset){
        const auto* object = rlt::segmentation_object(device, scene, pool, renderer, id);
        ASSERT_NE(object, nullptr);
        EXPECT_EQ(object, &pool.assemblies[asset.index].objects[0]);
    }

    template <typename SPEC>
    void expect_static_scene(DEVICE& device, const Scene& scene, const AssetPool& pool, const rlt::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const Frame<SPEC>& frame){
        const auto* object = rlt::segmentation_object(device, scene, pool, renderer, 0);
        ASSERT_NE(object, nullptr);
        EXPECT_EQ(object, &scene.objects[0]);
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            const auto background = observe(frame, camera, 0);
            EXPECT_GT(background.count, SPEC::CAM_PIXELS / 2);
            EXPECT_GT(background.mean_depth(), 7.0);
        }
    }

    template <typename SPEC>
    void expect_repeat_after_noop_update(DEVICE& device, rlt::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const Frame<SPEC>& expected){
        rlt::update(device, renderer);
        const auto repeated = capture(device, renderer);
        expect_raw_frame_valid(repeated);
        for(TI camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            EXPECT_TRUE(camera_equal(expected, camera, repeated, camera));
        }
    }
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, SHARED_SCENE_NO_DYNAMIC_OBJECTS){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::SHARED_SCENE_NO_DYNAMIC);
    auto& scene = state.scene;
    auto& pool = state.pool;
    RendererOwner<StaticSpec> baseline_owner(device);
    rlt::init(device, baseline_owner.renderer, scene);
    set_identical_cameras(device, baseline_owner.renderer);
    RendererOwner<OverlaySpec> overlay_owner(device);
    rlt::init(device, overlay_owner.renderer, scene, pool);
    set_identical_cameras(device, overlay_owner.renderer);
    overlay_scenarios::build_initial(device, overlay_owner.renderer, state);
    rlt::update(device, overlay_owner.renderer);

    const auto baseline = capture(device, baseline_owner.renderer);
    const auto empty_overlay = capture(device, overlay_owner.renderer);
    expect_raw_frame_valid(baseline);
    expect_raw_frame_valid(empty_overlay);
    expect_static_scene(device, scene, pool, baseline_owner.renderer, baseline);
    expect_static_scene(device, scene, pool, overlay_owner.renderer, empty_overlay);
    EXPECT_EQ(baseline.rgb, empty_overlay.rgb);
    EXPECT_EQ(baseline.depth, empty_overlay.depth);
    EXPECT_EQ(baseline.segmentation, empty_overlay.segmentation);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        EXPECT_TRUE(camera_equal(baseline, 0, baseline, camera));
    }
    expect_repeat_after_noop_update(device, overlay_owner.renderer, empty_overlay);
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, ALL_DYNAMIC_OBJECTS_SHARE_MESHES_AND_TRANSFORMS){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::ALL_SHARED_MESH_TRANSFORM);
    auto& scene = state.scene;
    auto& pool = state.pool;
    const auto red = state.assets[0];
    const auto green = state.assets[1];
    RendererOwner<OverlaySpec> owner(device);
    rlt::init(device, owner.renderer, scene, pool);
    set_identical_cameras(device, owner.renderer);
    overlay_scenarios::build_initial(device, owner.renderer, state);

    const uint32_t red_id = state.ids[0];
    const uint32_t green_id = state.ids[1];
    expect_maps_to_asset(device, scene, pool, owner.renderer, red_id, red);
    expect_maps_to_asset(device, scene, pool, owner.renderer, green_id, green);

    rlt::update(device, owner.renderer);
    const auto first = capture(device, owner.renderer);
    expect_raw_frame_valid(first);
    expect_static_scene(device, scene, pool, owner.renderer, first);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        expect_visible(first, camera, red_id, 0);
        expect_visible(first, camera, green_id, 1);
        EXPECT_TRUE(camera_equal(first, 0, first, camera));
    }
    expect_repeat_after_noop_update(device, owner.renderer, first);

    overlay_scenarios::apply_update(device, owner.renderer, state);
    rlt::update(device, owner.renderer);
    const auto second = capture(device, owner.renderer);
    expect_raw_frame_valid(second);
    expect_static_scene(device, scene, pool, owner.renderer, second);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        expect_camera_changed(first, second, camera);
        expect_instance_moved(first, second, camera, red_id);
        expect_visible(second, camera, red_id, 0);
        expect_visible(second, camera, green_id, 1);
        expect_same_instance_pixels(first, camera, second, camera, green_id);
        expect_same_instance_pixels(second, 0, camera, red_id);
        expect_same_instance_pixels(second, 0, camera, green_id);
        EXPECT_TRUE(camera_equal(second, 0, second, camera));
    }
    expect_repeat_after_noop_update(device, owner.renderer, second);
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, SOME_DYNAMIC_OBJECTS_SHARE_MESHES_AND_TRANSFORMS){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::PARTIALLY_SHARED_MESH_TRANSFORM);
    auto& scene = state.scene;
    auto& pool = state.pool;
    const auto shared = state.assets[0];
    std::array<AssetHandle, NUM_CAMERAS> private_assets;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        private_assets[camera] = state.assets[1 + camera];
    }
    RendererOwner<OverlaySpec> owner(device);
    rlt::init(device, owner.renderer, scene, pool);
    set_identical_cameras(device, owner.renderer);
    overlay_scenarios::build_initial(device, owner.renderer, state);
    const uint32_t shared_id = state.ids[0];
    std::array<uint32_t, NUM_CAMERAS> private_ids;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        private_ids[camera] = state.ids[1 + camera];
        expect_maps_to_asset(device, scene, pool, owner.renderer, private_ids[camera], private_assets[camera]);
    }
    expect_maps_to_asset(device, scene, pool, owner.renderer, shared_id, shared);

    rlt::update(device, owner.renderer);
    const auto first = capture(device, owner.renderer);
    expect_raw_frame_valid(first);
    expect_static_scene(device, scene, pool, owner.renderer, first);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        if(camera < 2){
            expect_visible(first, camera, shared_id, 0);
        }
        else{
            expect_absent(first, camera, shared_id);
        }
        expect_visible(first, camera, private_ids[camera], 2);
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(first, camera, private_ids[other]);
            }
        }
    }
    expect_same_instance_pixels(first, 0, 1, shared_id);
    expect_repeat_after_noop_update(device, owner.renderer, first);

    overlay_scenarios::apply_update(device, owner.renderer, state);
    rlt::update(device, owner.renderer);
    const auto second = capture(device, owner.renderer);
    expect_static_scene(device, scene, pool, owner.renderer, second);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        if(camera < 2){
            expect_camera_changed(first, second, camera);
            expect_instance_moved(first, second, camera, shared_id);
            expect_visible(second, camera, shared_id, 0);
        }
        else{
            EXPECT_TRUE(camera_equal(first, camera, second, camera));
            expect_absent(second, camera, shared_id);
        }
        expect_visible(second, camera, private_ids[camera], 2);
        expect_same_instance_pixels(first, camera, second, camera, private_ids[camera]);
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(second, camera, private_ids[other]);
            }
        }
    }
    expect_same_instance_pixels(second, 0, 1, shared_id);
    expect_repeat_after_noop_update(device, owner.renderer, second);
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, ALL_DYNAMIC_OBJECTS_SHARE_ONLY_MESHES){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::SHARED_MESH_INDIVIDUAL_TRANSFORM);
    auto& scene = state.scene;
    auto& pool = state.pool;
    const auto shared_mesh = state.assets[0];
    RendererOwner<OverlaySpec> owner(device);
    rlt::init(device, owner.renderer, scene, pool);
    set_identical_cameras(device, owner.renderer);
    overlay_scenarios::build_initial(device, owner.renderer, state);
    std::array<uint32_t, NUM_CAMERAS> ids;
    std::array<InstanceObservation, NUM_CAMERAS> observations;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        ids[camera] = state.ids[camera];
        expect_maps_to_asset(device, scene, pool, owner.renderer, ids[camera], shared_mesh);
    }

    rlt::update(device, owner.renderer);
    const auto first = capture(device, owner.renderer);
    expect_raw_frame_valid(first);
    expect_static_scene(device, scene, pool, owner.renderer, first);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        observations[camera] = expect_visible(first, camera, ids[camera], 1);
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(first, camera, ids[other]);
            }
        }
        if(camera > 0){
            EXPECT_GT(std::abs(observations[camera].column() - observations[camera - 1].column()), 2.0);
        }
    }
    expect_repeat_after_noop_update(device, owner.renderer, first);

    overlay_scenarios::apply_update(device, owner.renderer, state);
    rlt::update(device, owner.renderer);
    const auto second = capture(device, owner.renderer);
    expect_static_scene(device, scene, pool, owner.renderer, second);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        if(camera == 2){
            expect_camera_changed(first, second, camera);
            expect_instance_moved(first, second, camera, ids[camera]);
            expect_visible(second, camera, ids[camera], 1);
        }
        else{
            EXPECT_TRUE(camera_equal(first, camera, second, camera));
        }
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(second, camera, ids[other]);
            }
        }
    }
    expect_repeat_after_noop_update(device, owner.renderer, second);
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, DYNAMIC_OBJECT_SETS_ARE_DISJOINT){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::DISJOINT);
    auto& scene = state.scene;
    auto& pool = state.pool;
    const auto& scenario_definition = overlay_scenarios::definition(state.scenario);
    std::array<AssetHandle, NUM_CAMERAS> assets;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        assets[camera] = state.assets[camera];
    }
    RendererOwner<OverlaySpec> owner(device);
    rlt::init(device, owner.renderer, scene, pool);
    set_identical_cameras(device, owner.renderer);
    overlay_scenarios::build_initial(device, owner.renderer, state);
    std::array<uint32_t, NUM_CAMERAS> ids;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        ids[camera] = state.ids[camera];
        expect_maps_to_asset(device, scene, pool, owner.renderer, ids[camera], assets[camera]);
        if(camera > 0){
            EXPECT_NE(rlt::segmentation_object(device, scene, pool, owner.renderer, ids[camera]), rlt::segmentation_object(device, scene, pool, owner.renderer, ids[camera - 1]));
        }
    }

    rlt::update(device, owner.renderer);
    const auto first = capture(device, owner.renderer);
    expect_raw_frame_valid(first);
    expect_static_scene(device, scene, pool, owner.renderer, first);
    size_t previous_count = 0;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        const auto observation = expect_visible(first, camera, ids[camera], scenario_definition.assets[camera].dominant_channel);
        EXPECT_GT(observation.count, previous_count);
        previous_count = observation.count;
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(first, camera, ids[other]);
            }
        }
    }
    expect_repeat_after_noop_update(device, owner.renderer, first);

    overlay_scenarios::apply_update(device, owner.renderer, state);
    rlt::update(device, owner.renderer);
    const auto second = capture(device, owner.renderer);
    expect_static_scene(device, scene, pool, owner.renderer, second);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        if(camera == 1){
            expect_camera_changed(first, second, camera);
            expect_instance_moved(first, second, camera, ids[camera]);
            expect_visible(second, camera, ids[camera], scenario_definition.assets[camera].dominant_channel);
        }
        else{
            EXPECT_TRUE(camera_equal(first, camera, second, camera));
        }
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(second, camera, ids[other]);
            }
        }
    }
    expect_repeat_after_noop_update(device, owner.renderer, second);
}

TEST(RL_TOOLS_OVERLAY_SCENARIOS_SUITE, MIXED_DYNAMIC_SHARING_SCOPES){
    DEVICE device;
    rlt::init(device);
    auto state = overlay_scenarios::prepare(device, overlay_scenarios::Scenario::MIXED);
    auto& scene = state.scene;
    auto& pool = state.pool;
    const auto shared_all = state.assets[0];
    const auto shared_subset = state.assets[1];
    const auto mesh_only = state.assets[2];
    std::array<AssetHandle, NUM_CAMERAS> private_assets;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        private_assets[camera] = state.assets[3 + camera];
    }
    RendererOwner<OverlaySpec> owner(device);
    rlt::init(device, owner.renderer, scene, pool);
    set_identical_cameras(device, owner.renderer);
    overlay_scenarios::build_initial(device, owner.renderer, state);

    const uint32_t shared_all_id = state.ids[0];
    const uint32_t shared_subset_id = state.ids[1];
    std::array<uint32_t, NUM_CAMERAS> mesh_ids;
    std::array<uint32_t, NUM_CAMERAS> private_ids;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        const size_t mesh_placement = 2 + 2 * camera;
        mesh_ids[camera] = state.ids[mesh_placement];
        expect_maps_to_asset(device, scene, pool, owner.renderer, mesh_ids[camera], mesh_only);

        private_ids[camera] = state.ids[mesh_placement + 1];
        expect_maps_to_asset(device, scene, pool, owner.renderer, private_ids[camera], private_assets[camera]);
    }
    expect_maps_to_asset(device, scene, pool, owner.renderer, shared_all_id, shared_all);
    expect_maps_to_asset(device, scene, pool, owner.renderer, shared_subset_id, shared_subset);

    rlt::update(device, owner.renderer);
    const auto first = capture(device, owner.renderer);
    expect_raw_frame_valid(first);
    expect_static_scene(device, scene, pool, owner.renderer, first);
    size_t previous_private_count = 0;
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        expect_visible(first, camera, shared_all_id, 0);
        expect_visible(first, camera, mesh_ids[camera], 2);
        const auto private_observation = expect_visible(first, camera, private_ids[camera], 0);
        EXPECT_GT(private_observation.count, previous_private_count);
        previous_private_count = private_observation.count;
        if(camera < 2){
            expect_visible(first, camera, shared_subset_id, 1);
        }
        else{
            expect_absent(first, camera, shared_subset_id);
        }
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(first, camera, mesh_ids[other]);
                expect_absent(first, camera, private_ids[other]);
            }
        }
        expect_same_instance_pixels(first, 0, camera, shared_all_id);
    }
    expect_same_instance_pixels(first, 0, 1, shared_subset_id);
    expect_repeat_after_noop_update(device, owner.renderer, first);

    overlay_scenarios::apply_update(device, owner.renderer, state);
    rlt::update(device, owner.renderer);
    const auto second = capture(device, owner.renderer);
    expect_static_scene(device, scene, pool, owner.renderer, second);
    for(TI camera = 0; camera < NUM_CAMERAS; camera++){
        if(camera == 2){
            expect_camera_changed(first, second, camera);
            expect_instance_moved(first, second, camera, mesh_ids[camera]);
        }
        else{
            EXPECT_TRUE(camera_equal(first, camera, second, camera));
        }
        expect_visible(second, camera, shared_all_id, 0);
        expect_visible(second, camera, mesh_ids[camera], 2);
        expect_visible(second, camera, private_ids[camera], 0);
        expect_same_instance_pixels(first, camera, second, camera, shared_all_id);
        expect_same_instance_pixels(first, camera, second, camera, private_ids[camera]);
        if(camera < 2){
            expect_visible(second, camera, shared_subset_id, 1);
            expect_same_instance_pixels(first, camera, second, camera, shared_subset_id);
        }
        else{
            expect_absent(second, camera, shared_subset_id);
        }
        for(TI other = 0; other < NUM_CAMERAS; other++){
            if(other != camera){
                expect_absent(second, camera, mesh_ids[other]);
                expect_absent(second, camera, private_ids[other]);
            }
        }
        expect_same_instance_pixels(second, 0, camera, shared_all_id);
    }
    expect_same_instance_pixels(second, 0, 1, shared_subset_id);
    expect_repeat_after_noop_update(device, owner.renderer, second);
}
