#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_EMPTY_OBJECT_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include "render_copy.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

// Objects without triangles (light carriers like HSSD's habitat_default_* roots, meshes without
// faces, empty pool assets) render as nothing on every backend: their instances keep their ids
// and their lights shade the scene like scene lights.
namespace {
    using namespace rl_tools;
    namespace rrt = rendering::raytracing;
    namespace outputs = rrt::outputs;
    using DEVICE = devices::DefaultCPU;
    using TI = DEVICE::index_t;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_EMPTY_OBJECT_ACTIVE_BACKEND)
    using BACKEND = rrt::backends::Default;
#else
    using BACKEND = rrt::backends::Generic;
#endif
    template <bool OVERLAYS>
    struct Config: rrt::config::Default<float, TI> {
        using OUTPUTS = outputs::List<outputs::RGB<>, outputs::Segmentation<>>;
        using SHADING = rrt::High;
        static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 1;
        static constexpr TI NUM_OVERLAYS = OVERLAYS ? 1 : 0, MAX_OVERLAY_INSTANCES = OVERLAYS ? 2 : 0, MAX_OVERLAYS_PER_CAMERA = OVERLAYS ? 1 : 0;
    };
    constexpr float QUARTER_TURN_AHEAD[12] = {0, -1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0};
    constexpr uint32_t MISS = 0xffffffffu;

    enum class Lighting { NONE, SCENE_LIGHT, LIGHT_CARRIER };

    rrt::Mesh quad(float x, float half_extent, float red, float green, float blue) {
        rrt::Mesh mesh;
        mesh.vertices = {x, -half_extent, -half_extent, x, half_extent, -half_extent, x, half_extent, half_extent, x, -half_extent, half_extent};
        mesh.indices = {0, 2, 1, 0, 3, 2};
        mesh.color[0] = red;
        mesh.color[1] = green;
        mesh.color[2] = blue;
        return mesh;
    }

    // LIGHT_CARRIER places a light-only object (light +Y in its frame, turned to -X toward the
    // camera) and an object whose mesh has vertices but no faces; SCENE_LIGHT is the same light
    // as a scene light
    rendering::Bundle<float> bundle(DEVICE& device, Lighting lighting) {
        rendering::Bundle<float> result;
        add(device, result.scene, quad(8, 4, .6f, .6f, .6f));
        if(lighting == Lighting::SCENE_LIGHT) {
            result.scene.lights.push_back({0, {0, 0, 0}, {-1, 0, 0}, {.5f, .5f, .5f}, 0, 0, 1, 0, 0});
        }
        if(lighting == Lighting::LIGHT_CARRIER) {
            rrt::Object carrier;
            carrier.lights.push_back({0, {0, 0, 0}, {0, 1, 0}, {.5f, .5f, .5f}, 0, 0, 1, 0, 0});
            add(device, result.scene, carrier, QUARTER_TURN_AHEAD);
            rrt::Mesh faceless{};
            faceless.vertices = {3, 0, 0, 3, 1, 0, 3, 0, 1};
            add(device, result.scene, faceless);
        }
        rendering::datasets::compute_bounds(device, result);
        return result;
    }

    struct Frame {
        std::vector<uint32_t> rgb, ids;
    };

    template <typename SPEC>
    Frame capture(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer) {
        const float position[3] = {0, 0, 0}, target[3] = {1, 0, 0}, up[3] = {0, 0, 1};
        const auto camera = make_camera_data(position, target, up, 80.0f, 1.0f);
        golden::copy_in(device, renderer.device, &camera, cameras(device, renderer));
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            update(device, renderer);
        }
        render(device, renderer);
        synchronize(device, renderer);
        Frame frame;
        golden::copy_out(renderer.device, device, output(device, renderer, outputs::RGB<>{}), frame.rgb);
        golden::copy_out(renderer.device, device, output(device, renderer, outputs::Segmentation<>{}), frame.ids);
        return frame;
    }

    Frame render_scene(DEVICE& device, Lighting lighting) {
        auto scene = bundle(device, lighting);
        rrt::Renderer<rrt::Specification<Config<false>>, BACKEND> renderer;
        malloc(device, renderer);
        init(device, renderer, scene);
        const auto frame = capture(device, renderer);
        free(device, renderer);
        return frame;
    }

    std::size_t count(const std::vector<uint32_t>& ids, uint32_t id) {
        std::size_t matches = 0;
        for(const auto value : ids) {
            matches += value == id;
        }
        return matches;
    }
}

TEST(RENDERING_RAYTRACING_EMPTY_OBJECT, LIGHT_CARRIER_SHADES_LIKE_SCENE_LIGHT) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    const auto unlit = render_scene(device, Lighting::NONE);
    const auto scene_light = render_scene(device, Lighting::SCENE_LIGHT);
    const auto carrier = render_scene(device, Lighting::LIGHT_CARRIER);
    std::size_t lit = 0;
    for(std::size_t pixel = 0; pixel < unlit.rgb.size(); pixel++) {
        lit += unlit.rgb[pixel] != scene_light.rgb[pixel];
    }
    ASSERT_GT(lit, unlit.rgb.size() / 4);
    EXPECT_EQ(carrier.rgb, scene_light.rgb);
    EXPECT_EQ(carrier.ids, scene_light.ids);
    EXPECT_GT(count(carrier.ids, 0), 0u);
    EXPECT_GT(count(carrier.ids, MISS), 0u);
    EXPECT_EQ(count(carrier.ids, 0) + count(carrier.ids, MISS), carrier.ids.size());
}

TEST(RENDERING_RAYTRACING_EMPTY_OBJECT, EMPTY_POOL_ASSET_SPAWNS) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto scene = bundle(device, Lighting::LIGHT_CARRIER);
    rrt::AssetPool pool;
    const auto empty = add(device, pool, rrt::Object{});
    const auto marker = add(device, pool, quad(5, .5f, 1, 0, 0));
    rrt::Renderer<rrt::Specification<Config<true>>, BACKEND> renderer;
    malloc(device, renderer);
    init(device, renderer, scene, pool);
    attach(device, renderer, TI{0}, rrt::OverlayIndex{0});
    const auto empty_placement = spawn(device, renderer, rrt::OverlayIndex{0}, empty);
    spawn(device, renderer, rrt::OverlayIndex{0}, marker);
    const auto with_empty = capture(device, renderer);
    despawn(device, renderer, rrt::OverlayIndex{0}, empty_placement);
    const auto without_empty = capture(device, renderer);
    free(device, renderer);
    const uint32_t marker_id = (uint32_t)scene.scene.instances.size() + 1;
    constexpr std::size_t CENTER = 16 * 32 + 16;
    EXPECT_EQ(with_empty.ids[CENTER], marker_id);
    EXPECT_GT(count(with_empty.ids, 0), 0u);
    EXPECT_EQ(count(with_empty.ids, 0) + count(with_empty.ids, marker_id) + count(with_empty.ids, MISS), with_empty.ids.size());
    EXPECT_EQ(with_empty.rgb, without_empty.rgb);
    EXPECT_EQ(with_empty.ids, without_empty.ids);
}
