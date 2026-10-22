// Collision probes trace the composed world: the shared scene plus the overlays attached
// to the probing camera. This pins that contract — spawned overlay objects shorten probe
// distances for attached cameras only, and the scene hit still wins when it is closer —
// which is what makes overlay entities visible to free-space sampling.
#include <rl_tools/operations/cpu.h>

#ifdef RL_TOOLS_HYPERLUX_PROBES_OVERLAY_TEST_ACTIVE_BACKEND
#include <rl_tools/hyperlux/operations_cpu_mux.h>
#else
#include <rl_tools/hyperlux/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/hyperlux/datasets/operations_cpu.h>

#include <gtest/gtest.h>

#include "../overlay_scenario_cases.h"
#include "../render_copy.h"

#include <vector>

#ifndef RL_TOOLS_PROBES_OVERLAY_SUITE
#define RL_TOOLS_PROBES_OVERLAY_SUITE HYPERLUX_PROBES_OVERLAY_GENERIC
#endif

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
static constexpr T FOV = 80;
using TI = typename DEVICE::index_t;

namespace {
#ifdef RL_TOOLS_HYPERLUX_PROBES_OVERLAY_TEST_ACTIVE_BACKEND
    using BACKEND = rlt::hyperlux::backends::Default;
#else
    using BACKEND = rlt::hyperlux::backends::Generic;
#endif

    struct Config: rlt::hyperlux::config::Default<T, TI> {
        static constexpr TI CAM_WIDTH = 16;
        static constexpr TI CAM_HEIGHT = 16;
        static constexpr TI NUM_CAMERAS = 2;
        static constexpr TI NUM_PROBES = 1;  // probe 0 = camera forward
        using SHADING = rlt::hyperlux::Low;
        using OUTPUTS = rro::List<rro::RGB<>, rro::Probes>;
        static constexpr TI NUM_OVERLAYS = 1;
        static constexpr TI MAX_OVERLAY_INSTANCES = 1;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
    };
    using SPEC = rlt::hyperlux::Specification<Config>;

    constexpr float WALL_X = 10.0f;
    constexpr float OVERLAY_X = 2.0f;
    constexpr float BEHIND_WALL_X = 20.0f;

    template <typename RENDERER>
    std::vector<rlt::hyperlux::CollisionResult> probe_forward(DEVICE& device, RENDERER& renderer){
        rlt::probe(device, renderer);
        std::vector<rlt::hyperlux::CollisionResult> results;
        golden::copy_out(renderer.device, device, rlt::output(device, renderer, rro::Probes{}), results);
        return results;
    }
}

TEST(RL_TOOLS_PROBES_OVERLAY_SUITE, PROBES_SEE_ATTACHED_OVERLAYS_ONLY){
    DEVICE device;
    rlt::init(device);

    rlt::hyperlux::Bundle<T> bundle;
    rlt::hyperlux::Object wall;
    wall.name = "wall";
    wall.meshes.push_back(overlay_scenarios::detail::make_quad(10.0f, {{0.5f, 0.5f, 0.5f}}));
    rlt::add(device, bundle.scene, wall, overlay_scenarios::pose(WALL_X, 0.0f, 0.0f).data());
    rlt::hyperlux::datasets::compute_bounds(device, bundle);

    rlt::hyperlux::AssetPool pool;
    rlt::hyperlux::Object box;
    box.name = "box";
    box.meshes.push_back(overlay_scenarios::detail::make_quad(1.0f, {{0.9f, 0.1f, 0.1f}}));
    const auto asset = rlt::add(device, pool, box);

    rlt::HyperLux<SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, bundle, pool);

    const T position[3] = {0, 0, 0};
    const T look_at[3] = {1, 0, 0};
    const T up[3] = {0, 0, 1};
    constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
    const auto camera = rlt::make_camera_data(position, look_at, up, FOV, aspect);
    rlt::hyperlux::Camera<T> cameras[SPEC::NUM_CAMERAS] = {camera, camera};
    golden::copy_in(device, renderer.device, cameras, rlt::cameras(device, renderer));

    // no overlays yet: both forward probes hit the wall
    auto results = probe_forward(device, renderer);
    ASSERT_EQ(results.size(), SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        EXPECT_EQ(results[camera_i].hit, 1);
        EXPECT_NEAR(results[camera_i].distance, WALL_X, 1e-3);
    }

    // overlay attached to camera 0 only: its probe hits the box, camera 1 still sees the wall
    rlt::attach(device, renderer, (TI)0, rlt::hyperlux::OverlayIndex{0});
    const auto placement = rlt::spawn(device, renderer, rlt::hyperlux::OverlayIndex{0}, asset, rlt::make_pose(overlay_scenarios::pose(OVERLAY_X, 0.0f, 0.0f).data()));
    rlt::update(device, renderer);
    results = probe_forward(device, renderer);
    EXPECT_EQ(results[0].hit, 1);
    EXPECT_NEAR(results[0].distance, OVERLAY_X, 1e-3);
    EXPECT_EQ(results[1].hit, 1);
    EXPECT_NEAR(results[1].distance, WALL_X, 1e-3);

    // overlay moved behind the wall: the closer scene hit wins again
    rlt::set_pose(device, renderer, rlt::hyperlux::OverlayIndex{0}, placement, rlt::make_pose(overlay_scenarios::pose(BEHIND_WALL_X, 0.0f, 0.0f).data()));
    rlt::update(device, renderer);
    results = probe_forward(device, renderer);
    EXPECT_EQ(results[0].hit, 1);
    EXPECT_NEAR(results[0].distance, WALL_X, 1e-3);
    EXPECT_NEAR(results[1].distance, WALL_X, 1e-3);

    rlt::free(device, renderer);
}
