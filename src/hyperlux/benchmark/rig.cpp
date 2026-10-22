// per-frame cost of articulated overlays: set_pose with shutter states (joint evaluation and
// motion samples on the host) and the update + render that consumes it, for a batch of two-link robots
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/hyperlux/operations_cpu_mux.h>
#include <rl_tools/hyperlux/datasets/operations_cpu.h>
#include <metra/metra.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE::index_t;
constexpr TI ROBOTS = 16, PARTS = 2, SAMPLES = 8, FRAMES = 30;
struct Config: rlt::hyperlux::config::Default<float, TI>{
    static constexpr TI CAM_WIDTH = 64, CAM_HEIGHT = 64, NUM_CAMERAS = 1, NUM_PROBES = 0;
    static constexpr TI NUM_OVERLAYS = ROBOTS, MAX_OVERLAY_INSTANCES = PARTS, MAX_OVERLAYS_PER_CAMERA = ROBOTS;
    static constexpr bool ENABLE_MOTION_BLUR = true, ENABLE_DYNAMIC_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = SAMPLES;
    using SHADING = rlt::hyperlux::Low;
};
using SPEC = rlt::hyperlux::Specification<Config>;
using Renderer = rlt::HyperLux<SPEC>;
using Clock = std::chrono::steady_clock;

int main(){
    DEVICE device;
    rlt::init(device);
    rlt::hyperlux::Object object;
    rlt::hyperlux::Mesh mesh;
    mesh.vertices = {0, -0.15f, -0.04f, 0, 0.15f, -0.04f, 0, 0.15f, 0.04f, 0, -0.15f, 0.04f};
    mesh.indices = {0, 2, 1, 0, 3, 2};
    mesh.color[0] = 0.8f;
    mesh.color[1] = 0.2f;
    mesh.color[2] = 0.1f;
    object.meshes.push_back(mesh);
    const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    rlt::hyperlux::ObjectAssembly assembly;
    rlt::add(device, assembly, object, identity);
    rlt::add(device, assembly, object, identity);
    auto& hinge = assembly.parts[0].joint;
    hinge.type = rlt::hyperlux::Joint::Type::REVOLUTE;
    hinge.axis[0] = 1;
    hinge.axis[2] = 0;
    hinge.pivot[1] = 0.2f;
    hinge.coordinate = 0;
    assembly.parts[1].parent = 0;
    assembly.parts[1].joint.type = rlt::hyperlux::Joint::Type::PRISMATIC;
    assembly.parts[1].joint.coordinate = 1;
    rlt::hyperlux::AssetPool pool;
    const auto asset = rlt::add(device, pool, assembly);
    rlt::hyperlux::Bundle<float> scene;
    const float background[12] = {1, 0, 0, 8, 0, 1, 0, 0, 0, 0, 1, 0};
    mesh.vertices = {0, -5, -5, 0, 5, -5, 0, 5, 5, 0, -5, 5};
    object.meshes[0] = mesh;
    rlt::add(device, scene.scene, object, background);
    rlt::hyperlux::datasets::compute_bounds(device, scene);
    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::hyperlux::OverlayPlacement placements[ROBOTS];
    for(TI robot = 0; robot < ROBOTS; robot++){
        rlt::attach(device, renderer, TI(0), rlt::hyperlux::OverlayIndex{robot});
        placements[robot] = rlt::spawn(device, renderer, rlt::hyperlux::OverlayIndex{robot}, asset);
    }
    const float position[3] = {0, 0, 0}, look_at[3] = {1, 0, 0}, up[3] = {0, 0, 1};
    auto camera = rlt::make_camera_data(position, look_at, up, 60.f, 1.f);
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
    camera_alias._data = &camera;
    rlt::copy(device, renderer.device, camera_alias, renderer.cameras);
    rlt::copy(device, renderer.device, camera_alias, renderer.cameras_open);

    std::vector<double> posing, rendering;
    for(TI frame = 0; frame < FRAMES + 3; frame++){
        const auto begin = Clock::now();
        for(TI robot = 0; robot < ROBOTS; robot++){
            float base[12] = {1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0};
            base[7] = (float(robot % 4) - 1.5f) * 0.8f;
            base[11] = (float(robot / 4) - 1.5f) * 0.8f;
            const float open[2] = {float(frame) * 0.2f, 0.1f};
            const float close[2] = {float(frame) * 0.2f + 7.f, 0.15f};
            rlt::set_pose(device, renderer, rlt::hyperlux::OverlayIndex{robot}, placements[robot], rlt::make_pose(base), rlt::make_pose(base), open, close);
        }
        const auto posed = Clock::now();
        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        const auto rendered = Clock::now();
        if(frame >= 3){
            posing.push_back(std::chrono::duration<double>(posed - begin).count());
            rendering.push_back(std::chrono::duration<double>(rendered - posed).count());
        }
    }
    metra::log("hyperlux/rig/pose_seconds", posing);
    metra::log("hyperlux/rig/render_seconds", rendering);
    std::cout << "hyperlux/rig/pose_us=" << std::accumulate(posing.begin(), posing.end(), 0.0) / FRAMES * 1e6 << '\n';
    std::cout << "hyperlux/rig/render_us=" << std::accumulate(rendering.begin(), rendering.end(), 0.0) / FRAMES * 1e6 << '\n';
    rlt::free(device, renderer);
    return 0;
}
