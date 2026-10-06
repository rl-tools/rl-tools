// per-frame cost of articulated overlays: rig sampling on the host, the per-placement motion
// submission, and the update + render that consumes it, for a batch of two-link robots
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/rig/operations_cpu.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include <metra/metra.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE::index_t;
using RIG_SPEC = rlt::rendering::rig::Specification<float, TI, 2, 2>;
constexpr TI ROBOTS = 16, PARTS = 2, SAMPLES = 8, FRAMES = 30;
struct Config: rlt::rendering::raytracing::config::Default<float, TI>{
    static constexpr TI CAM_WIDTH = 64, CAM_HEIGHT = 64, NUM_CAMERAS = 1, NUM_PROBES = 0;
    static constexpr TI NUM_OVERLAYS = ROBOTS, MAX_OVERLAY_INSTANCES = PARTS, MAX_OVERLAYS_PER_CAMERA = ROBOTS;
    static constexpr bool ENABLE_MOTION_BLUR = true, ENABLE_DYNAMIC_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = SAMPLES;
    using SHADING = rlt::rendering::raytracing::Low;
};
using SPEC = rlt::rendering::raytracing::Specification<Config>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;
using Clock = std::chrono::steady_clock;

int main(){
    DEVICE device;
    rlt::init(device);
    rlt::rendering::rig::Model<RIG_SPEC> rig;
    rig.num_links = 2;
    rig.num_parts = 2;
    rig.links[0].joint = rlt::rendering::rig::Joint::REVOLUTE;
    rig.links[0].axis[0] = 1;
    rig.links[0].axis[2] = 0;
    rig.links[0].origin[7] = 0.2f;
    rig.parts[0].offset[7] = -0.2f;
    rig.links[1].parent = 0;
    rig.links[1].joint = rlt::rendering::rig::Joint::PRISMATIC;
    rig.parts[1].link = 1;
    if(!rlt::init(device, rig)){
        return 1;
    }
    rlt::rendering::Object object;
    rlt::rendering::Mesh mesh;
    mesh.vertices = {0, -0.15f, -0.04f, 0, 0.15f, -0.04f, 0, 0.15f, 0.04f, 0, -0.15f, 0.04f};
    mesh.indices = {0, 2, 1, 0, 3, 2};
    mesh.color[0] = 0.8f;
    mesh.color[1] = 0.2f;
    mesh.color[2] = 0.1f;
    object.meshes.push_back(mesh);
    const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    rlt::rendering::ObjectAssembly assembly;
    rlt::add(device, assembly, object, identity);
    rlt::add(device, assembly, object, identity);
    rlt::rendering::AssetPool pool;
    const auto asset = rlt::add(device, pool, assembly);
    rlt::rendering::Bundle<float> scene;
    const float background[12] = {1, 0, 0, 8, 0, 1, 0, 0, 0, 0, 1, 0};
    mesh.vertices = {0, -5, -5, 0, 5, -5, 0, 5, 5, 0, -5, 5};
    object.meshes[0] = mesh;
    rlt::add(device, scene.scene, object, background);
    rlt::rendering::datasets::compute_bounds(device, scene);
    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::rendering::raytracing::OverlayPlacement placements[ROBOTS];
    for(TI robot = 0; robot < ROBOTS; robot++){
        rlt::attach(device, renderer, TI(0), rlt::rendering::raytracing::OverlayIndex{robot});
        placements[robot] = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{robot}, asset, identity);
    }
    const float position[3] = {0, 0, 0}, look_at[3] = {1, 0, 0}, up[3] = {0, 0, 1};
    auto camera = rlt::make_camera_data(position, look_at, up, 60.f, 1.f);
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
    camera_alias._data = &camera;
    rlt::copy(device, renderer.device, camera_alias, renderer.cameras);
    rlt::copy(device, renderer.device, camera_alias, renderer.cameras_open);

    rlt::rendering::rig::Buffer<RIG_SPEC> buffer;
    float samples[ROBOTS][SAMPLES][PARTS][12], close[ROBOTS][PARTS][12];
    std::vector<double> kinematics, submission, rendering;
    for(TI frame = 0; frame < FRAMES + 3; frame++){
        const auto begin = Clock::now();
        for(TI robot = 0; robot < ROBOTS; robot++){
            float base[12] = {1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0};
            base[7] = (float(robot % 4) - 1.5f) * 0.8f;
            base[11] = (float(robot / 4) - 1.5f) * 0.8f;
            const float open[RIG_SPEC::MAX_COORDINATES] = {float(frame) * 0.2f, 0.1f};
            const float end[RIG_SPEC::MAX_COORDINATES] = {float(frame) * 0.2f + 7.f, 0.15f};
            rlt::sample(device, rig, base, base, open, end, SAMPLES, buffer, &samples[robot][0][0][0], PARTS * 12, &close[robot][0][0]);
        }
        const auto sampled = Clock::now();
        for(TI robot = 0; robot < ROBOTS; robot++){
            rlt::set_motion_transforms(device, renderer, rlt::rendering::raytracing::OverlayIndex{robot}, placements[robot], &samples[robot][0][0][0], &close[robot][0][0]);
        }
        const auto submitted = Clock::now();
        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        const auto rendered = Clock::now();
        if(frame >= 3){
            kinematics.push_back(std::chrono::duration<double>(sampled - begin).count());
            submission.push_back(std::chrono::duration<double>(submitted - sampled).count());
            rendering.push_back(std::chrono::duration<double>(rendered - submitted).count());
        }
    }
    metra::log("rendering/rig/kinematics_seconds", kinematics);
    metra::log("rendering/rig/submission_seconds", submission);
    metra::log("rendering/rig/render_seconds", rendering);
    for(const auto& metric : {std::make_pair("kinematics", &kinematics), std::make_pair("submission", &submission), std::make_pair("render", &rendering)}){
        std::cout << "rendering/rig/" << metric.first << "_us=" << std::accumulate(metric.second->begin(), metric.second->end(), 0.0) / FRAMES * 1e6 << '\n';
    }
    rlt::free(device, renderer);
    return 0;
}
