#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>

#include <cmath>
#include <iostream>
#include <string>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
static constexpr T FOV = 1.3962634015954636;
using TI = typename DEVICE::index_t;

struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 512, CAM_HEIGHT = 512, NUM_CAMERAS = 2, NUM_PROBES = 1;
    using SHADING = rlt::rendering::raytracing::High;
    static constexpr TI NUM_OVERLAYS = 3;
    static constexpr TI MAX_OVERLAY_INSTANCES = 5;
    static constexpr TI MAX_OVERLAYS_PER_CAMERA = 3;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

void yaw_transform(const T position[3], T yaw, float transform[12]){
    const float half = static_cast<float>(yaw) / 2;
    const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
    const float position_float[3] = {static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2])};
    rlt::make_transform(position_float, quaternion_wxyz, transform);
}

int render_and_save(DEVICE& device, Renderer& renderer, const std::string& output_prefix, const std::string& suffix){
    rlt::render(device, renderer);
    rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_buffer;
    rlt::malloc(device, frame_buffer);
    rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_buffer);
    int status = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const std::string path = output_prefix + "_camera_" + std::to_string(camera_i) + suffix + ".png";
        if(stbi_write_png(path.c_str(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 4, rlt::data(frame_buffer) + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH * 4) == 0){
            std::cerr << "Failed to write " << path << std::endl;
            status = 1;
        }
        else{
            std::cout << "Wrote " << path << std::endl;
        }
    }
    rlt::free(device, frame_buffer);
    return status;
}

int main(int argc, char** argv){
    const std::string scene_path = argc > 1 ? argv[1] : "tests/data/ProcTHOR-Train-1.glb";
    const std::string output_prefix = argc > 2 ? argv[2] : "minimal_overlay";

    DEVICE device;
    rlt::init(device);

    Renderer renderer;
    rlt::malloc(device, renderer);

    rlt::rendering::raytracing::Scene scene;
    if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, scene_path)){
        std::cerr << "Failed to load scene: " << scene_path << std::endl;
        rlt::free(device, renderer);
        return 1;
    }
    constexpr T LIGHT_INTENSITY_SCALE = 6;
    for(auto& light : scene.lights){
        for(TI channel_i = 0; channel_i < 3; channel_i++){
            light.color[channel_i] *= LIGHT_INTENSITY_SCALE;
        }
    }

    rlt::rendering::raytracing::AssetPool pool;
    const std::string drone_paths[3] = {"tests/data/x500.glb", "tests/data/savagebee_pusher.glb", "tests/data/soft.glb"};
    rlt::rendering::raytracing::AssetHandle drone_assets[3];
    for(TI drone_i = 0; drone_i < 3; drone_i++){
        rlt::rendering::raytracing::ObjectAssembly assembly;
        if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, assembly, drone_paths[drone_i])){
            std::cerr << "Failed to load drone: " << drone_paths[drone_i] << std::endl;
            rlt::free(device, renderer);
            return 1;
        }
        drone_assets[drone_i] = rlt::add(device, pool, assembly);
    }

    rlt::init(device, renderer, scene, pool);

    // drones A and B face each other across the living room (FLU frame) with a clear sight
    // line; the shared drone hovers off to the side of that line. Each camera aims straight at
    // the other 3rd-person drone, so it would be dead center in frame — the overlay
    // attachments are the only thing removing it
    constexpr T PI = static_cast<T>(3.14159265358979323846);
    constexpr T drone_a_position[3] = {static_cast<T>(-6.3), static_cast<T>(-4.5), static_cast<T>(1.6)};
    constexpr T drone_b_position[3] = {static_cast<T>(-4.3), static_cast<T>(-4.5), static_cast<T>(1.6)};
    constexpr T drone_shared_position[3] = {static_cast<T>(-5.3), static_cast<T>(-5.35), static_cast<T>(2.0)};

    float transform[12];
    yaw_transform(drone_a_position, 0, transform);
    rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, drone_assets[0], transform);
    yaw_transform(drone_b_position, PI, transform);
    rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{1}, drone_assets[1], transform);
    yaw_transform(drone_shared_position, PI / 2, transform);
    rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{2}, drone_assets[2], transform);

    rlt::attach(device, renderer, 0, rlt::rendering::raytracing::OverlayIndex{0});
    rlt::attach(device, renderer, 0, rlt::rendering::raytracing::OverlayIndex{2});
    rlt::attach(device, renderer, 1, rlt::rendering::raytracing::OverlayIndex{1});
    rlt::attach(device, renderer, 1, rlt::rendering::raytracing::OverlayIndex{2});
    rlt::update(device, renderer);

    constexpr T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    const T up[3] = {0, 0, 1};
    const T camera_0_position[3] = {drone_a_position[0] - static_cast<T>(1.0), drone_a_position[1], drone_a_position[2] + static_cast<T>(0.8)};
    const T camera_1_position[3] = {drone_b_position[0] + static_cast<T>(1.0), drone_b_position[1], drone_b_position[2] + static_cast<T>(0.8)};
    rlt::Tensor<typename Renderer::CAMERA_TENSOR_SPEC> camera_staging;
    rlt::malloc(device, camera_staging);
    rlt::set(device, camera_staging, rlt::make_camera_data(camera_0_position, drone_b_position, up, FOV, aspect), 0);
    rlt::set(device, camera_staging, rlt::make_camera_data(camera_1_position, drone_a_position, up, FOV, aspect), 1);
    rlt::copy(device, renderer.device, camera_staging, rlt::cameras(device, renderer));
    rlt::free(device, camera_staging);

    int status = render_and_save(device, renderer, output_prefix, "");

    // proof shot: attach the remaining overlay to each camera — the other 3rd-person drone
    // appears dead center, exactly where the first render filtered it out
    rlt::attach(device, renderer, 0, rlt::rendering::raytracing::OverlayIndex{1});
    rlt::attach(device, renderer, 1, rlt::rendering::raytracing::OverlayIndex{0});
    rlt::update(device, renderer);
    status |= render_and_save(device, renderer, output_prefix, "_all");

    rlt::free(device, renderer);
    return status;
}
