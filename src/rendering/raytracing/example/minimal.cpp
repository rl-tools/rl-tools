#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 256, CAM_HEIGHT = 256, NUM_CAMERAS = 8, NUM_PROBES = 1;
    using SHADING = rlt::rendering::raytracing::High;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

int main(int argc, char** argv){
    const std::string scene_path = argc > 1 ? argv[1] : "tests/data/ProcTHOR-Train-1.glb";
    const std::string output_prefix = argc > 2 ? argv[2] : "minimal";

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
    rlt::init(device, renderer, scene);

    // panorama: one camera per yaw from an interior point of the default scene (FLU frame)
    constexpr T PI = static_cast<T>(3.14159265358979323846);
    constexpr T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    constexpr T position[3] = {static_cast<T>(-6.28), static_cast<T>(-4.18), static_cast<T>(1.5)};
    const T up[3] = {0, 0, 1};
    std::vector<rlt::rendering::raytracing::Camera<T>> camera_staging(SPEC::NUM_CAMERAS);
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const T angle = 2 * PI * static_cast<T>(camera_i) / static_cast<T>(SPEC::NUM_CAMERAS);
        const T look_at[3] = {
            position[0] + std::cos(angle),
            position[1] + std::sin(angle),
            position[2]
        };
        camera_staging[camera_i] = rlt::make_camera_data(position, look_at, up, SPEC::COS_FOVY, aspect);
    }
    rlt::copy_to_renderer(device, renderer, camera_staging.data(), rlt::data(rlt::cameras(device, renderer)), camera_staging.size());

    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);

    std::vector<uint32_t> frame_buffer(static_cast<size_t>(SPEC::NUM_CAMERAS) * SPEC::CAM_PIXELS);
    rlt::copy_from_renderer(device, renderer, rlt::data(rlt::frame_buffer(device, renderer)), frame_buffer.data(), frame_buffer.size());

    int status = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const std::string path = output_prefix + "_" + std::to_string(camera_i) + ".png";
        if(stbi_write_png(path.c_str(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 4, frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH * 4) == 0){
            std::cerr << "Failed to write " << path << std::endl;
            status = 1;
        }
        else{
            std::cout << "Wrote " << path << std::endl;
        }
    }

    rlt::free(device, renderer);
    return status;
}
