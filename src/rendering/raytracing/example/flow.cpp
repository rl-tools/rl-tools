// Two-instant optical flow: a cube (overlay) translates across the living room of the default
// scene while the camera strafes. Frame 0 establishes the state, then one steady-state step
// (advance_flow -> move -> update -> render) produces backward flow on the current frame,
// forward flow on the previous frame, and the FlowStatus buffers that mask both.
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include <iostream>
#include <string>

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
static constexpr T FOV = 80;
using TI = typename DEVICE::index_t;

struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 256, CAM_HEIGHT = 256, NUM_CAMERAS = 1, NUM_PROBES = 1;
    using SHADING = rlt::rendering::raytracing::Low;
    using OUTPUTS = rro::List<rro::RGB<>, rro::Flow<>, rro::FlowStatus<>, rro::Flow<rro::direction::Forward>, rro::FlowStatus<rro::direction::Forward>, rro::Probes>;
    static constexpr TI NUM_OVERLAYS = 1;
    static constexpr TI MAX_OVERLAY_INSTANCES = 1;
    static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr T CUBE_HALF_EXTENT = 0.3;
static constexpr T CUBE_POSITION_PREVIOUS[3] = {-5.2, -5.0, 1.4};
static constexpr T CUBE_POSITION_CURRENT[3] = {-4.9, -4.5, 1.5};
static constexpr T CAMERA_POSITION_PREVIOUS[3] = {-6.5, -4.2, 1.6};
static constexpr T CAMERA_POSITION_CURRENT[3] = {-6.4, -4.4, 1.6};
static constexpr T CAMERA_LOOK_AT[3] = {-4.5, -5.0, 1.3};

rlt::rendering::raytracing::Mesh make_cube(){
    using namespace rlt::rendering::raytracing::constants;
    rlt::rendering::raytracing::Mesh mesh;
    mesh.color[0] = 0.85f;
    mesh.color[1] = 0.3f;
    mesh.color[2] = 0.1f;
    for(TI vertex_i = 0; vertex_i < (TI)NUM_VERTICES; vertex_i++){
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            mesh.vertices.push_back(default_vertices[vertex_i][dim_i] * static_cast<float>(CUBE_HALF_EXTENT));
        }
    }
    for(TI triangle_i = 0; triangle_i < (TI)NUM_INDICES; triangle_i++){
        for(TI corner_i = 0; corner_i < 3; corner_i++){
            mesh.indices.push_back(default_indices[triangle_i][corner_i]);
        }
    }
    return mesh;
}

void translation(const T position[3], float transform[12]){
    const float quaternion_wxyz[4] = {1, 0, 0, 0};
    const float position_float[3] = {static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2])};
    rlt::make_transform(position_float, quaternion_wxyz, transform);
}

void set_camera(DEVICE& device, Renderer& renderer, const T position[3]){
    constexpr T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
    const T up[3] = {0, 0, 1};
    rlt::Tensor<typename Renderer::CAMERA_TENSOR_SPEC> camera_staging;
    rlt::malloc(device, camera_staging);
    rlt::set(device, camera_staging, rlt::make_camera_data(position, CAMERA_LOOK_AT, up, FOV, aspect), 0);
    rlt::copy(device, renderer.device, camera_staging, rlt::cameras(device, renderer));
    rlt::free(device, camera_staging);
}

int main(int argc, char** argv){
    const std::string scene_path = argc > 1 ? argv[1] : "tests/data/ProcTHOR-Train-1.glb";
    const std::string output_prefix = argc > 2 ? argv[2] : "flow";

    DEVICE device;
    rlt::init(device);

    Renderer renderer;
    rlt::malloc(device, renderer);

    rlt::rendering::Bundle<T> bundle;
    if(!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, scene_path)){
        std::cerr << "Failed to load scene: " << scene_path << std::endl;
        rlt::free(device, renderer);
        return 1;
    }
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube = rlt::add(device, pool, make_cube());
    rlt::init(device, renderer, bundle, pool);
    rlt::attach(device, renderer, 0, rlt::rendering::raytracing::OverlayIndex{0});

    // frame 0: spawn initialises the previous instant equal to the current one
    float transform[12];
    translation(CUBE_POSITION_PREVIOUS, transform);
    const auto placement = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, cube, rlt::make_pose(transform));
    set_camera(device, renderer, CAMERA_POSITION_PREVIOUS);
    rlt::update(device, renderer);
    rlt::render(device, renderer);

    // steady-state step: publish frame 0 as the previous instant, move camera and object, republish
    rlt::advance_flow(device, renderer);
    set_camera(device, renderer, CAMERA_POSITION_CURRENT);
    translation(CUBE_POSITION_CURRENT, transform);
    rlt::set_pose(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, rlt::make_pose(transform));
    rlt::update(device, renderer);
    rlt::render(device, renderer);

    rlt::save_image(device, renderer, (output_prefix + "_rgb.png").c_str());
    rlt::save_flow_image(device, renderer, (output_prefix + "_backward.png").c_str());
    rlt::save_flow_status_image(device, renderer, (output_prefix + "_backward_status.png").c_str());
    rlt::save_flow_forward_image(device, renderer, (output_prefix + "_forward.png").c_str());
    rlt::save_flow_forward_status_image(device, renderer, (output_prefix + "_forward_status.png").c_str());
    std::cout << "Wrote " << output_prefix << "_{rgb,backward,backward_status,forward,forward_status}.png" << std::endl;

    rlt::free(device, renderer);
    return 0;
}
