#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>

#include <emscripten.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace lissajous = rl_tools::rl::environments::l2f::parameters::trajectories::lissajous;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 480;
static constexpr TI CAM_HEIGHT = 270;
static constexpr TI NUM_CAMERAS = 2;
static constexpr TI NUM_PROBES = 1;
static constexpr TI ONBOARD_CAMERA = 0;
static constexpr TI THIRD_PERSON_CAMERA = 1;

struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = ::CAM_WIDTH, CAM_HEIGHT = ::CAM_HEIGHT, NUM_CAMERAS = ::NUM_CAMERAS, NUM_PROBES = ::NUM_PROBES;
    using SHADING = rlt::rendering::raytracing::High;
    static constexpr bool OUTPUT_RGB = true;
    static constexpr bool ENABLE_ANTI_ALIASING = true;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = 2;
    static constexpr TI NUM_OVERLAYS = 1;
    static constexpr TI MAX_OVERLAY_INSTANCES = 8;
    static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
    static constexpr bool ENABLE_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = 8;
    static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr T GRAVITY = 9.81;
static constexpr T ONBOARD_FOV = 80;
static constexpr T THIRD_PERSON_FOV = 37.24225668350351;
static constexpr T CAMERA_MOUNT_BODY[3] = {0.10, 0, 0.32};
static constexpr T CAMERA_PITCH_DOWN = 0.3;
static constexpr T TRIPOD_POSITION[3] = {-7.6, -7.0, 2.0};
static constexpr T PROP_RATE = 150.0;
static constexpr T SHUTTER_FRACTION = 0.5;
static constexpr T CENTER[3] = {-5.0, -5.2, 1.55};
static constexpr double FPS = 60.0;

static void normalize(const T v[3], T out[3]) {
    const T norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    out[0] = v[0] / norm; out[1] = v[1] / norm; out[2] = v[2] / norm;
}

static void cross(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

struct Pose {
    T position[3];
    T basis[3][3]; // world-frame body axes: basis[0] = x_b (forward), basis[1] = y_b (left), basis[2] = z_b (up)
};

static void rotate_body_to_world(const Pose& pose, const T v[3], T out[3]) {
    for (TI dim = 0; dim < 3; dim++) {
        out[dim] = pose.basis[0][dim]*v[0] + pose.basis[1][dim]*v[1] + pose.basis[2][dim]*v[2];
    }
}

static Pose flat_pose(DEVICE& device, const lissajous::Parameters<T>& trajectory, const T center[3], T time) {
    const auto step = lissajous::evaluate(device, trajectory, time);
    const T finite_difference_h = 1e-3;
    const auto step_before = lissajous::evaluate(device, trajectory, time - finite_difference_h);
    const auto step_after = lissajous::evaluate(device, trajectory, time + finite_difference_h);

    Pose pose;
    for (TI dim = 0; dim < 3; dim++) {
        pose.position[dim] = center[dim] + step.position[dim];
    }
    T thrust[3];
    for (TI dim = 0; dim < 3; dim++) {
        thrust[dim] = (step_after.linear_velocity[dim] - step_before.linear_velocity[dim]) / (2 * finite_difference_h);
    }
    thrust[2] += GRAVITY;
    normalize(thrust, pose.basis[2]);

    const T heading[3] = {1, 0, 0};
    T left[3];
    cross(pose.basis[2], heading, left);
    normalize(left, pose.basis[1]);
    cross(pose.basis[1], pose.basis[2], pose.basis[0]);
    return pose;
}

static void pose_to_transform(const Pose& pose, float out[12]) {
    for (TI row = 0; row < 3; row++) {
        out[row * 4 + 0] = pose.basis[0][row];
        out[row * 4 + 1] = pose.basis[1][row];
        out[row * 4 + 2] = pose.basis[2][row];
        out[row * 4 + 3] = pose.position[row];
    }
}

static void status(const char* message) {
    EM_ASM({ if(Module.rlt_status){ Module.rlt_status(UTF8ToString($0)); } }, message);
}

static void present(const uint32_t* frame) {
    EM_ASM({ if(Module.rlt_present){ Module.rlt_present($0, $1, $2, $3); } }, frame, (int)CAM_WIDTH, (int)CAM_HEIGHT, (int)NUM_CAMERAS);
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_run(const char* scene_path, const char* drone_path, int max_frames) {
    DEVICE device;
    rlt::init(device);

    status("Parsing scene ...");
    rlt::rendering::raytracing::Scene scene;
    if (!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, scene_path)) {
        status("Failed to load the scene GLB");
        return 1;
    }

    status("Parsing drone ...");
    rlt::rendering::raytracing::ObjectAssembly drone_assembly;
    if (!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, drone_assembly, drone_path)) {
        status("Failed to load the drone GLB");
        return 1;
    }
    // node origins are not required to sit at the rotor hubs (e.g. generated meshes with
    // identity node transforms), so each prop spins about its part-local AABB center
    struct PropPart {
        TI part;
        T pivot[2];
        T direction;
    };
    std::vector<PropPart> props;
    for (size_t part_i = 0; part_i < drone_assembly.parts.size(); part_i++) {
        const auto& object = drone_assembly.objects[drone_assembly.parts[part_i].object];
        if (object.name.rfind("prop_", 0) == 0) {
            T low[3] = {0, 0, 0}, high[3] = {0, 0, 0};
            bool first = true;
            for (const auto& mesh : object.meshes) {
                for (size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3) {
                    for (TI dim = 0; dim < 3; dim++) {
                        const T value = mesh.vertices[vertex_i + dim];
                        low[dim] = first ? value : std::min(low[dim], value);
                        high[dim] = first ? value : std::max(high[dim], value);
                    }
                    first = false;
                }
            }
            PropPart prop;
            prop.part = part_i;
            prop.pivot[0] = (low[0] + high[0]) / 2;
            prop.pivot[1] = (low[1] + high[1]) / 2;
            prop.direction = prop.pivot[0] * prop.pivot[1] > 0 ? 1 : -1;
            props.push_back(prop);
        }
    }
    if (drone_assembly.parts.empty() || drone_assembly.objects[drone_assembly.parts[0].object].name != "body") {
        status("Drone assembly must have a scene-root node named 'body' as its first part");
        return 1;
    }

    rlt::rendering::raytracing::AssetPool pool;
    const auto drone_asset = rlt::add(device, pool, drone_assembly);

    status("Building BVH and uploading to the GPU ...");
    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, ONBOARD_CAMERA, rlt::rendering::raytracing::OverlayIndex{0});
    rlt::attach(device, renderer, THIRD_PERSON_CAMERA, rlt::rendering::raytracing::OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto placement = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, drone_asset, identity);

    lissajous::Parameters<T> trajectory = lissajous::default_parameters<T>;
    trajectory.A = 0.45;
    trajectory.B = 0.5;
    trajectory.C = 0.2;
    trajectory.ramp_duration = 0.0;

    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    constexpr size_t CAM_PIXELS = static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT);
    std::vector<uint32_t> frame(NUM_CAMERAS * CAM_PIXELS);

    const auto make_cameras = [&](const Pose& pose, rlt::rendering::raytracing::Camera<T> cameras[NUM_CAMERAS]){
        {
            T mount[3], forward[3], up[3];
            rotate_body_to_world(pose, CAMERA_MOUNT_BODY, mount);
            const T pitch_cos = std::cos(CAMERA_PITCH_DOWN), pitch_sin = std::sin(CAMERA_PITCH_DOWN);
            const T forward_body[3] = {pitch_cos, 0, -pitch_sin};
            const T up_body[3] = {pitch_sin, 0, pitch_cos};
            rotate_body_to_world(pose, forward_body, forward);
            rotate_body_to_world(pose, up_body, up);
            const T position[3] = {pose.position[0] + mount[0], pose.position[1] + mount[1], pose.position[2] + mount[2]};
            const T look_at[3] = {position[0] + forward[0], position[1] + forward[1], position[2] + forward[2]};
            cameras[ONBOARD_CAMERA] = rlt::make_camera_data(position, look_at, up, ONBOARD_FOV, aspect);
        }
        {
            const T up[3] = {0, 0, 1};
            cameras[THIRD_PERSON_CAMERA] = rlt::make_camera_data(TRIPOD_POSITION, pose.position, up, THIRD_PERSON_FOV, aspect);
        }
    };
    const auto prop_spin = [](const PropPart& prop, T angle, float out[12]){
        const float c = std::cos(angle), s = std::sin(angle);
        const float px = prop.pivot[0], py = prop.pivot[1];
        const float spin[12] = {
            c, -s, 0, px - c * px + s * py,
            s,  c, 0, py - s * px - c * py,
            0,  0, 1, 0,
        };
        std::memcpy(out, spin, sizeof(spin));
    };

    status("Rendering ...");
    // max_frames > 0 renders a deterministic fixed-timestep sequence (headless verification);
    // otherwise wall-clock time drives the trajectory so frame pacing does not change the flight
    const double start_ms = emscripten_get_now();
    for (int frame_i = 0; max_frames <= 0 || frame_i < max_frames; frame_i++) {
        const T time_open = max_frames > 0 ? static_cast<T>(frame_i / FPS) : static_cast<T>((emscripten_get_now() - start_ms) / 1000.0);
        const T time_close = time_open + static_cast<T>(SHUTTER_FRACTION / FPS);
        const Pose pose_open = flat_pose(device, trajectory, CENTER, time_open);
        const Pose pose_close = flat_pose(device, trajectory, CENTER, time_close);

        float pose_transform_open[12], pose_transform_close[12];
        pose_to_transform(pose_open, pose_transform_open);
        pose_to_transform(pose_close, pose_transform_close);
        rlt::set_transform_pair(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, pose_transform_open, pose_transform_close);
        for (const auto& prop : props) {
            float spin_open[12], spin_close[12];
            prop_spin(prop, prop.direction * PROP_RATE * time_open, spin_open);
            prop_spin(prop, prop.direction * PROP_RATE * time_close, spin_close);
            rlt::set_transform_pair(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, prop.part, spin_open, spin_close);
        }

        rlt::rendering::raytracing::Camera<T> cameras_open[NUM_CAMERAS], cameras_close[NUM_CAMERAS];
        make_cameras(pose_open, cameras_open);
        make_cameras(pose_close, cameras_close);
        rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
        camera_alias._data = cameras_open;
        rlt::copy(device, renderer.device, camera_alias, rlt::cameras_open(device, renderer));
        camera_alias._data = cameras_close;
        rlt::copy(device, renderer.device, camera_alias, rlt::cameras_close(device, renderer));

        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);

        rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
        frame_alias._data = frame.data();
        rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);

        present(frame.data());
        emscripten_sleep(1);
    }

    rlt::free(device, renderer);
    return 0;
}

int main() {
    return 0;
}
