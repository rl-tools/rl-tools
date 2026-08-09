#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace lissajous = rl_tools::rl::environments::l2f::parameters::trajectories::lissajous;

using T = float;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 960;
static constexpr TI CAM_HEIGHT = 540;
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
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr T GRAVITY = 9.81;
static constexpr T ONBOARD_FOV = SPEC::COS_FOVY;
static constexpr T THIRD_PERSON_FOV = 0.65;
static constexpr T CAMERA_MOUNT_BODY[3] = {0.10, 0, 0.32};
static constexpr T CAMERA_PITCH_DOWN = 0.3;
static constexpr T TRIPOD_POSITION[3] = {-7.6, -7.0, 2.0};
static constexpr T PROP_RATE = 45.0;

struct Options {
    std::string scene_path = "tests/data/ProcTHOR-Train-1.glb";
    std::string drone_path = "tests/data/x500.glb";
    std::string output_prefix = "drone";
    std::string ffmpeg = "ffmpeg";
    T center[3] = {-5.0, -5.2, 1.55};
    int frames = 390;
    double fps = 60.0;
    int snapshot = 0;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            options.scene_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--drone") == 0 && i + 1 < argc) {
            options.drone_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--output-prefix") == 0 && i + 1 < argc) {
            options.output_prefix = argv[++i];
        }
        else if (std::strcmp(argv[i], "--ffmpeg") == 0 && i + 1 < argc) {
            options.ffmpeg = argv[++i];
        }
        else if (std::strcmp(argv[i], "--center") == 0 && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%f,%f,%f", &options.center[0], &options.center[1], &options.center[2]) != 3) {
                std::cerr << "Invalid value for --center (expected x,y,z)" << std::endl;
                std::exit(1);
            }
        }
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            options.frames = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            options.fps = std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--snapshot") == 0 && i + 1 < argc) {
            options.snapshot = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [--scene <path.glb>] [--drone <path.glb>] [--output-prefix <prefix>] [--ffmpeg <path>|none] [--center <x,y,z>] [--frames <n>] [--fps <fps>] [--snapshot <every-n>]\n";
            std::exit(0);
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::exit(1);
        }
    }
    return options;
}

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

static Pose flat_pose(DEVICE& device, const lissajous::Parameters<T>& trajectory, const T center[3], T time, T& last_yaw) {
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

    const T speed_xy = std::sqrt(step.linear_velocity[0]*step.linear_velocity[0] + step.linear_velocity[1]*step.linear_velocity[1]);
    if (speed_xy > 1e-3) {
        last_yaw = std::atan2(step.linear_velocity[1], step.linear_velocity[0]);
    }
    const T heading[3] = {std::cos(last_yaw), std::sin(last_yaw), 0};
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

static void upload_cameras(DEVICE& device, Renderer& renderer, const rlt::rendering::raytracing::Camera<T> cameras[NUM_CAMERAS]) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    cudaMemcpy(rlt::data(rlt::cameras(device, renderer)), cameras, NUM_CAMERAS * sizeof(cameras[0]), cudaMemcpyHostToDevice);
#else
    std::memcpy(rlt::data(rlt::cameras(device, renderer)), cameras, NUM_CAMERAS * sizeof(cameras[0]));
#endif
}

static FILE* open_video_pipe(const Options& options, const std::string& output_path) {
    std::ostringstream command;
    command
        << options.ffmpeg << " -y -hide_banner -loglevel error -f rawvideo -pixel_format rgba "
        << "-video_size " << CAM_WIDTH << "x" << CAM_HEIGHT << " "
        << "-framerate " << options.fps << " -i - "
        << "-an -c:v libx264 -pix_fmt yuv420p "
        << output_path;
    FILE* pipe = popen(command.str().c_str(), "w");
    if (pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg: " << command.str() << std::endl;
        std::exit(1);
    }
    return pipe;
}

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    if (!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        return 1;
    }

    rlt::rendering::raytracing::ObjectAssembly drone_assembly;
    if (!rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, drone_assembly, options.drone_path)) {
        std::cerr << "Failed to load drone assembly: " << options.drone_path << std::endl;
        return 1;
    }
    // node origins are not required to sit at the rotor hubs (e.g. generated meshes with
    // identity node transforms), so each prop spins about its part-local AABB center
    struct PropPart {
        TI part;
        T pivot[2];
        T direction;
        T angle;
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
            prop.angle = 0;
            props.push_back(prop);
        }
        else if (object.name == "body" && part_i != 0) {
            std::cerr << "Drone assembly part 0 must be the body (the pose part), got it at part " << part_i << std::endl;
            return 1;
        }
    }
    if (drone_assembly.parts.empty() || drone_assembly.objects[drone_assembly.parts[0].object].name != "body") {
        std::cerr << "Drone assembly must have a scene-root node named 'body' as its first part" << std::endl;
        return 1;
    }

    rlt::rendering::raytracing::AssetPool pool;
    const auto drone_asset = rlt::add(device, pool, drone_assembly);

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
    trajectory.ramp_duration = 1.0;

    const bool video = options.ffmpeg != "none";
    FILE* onboard_pipe = nullptr;
    FILE* third_person_pipe = nullptr;
    if (video) {
        onboard_pipe = open_video_pipe(options, options.output_prefix + "_onboard.mp4");
        third_person_pipe = open_video_pipe(options, options.output_prefix + "_third_person.mp4");
    }

    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    constexpr size_t CAM_PIXELS = static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT);
    std::vector<uint32_t> frame(NUM_CAMERAS * CAM_PIXELS);
    T last_yaw = 0;

    for (int frame_i = 0; frame_i < options.frames; frame_i++) {
        const T time = static_cast<T>(frame_i / options.fps);
        const Pose pose = flat_pose(device, trajectory, options.center, time, last_yaw);

        float pose_transform[12];
        pose_to_transform(pose, pose_transform);
        rlt::set_transform(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, pose_transform);
        for (auto& prop : props) {
            prop.angle += prop.direction * PROP_RATE / options.fps;
            const float c = std::cos(prop.angle), s = std::sin(prop.angle);
            const float px = prop.pivot[0], py = prop.pivot[1];
            const float spin[12] = {
                c, -s, 0, px - c * px + s * py,
                s,  c, 0, py - s * px - c * py,
                0,  0, 1, 0,
            };
            rlt::set_transform(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, prop.part, spin);
        }

        rlt::rendering::raytracing::Camera<T> cameras[NUM_CAMERAS];
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
        upload_cameras(device, renderer, cameras);

        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        cudaMemcpy(frame.data(), rlt::data(rlt::frame_buffer(device, renderer)), frame.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
#else
        std::memcpy(frame.data(), rlt::data(rlt::frame_buffer(device, renderer)), frame.size() * sizeof(uint32_t));
#endif
        if (video) {
            const size_t bytes = CAM_PIXELS * sizeof(uint32_t);
            if (std::fwrite(frame.data(), 1, bytes, onboard_pipe) != bytes
                || std::fwrite(frame.data() + CAM_PIXELS, 1, bytes, third_person_pipe) != bytes) {
                std::cerr << "Failed writing frame " << frame_i << " to ffmpeg." << std::endl;
                return 1;
            }
        }
        if (options.snapshot > 0 && frame_i % options.snapshot == 0) {
            char snapshot_path[256];
            std::snprintf(snapshot_path, sizeof(snapshot_path), "%s_frame_%05d.png", options.output_prefix.c_str(), frame_i);
            rlt::save_image(device, renderer, snapshot_path);
        }
        if (frame_i % 60 == 0) {
            std::cout << "frame " << frame_i << "/" << options.frames << std::endl;
        }
    }

    if (video) {
        const int onboard_status = pclose(onboard_pipe);
        const int third_person_status = pclose(third_person_pipe);
        if (onboard_status != 0 || third_person_status != 0) {
            std::cerr << "ffmpeg exited with status onboard=" << onboard_status << " third_person=" << third_person_status << std::endl;
            return 1;
        }
        std::cout << "Wrote " << options.output_prefix << "_onboard.mp4 and " << options.output_prefix << "_third_person.mp4" << std::endl;
    }

    rlt::free(device, renderer);
    return 0;
}
