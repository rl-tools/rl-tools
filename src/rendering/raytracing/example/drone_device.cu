// GPU-resident twin of drone.cpp: one CUDA kernel per frame computes the trajectory pose,
// prop spins, and camera pair and writes the renderer's device tensors (transforms_pair,
// cameras, cameras_open) in place on the render stream. The host's only per-frame role is
// enqueueing that kernel and the renderer verbs; the framebuffer readback exists solely to
// feed the mp4 pipes — a training loop would read observation() in place instead.
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cuda.h>
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>

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
using DEVICE_GPU = rlt::devices::DefaultCUDA;

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
    static constexpr bool ENABLE_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = 16;
    static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

static constexpr T GRAVITY = 9.81;
static constexpr T ONBOARD_FOV = SPEC::COS_FOVY;
static constexpr T THIRD_PERSON_FOV = 0.65;
static constexpr T CAMERA_MOUNT_X = 0.10, CAMERA_MOUNT_Y = 0, CAMERA_MOUNT_Z = 0.32; // scalars: constexpr arrays are not usable in device code
static constexpr T CAMERA_PITCH_DOWN = 0.3;
static constexpr T TRIPOD_X = -7.6, TRIPOD_Y = -7.0, TRIPOD_Z = 2.0;
static constexpr T PROP_RATE = 150.0;
static constexpr T SHUTTER_FRACTION = 0.5;

struct Options {
    std::string scene_path = "tests/data/ProcTHOR-Train-1.glb";
    std::string drone_path = "tests/data/x500.glb";
    std::string output_prefix = "drone_device";
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

RL_TOOLS_FUNCTION_PLACEMENT static void normalize3(const T v[3], T out[3]) {
    const T norm = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    out[0] = v[0] / norm; out[1] = v[1] / norm; out[2] = v[2] / norm;
}

RL_TOOLS_FUNCTION_PLACEMENT static void cross3(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

struct Pose {
    T position[3];
    T basis[3][3]; // world-frame body axes: basis[0] = x_b (forward), basis[1] = y_b (left), basis[2] = z_b (up)
};

RL_TOOLS_FUNCTION_PLACEMENT static void rotate_body_to_world(const Pose& pose, const T v[3], T out[3]) {
    for (TI dim = 0; dim < 3; dim++) {
        out[dim] = pose.basis[0][dim]*v[0] + pose.basis[1][dim]*v[1] + pose.basis[2][dim]*v[2];
    }
}

template <typename T_DEVICE>
RL_TOOLS_FUNCTION_PLACEMENT static Pose flat_pose(T_DEVICE& device, const lissajous::Parameters<T>& trajectory, const T center[3], T time) {
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
    normalize3(thrust, pose.basis[2]);

    const T heading[3] = {1, 0, 0};
    T left[3];
    cross3(pose.basis[2], heading, left);
    normalize3(left, pose.basis[1]);
    cross3(pose.basis[1], pose.basis[2], pose.basis[0]);
    return pose;
}

RL_TOOLS_FUNCTION_PLACEMENT static void pose_to_transform(const Pose& pose, float out[12]) {
    for (TI row = 0; row < 3; row++) {
        out[row * 4 + 0] = pose.basis[0][row];
        out[row * 4 + 1] = pose.basis[1][row];
        out[row * 4 + 2] = pose.basis[2][row];
        out[row * 4 + 3] = pose.position[row];
    }
}

RL_TOOLS_FUNCTION_PLACEMENT static void prop_spin_transform(T pivot_x, T pivot_y, T angle, float out[12]) {
    const float c = cosf(angle), s = sinf(angle);
    out[0] = c;  out[1] = -s; out[2]  = 0; out[3]  = pivot_x - c * pivot_x + s * pivot_y;
    out[4] = s;  out[5] = c;  out[6]  = 0; out[7]  = pivot_y - s * pivot_x - c * pivot_y;
    out[8] = 0;  out[9] = 0;  out[10] = 1; out[11] = 0;
}

RL_TOOLS_FUNCTION_PLACEMENT static void make_cameras(const Pose& pose, T aspect, rlt::rendering::raytracing::Camera<T> cameras[NUM_CAMERAS]) {
    {
        T mount[3], forward[3], up[3];
        const T mount_body[3] = {CAMERA_MOUNT_X, CAMERA_MOUNT_Y, CAMERA_MOUNT_Z};
        rotate_body_to_world(pose, mount_body, mount);
        const T pitch_cos = cosf(CAMERA_PITCH_DOWN), pitch_sin = sinf(CAMERA_PITCH_DOWN);
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
        const T tripod[3] = {TRIPOD_X, TRIPOD_Y, TRIPOD_Z};
        cameras[THIRD_PERSON_CAMERA] = rlt::make_camera_data(tripod, pose.position, up, THIRD_PERSON_FOV, aspect);
    }
}

static constexpr TI MAX_PROPS = SPEC::MAX_OVERLAY_INSTANCES;
struct DroneKernelParams {
    lissajous::Parameters<T> trajectory;
    T center[3];
    T aspect;
    TI body_slot;
    TI num_props;
    TI prop_slots[MAX_PROPS];
    T prop_pivots[MAX_PROPS][2];
    T prop_directions[MAX_PROPS];
};

// single drone -> single thread: the whole per-frame state (pose pair, prop articulation
// pairs, camera pair) is produced on-device; the expand/update/render verbs consume it with
// no host data path
template <typename T_DEVICE>
__global__ void drone_frame_kernel(T_DEVICE device, DroneKernelParams params, T time_open, T time_close,
                                   float* transforms_pair, rlt::rendering::raytracing::Camera<T>* cameras_open, rlt::rendering::raytracing::Camera<T>* cameras_close) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    constexpr TI SLOTS = SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
    const Pose pose_open = flat_pose(device, params.trajectory, params.center, time_open);
    const Pose pose_close = flat_pose(device, params.trajectory, params.center, time_close);

    pose_to_transform(pose_open, transforms_pair + params.body_slot * 12);
    pose_to_transform(pose_close, transforms_pair + (SLOTS + params.body_slot) * 12);
    for (TI prop_i = 0; prop_i < params.num_props; prop_i++) {
        const TI slot = params.prop_slots[prop_i];
        prop_spin_transform(params.prop_pivots[prop_i][0], params.prop_pivots[prop_i][1], params.prop_directions[prop_i] * PROP_RATE * time_open, transforms_pair + slot * 12);
        prop_spin_transform(params.prop_pivots[prop_i][0], params.prop_pivots[prop_i][1], params.prop_directions[prop_i] * PROP_RATE * time_close, transforms_pair + (SLOTS + slot) * 12);
    }

    make_cameras(pose_open, params.aspect, cameras_open);
    make_cameras(pose_close, params.aspect, cameras_close);
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
    DEVICE_GPU device_gpu;

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

    DroneKernelParams kernel_params{};
    kernel_params.trajectory = lissajous::default_parameters<T>;
    kernel_params.trajectory.A = 0.45;
    kernel_params.trajectory.B = 0.5;
    kernel_params.trajectory.C = 0.2;
    kernel_params.trajectory.ramp_duration = 0.0;
    for (TI dim = 0; dim < 3; dim++) {
        kernel_params.center[dim] = options.center[dim];
    }
    kernel_params.aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    kernel_params.body_slot = (TI)placement.first_slot;
    kernel_params.num_props = 0;
    for (size_t part_i = 0; part_i < drone_assembly.parts.size(); part_i++) {
        const auto& object = drone_assembly.objects[drone_assembly.parts[part_i].object];
        if (object.name.rfind("prop_", 0) != 0) continue;
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
        const TI prop_i = kernel_params.num_props++;
        kernel_params.prop_slots[prop_i] = (TI)(placement.first_slot + part_i);
        kernel_params.prop_pivots[prop_i][0] = (low[0] + high[0]) / 2;
        kernel_params.prop_pivots[prop_i][1] = (low[1] + high[1]) / 2;
        kernel_params.prop_directions[prop_i] = kernel_params.prop_pivots[prop_i][0] * kernel_params.prop_pivots[prop_i][1] > 0 ? 1 : -1;
    }

    // publish the spawn (slot structure + attachments) and clear the host-verb dirty state:
    // from here on the loop is producer-only, so update_launch never uploads anything
    rlt::update(device, renderer);

    const bool video = options.ffmpeg != "none";
    FILE* onboard_pipe = nullptr;
    FILE* third_person_pipe = nullptr;
    if (video) {
        onboard_pipe = open_video_pipe(options, options.output_prefix + "_onboard.mp4");
        third_person_pipe = open_video_pipe(options, options.output_prefix + "_third_person.mp4");
    }

    constexpr size_t CAM_PIXELS = static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT);
    std::vector<uint32_t> frame(NUM_CAMERAS * CAM_PIXELS);
    cudaStream_t render_stream = rlt::stream(device, renderer);
    float* transforms_pair = rlt::data(rlt::transforms_pair(device, renderer));
    auto* cameras_open = rlt::data(rlt::cameras_open(device, renderer));
    auto* cameras_close = rlt::data(rlt::cameras_close(device, renderer));

    for (int frame_i = 0; frame_i < options.frames; frame_i++) {
        const T time_open = static_cast<T>(frame_i / options.fps);
        const T time_close = static_cast<T>((frame_i + SHUTTER_FRACTION) / options.fps);

        // enqueue-only frame: producer kernel on the render stream, then the renderer verbs
        drone_frame_kernel<<<1, 1, 0, render_stream>>>(device_gpu, kernel_params, time_open, time_close, transforms_pair, cameras_open, cameras_close);
        rlt::expand_motion_transforms_launch(device, renderer);
        rlt::update_launch(device, renderer);
        rlt::render_launch(device, renderer);
        rlt::render_sync(device, renderer);

        // consumer only: the video pipe needs the frames on the host
        cudaMemcpy(frame.data(), rlt::data(rlt::frame_buffer(device, renderer)), frame.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

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
            stbi_write_png(snapshot_path, CAM_WIDTH, NUM_CAMERAS * CAM_HEIGHT, 4, frame.data(), CAM_WIDTH * 4);
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
