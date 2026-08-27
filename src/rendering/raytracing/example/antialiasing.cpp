#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using T = float;
static constexpr T FOV = 80;
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 80;
static constexpr TI CAM_HEIGHT = 50;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;
static constexpr TI NUM_PANELS = 4;

template <TI AA_GRID>
struct RendererConfig: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = ::CAM_WIDTH, CAM_HEIGHT = ::CAM_HEIGHT, NUM_CAMERAS = ::NUM_CAMERAS, NUM_PROBES = ::NUM_PROBES;
    static constexpr bool ENABLE_ANTI_ALIASING = AA_GRID > 1;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = AA_GRID;
};
template <TI AA_GRID>
using RendererSpec = rlt::rendering::raytracing::Specification<RendererConfig<AA_GRID>>;

template <typename SPEC>
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

struct Options {
    std::string scene_path = "tests/data/ProcTHOR-Train-1.glb";
    std::string output_path = "raytracing_antialiasing_levels.mp4";
    int frames = 120;
    double fps = 30.0;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i++) {
        if ((std::strcmp(argv[i], "--scene") == 0 || std::strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            options.scene_path = argv[++i];
        }
        else if ((std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            options.output_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            options.frames = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            options.fps = std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [--scene <path.glb>] [--output <path.mp4>] [--frames <n>] [--fps <fps>]\n";
            std::exit(0);
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::exit(1);
        }
    }
    return options;
}

template <typename SPEC>
void setup_renderer(DEVICE& device, Renderer<SPEC>& renderer, const rlt::rendering::Bundle<T>& bundle) {
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, bundle);

    const T up[3] = {0, 0, 1};
    rlt::generate_cameras(device, renderer, bundle.metadata.center, (bundle.metadata.max_ray_length / 2), up, FOV);
}

rlt::rendering::raytracing::Camera<T> make_orbit_camera(const rlt::rendering::SceneMetadata<T>& metadata, T angle) {
    const T center[3] = {
        metadata.center[0],
        metadata.center[1],
        metadata.center[2]
    };
    const T camera_radius = (metadata.max_ray_length / 2);
    const T radius = camera_radius > T{0} ? camera_radius * T{0.62} : T{3};
    const T position[3] = {
        center[0] + radius * std::cos(angle),
        center[1] + radius * std::sin(angle),
        center[2] + metadata.half_extent[2] * T{0.18} + radius * T{0.12}
    };
    const T look_at[3] = {
        center[0],
        center[1],
        center[2] + metadata.half_extent[2] * T{0.06}
    };
    const T up[3] = {0, 0, 1};
    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    return rlt::make_camera_data(position, look_at, up, FOV, aspect);
}

template <typename SPEC>
void render_panel(DEVICE& device, Renderer<SPEC>& renderer, const rlt::rendering::raytracing::Camera<T>& camera, std::vector<uint32_t>& panel) {
    auto camera_staging = camera;
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
    camera_alias._data = &camera_staging;
    rlt::copy(device, renderer.device, camera_alias, rlt::cameras(device, renderer));

    rlt::render(device, renderer);

    panel.resize(static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT));
    rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
    frame_alias._data = panel.data();
    rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
}

void copy_panel(const std::vector<uint32_t>& panel, std::vector<uint32_t>& frame, TI panel_i) {
    constexpr TI OUT_WIDTH = CAM_WIDTH * NUM_PANELS;
    for (TI y = 0; y < CAM_HEIGHT; y++) {
        const uint32_t* src = panel.data() + static_cast<size_t>(y) * CAM_WIDTH;
        uint32_t* dst = frame.data() + static_cast<size_t>(y) * OUT_WIDTH + static_cast<size_t>(panel_i) * CAM_WIDTH;
        std::memcpy(dst, src, static_cast<size_t>(CAM_WIDTH) * sizeof(uint32_t));
    }
}

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    DEVICE device;
    rlt::init(device);

    Renderer<RendererSpec<1>> renderer_1;
    Renderer<RendererSpec<2>> renderer_2;
    Renderer<RendererSpec<3>> renderer_3;
    Renderer<RendererSpec<4>> renderer_4;

    rlt::rendering::Bundle<T> bundle;
    if (!rlt::load<rlt::rendering::raytracing::Medium, true>(device, bundle, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        return 1;
    }
    setup_renderer(device, renderer_1, bundle);
    setup_renderer(device, renderer_2, bundle);
    setup_renderer(device, renderer_3, bundle);
    setup_renderer(device, renderer_4, bundle);

    constexpr TI OUT_WIDTH = CAM_WIDTH * NUM_PANELS;
    constexpr TI OUT_HEIGHT = CAM_HEIGHT;

    std::ostringstream ffmpeg_cmd;
    ffmpeg_cmd
        << "ffmpeg -y -f rawvideo -pixel_format rgba "
        << "-video_size " << OUT_WIDTH << "x" << OUT_HEIGHT << " "
        << "-framerate " << options.fps << " -i - "
        << "-an -c:v libx264 -pix_fmt yuv420p "
        << options.output_path;

    FILE* mp4_pipe = popen(ffmpeg_cmd.str().c_str(), "w");
    if (mp4_pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg." << std::endl;
        return 1;
    }

    std::cout << "Writing " << options.output_path << std::endl;
    std::cout << "Panels left-to-right: 1x, 2x2, 3x3, 4x4 anti-aliasing" << std::endl;

    std::vector<uint32_t> panel;
    std::vector<uint32_t> frame(static_cast<size_t>(OUT_WIDTH) * static_cast<size_t>(OUT_HEIGHT));

    const T angle_per_frame = T{0.018};

    for (int frame_i = 0; frame_i < options.frames; frame_i++) {
        const T angle = static_cast<T>(frame_i) * angle_per_frame;
        const auto camera = make_orbit_camera(bundle.metadata, angle);

        render_panel(device, renderer_1, camera, panel);
        copy_panel(panel, frame, 0);
        render_panel(device, renderer_2, camera, panel);
        copy_panel(panel, frame, 1);
        render_panel(device, renderer_3, camera, panel);
        copy_panel(panel, frame, 2);
        render_panel(device, renderer_4, camera, panel);
        copy_panel(panel, frame, 3);

        const size_t bytes = frame.size() * sizeof(uint32_t);
        const size_t written = std::fwrite(frame.data(), 1, bytes, mp4_pipe);
        if (written != bytes) {
            std::cerr << "Failed writing frame " << frame_i << " to ffmpeg." << std::endl;
            pclose(mp4_pipe);
            return 1;
        }
    }

    const int ffmpeg_status = pclose(mp4_pipe);
    if (ffmpeg_status != 0) {
        std::cerr << "ffmpeg exited with status " << ffmpeg_status << std::endl;
        return 1;
    }

    rlt::free(device, renderer_4);
    rlt::free(device, renderer_3);
    rlt::free(device, renderer_2);
    rlt::free(device, renderer_1);

    std::cout << "Wrote " << options.output_path << std::endl;
    return 0;
}
