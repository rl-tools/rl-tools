#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>


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
using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

static constexpr TI CAM_WIDTH = 256;
static constexpr TI CAM_HEIGHT = 144;
static constexpr TI NUM_CAMERAS = 1;
static constexpr TI NUM_PROBES = 1;
static constexpr TI NUM_PANELS = 6;

template <TI SAMPLES>
using RendererSpec = rlt::rendering::raytracing::Specification<T, TI, CAM_WIDTH, CAM_HEIGHT, NUM_CAMERAS, NUM_PROBES, rlt::rendering::raytracing::Medium, (SAMPLES > 1), SAMPLES>;

template <typename SPEC>
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

struct Options {
    std::string scene_path = "tests/data/ProcTHOR-Train-1.glb";
    std::string output_path = "raytracing_motion_blur_levels.mp4";
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
void setup_renderer(DEVICE& device, Renderer<SPEC>& renderer, const rlt::rendering::raytracing::Scene& scene) {
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, scene);

    const T up[3] = {0, 0, 1};
    rlt::generate_cameras(device, renderer, renderer.scene_center, renderer.camera_radius, up, SPEC::COS_FOVY);
}

rlt::rendering::raytracing::Camera<T> make_orbit_camera(const Renderer<RendererSpec<1>>& renderer, T angle) {
    const T center[3] = {
        renderer.scene_center[0],
        renderer.scene_center[1],
        renderer.scene_center[2]
    };
    const T radius = renderer.camera_radius > T{0} ? renderer.camera_radius * T{0.65} : T{3};
    const T position[3] = {
        center[0] + radius * std::cos(angle),
        center[1] + radius * std::sin(angle),
        center[2] + renderer.scene_half_extent[2] * T{0.25} + radius * T{0.15}
    };
    const T look_at[3] = {
        center[0],
        center[1],
        center[2] + renderer.scene_half_extent[2] * T{0.10}
    };
    const T up[3] = {0, 0, 1};
    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    return rlt::make_camera_data(position, look_at, up, RendererSpec<1>::COS_FOVY, aspect);
}

template <typename SPEC>
void render_panel(DEVICE& device, Renderer<SPEC>& renderer, const rlt::rendering::raytracing::Camera<T>& camera_open, const rlt::rendering::raytracing::Camera<T>& camera_close, std::vector<uint32_t>& panel) {
    if constexpr (SPEC::ENABLE_MOTION_BLUR) {
        rlt::set(device, renderer.cameras_open, camera_open, static_cast<TI>(0));
        rlt::set(device, renderer.cameras, camera_close, static_cast<TI>(0));
        rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
    }
    else {
        rlt::set(device, renderer.cameras, camera_close, static_cast<TI>(0));
        rlt::set_cameras(device, renderer, renderer.cameras);
    }

    rlt::render_rgb_only(device, renderer);
    rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);

    const uint32_t* src = rlt::data(renderer.frame_buffer);
    panel.assign(src, src + static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT));
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
    Renderer<RendererSpec<4>> renderer_4;
    Renderer<RendererSpec<8>> renderer_8;
    Renderer<RendererSpec<16>> renderer_16;
    Renderer<RendererSpec<32>> renderer_32;

    rlt::rendering::raytracing::Scene scene;
    if (!rlt::load<rlt::rendering::raytracing::Medium, true>(device, scene, options.scene_path)) {
        std::cerr << "Failed to load scene: " << options.scene_path << std::endl;
        return 1;
    }
    setup_renderer(device, renderer_1, scene);
    setup_renderer(device, renderer_2, scene);
    setup_renderer(device, renderer_4, scene);
    setup_renderer(device, renderer_8, scene);
    setup_renderer(device, renderer_16, scene);
    setup_renderer(device, renderer_32, scene);

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
    std::cout << "Panels left-to-right: 1, 2, 4, 8, 16, 32 samples" << std::endl;

    std::vector<uint32_t> panel;
    std::vector<uint32_t> frame(static_cast<size_t>(OUT_WIDTH) * static_cast<size_t>(OUT_HEIGHT));

    const T angle_per_frame = T{0.030};
    const T shutter_angle = T{0.220};

    for (int frame_i = 0; frame_i < options.frames; frame_i++) {
        const T close_angle = static_cast<T>(frame_i) * angle_per_frame;
        const auto camera_open = make_orbit_camera(renderer_1, close_angle - shutter_angle);
        const auto camera_close = make_orbit_camera(renderer_1, close_angle);

        render_panel(device, renderer_1, camera_open, camera_close, panel);
        copy_panel(panel, frame, 0);
        render_panel(device, renderer_2, camera_open, camera_close, panel);
        copy_panel(panel, frame, 1);
        render_panel(device, renderer_4, camera_open, camera_close, panel);
        copy_panel(panel, frame, 2);
        render_panel(device, renderer_8, camera_open, camera_close, panel);
        copy_panel(panel, frame, 3);
        render_panel(device, renderer_16, camera_open, camera_close, panel);
        copy_panel(panel, frame, 4);
        render_panel(device, renderer_32, camera_open, camera_close, panel);
        copy_panel(panel, frame, 5);

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

    rlt::free(device, renderer_32);
    rlt::free(device, renderer_16);
    rlt::free(device, renderer_8);
    rlt::free(device, renderer_4);
    rlt::free(device, renderer_2);
    rlt::free(device, renderer_1);

    std::cout << "video written: " << options.output_path << std::endl;
    return 0;
}
