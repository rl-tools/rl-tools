#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0

#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH 2

#ifndef RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"
#include <rl_tools/rendering/raytracing/save_cpu.h>


#include <iostream>
#include <chrono>
#include <type_traits>
#include <cmath>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <array>

namespace rlt = rl_tools;

static constexpr bool OUTPUT_RGB = RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH;
static constexpr bool OUTPUT_DEPTH = RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB;

int main(int argc, char** argv) {
    bool output_video = false;
    bool no_probe = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--output-video") == 0) {
            output_video = true;
        }
        else if (std::strcmp(argv[i], "--no-probe") == 0) {
            no_probe = true;
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    using T = float;
    using TI = typename rlt::devices::DEVICE_FACTORY<>::index_t;
    static constexpr TI NUM_ENVS = 256;
    constexpr T PI = static_cast<T>(3.14159265358979323846);
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, 64, 64, 64, rlt::rendering::raytracing::Low, false, 1, false, 1, OUTPUT_RGB, OUTPUT_DEPTH>;

    static_assert(std::is_standard_layout_v<rlt::rl::environments::raytracing_example::Parameters<SPEC>>);
    static_assert(std::is_trivially_copyable_v<rlt::rl::environments::raytracing_example::Parameters<SPEC>>);
    static_assert(std::is_standard_layout_v<rlt::rl::environments::raytracing_example::State<SPEC>>);
    static_assert(std::is_trivially_copyable_v<rlt::rl::environments::raytracing_example::State<SPEC>>);

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::init(device, rng, 0);

    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = "tests/data/ProcTHOR-Train-1.glb";

    using PARAMETERS_SPEC = rlt::tensor::Specification<rlt::rl::environments::raytracing_example::Parameters<SPEC>, TI, rlt::tensor::Shape<TI, NUM_ENVS>>;
    using STATE_SPEC = rlt::tensor::Specification<rlt::rl::environments::raytracing_example::State<SPEC>, TI, rlt::tensor::Shape<TI, NUM_ENVS>>;
    using PIXELS_SPEC = rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, NUM_ENVS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>>;

    rlt::Tensor<PARAMETERS_SPEC> parameters;
    rlt::Tensor<STATE_SPEC> states;
    rlt::Tensor<PIXELS_SPEC> pixels;
    std::vector<rlt::rl::environments::raytracing_example::State<SPEC>> base_states(NUM_ENVS);

    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, pixels);

    rlt::malloc(device, env);
    rlt::init(device, env);

    for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
        rlt::rl::environments::raytracing_example::Parameters<SPEC> p;
        rlt::sample_initial_parameters(device, env, p, rng);
        rlt::rl::environments::raytracing_example::State<SPEC> s;
        rlt::sample_initial_state(device, env, p, s, rng);
        // const T angle = static_cast<T>(env_i) * static_cast<T>(0.01);
        // s.position[0] = -0.937;
        // s.position[1] = 1.690;
        // s.position[2] = 8.410;
        // s.velocity[0] = 0;
        // s.velocity[1] = 0;
        // s.velocity[2] = 0;
        // s.yaw = angle + static_cast<T>(3.14159265358979323846) / static_cast<T>(2.0);
        rlt::set(device, parameters, p, env_i);
        rlt::set(device, states, s, env_i);
        base_states[env_i] = s;
    }


    constexpr TI STEPS = 1024*4;
#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    if (output_video) {
        std::cerr << "--output-video requires RGB or RGBD output mode." << std::endl;
        return 1;
    }
#else
    FILE* mp4_pipe = nullptr;
    std::vector<uint32_t> per_camera_rgba;
    std::vector<uint32_t> megaframe_rgba;
    if (output_video) {
        constexpr TI GRID_WIDTH = SPEC::RAYTRACING_SPEC::GRID_COLS * SPEC::CAM_WIDTH;
        constexpr TI GRID_HEIGHT = SPEC::RAYTRACING_SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
        per_camera_rgba.resize(static_cast<size_t>(NUM_ENVS) * static_cast<size_t>(SPEC::CAM_WIDTH) * static_cast<size_t>(SPEC::CAM_HEIGHT));
        megaframe_rgba.resize(static_cast<size_t>(GRID_WIDTH) * static_cast<size_t>(GRID_HEIGHT));

        const double frame_rate = env.dt > 0 ? (1.0 / static_cast<double>(env.dt)) : 30.0;
        std::ostringstream ffmpeg_cmd;
        ffmpeg_cmd
            << "ffmpeg -y -f rawvideo -pixel_format rgba "
            << "-video_size " << GRID_WIDTH << "x" << GRID_HEIGHT << " "
            << "-framerate " << frame_rate << " -i - "
            << "-an -c:v libx264 -pix_fmt yuv420p "
            << "raytracing_example_megaframe.mp4";
        mp4_pipe = popen(ffmpeg_cmd.str().c_str(), "w");
        if (mp4_pipe == nullptr) {
            std::cerr << "Failed to start ffmpeg." << std::endl;
            return 1;
        }
    }
#endif

    // camera staging fills while the GPU renders; upload_cameras pushes it into the
    // backend-native cameras tensor once the previous launch has synced
    std::vector<rlt::rendering::raytracing::Camera<T>> camera_staging(NUM_ENVS);
    auto compute_states_and_cameras = [&](TI step_i) {
        for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
            const T yaw_phase = static_cast<T>(0.02 * env_i + 0.08 * step_i);
            auto s = base_states[env_i];
            s.velocity[0] = 0;
            s.velocity[1] = 0;
            s.velocity[2] = 0;
            s.yaw = static_cast<T>(0.5) * PI * (static_cast<T>(1) + std::sin(yaw_phase));
            rlt::set(device, states, s, env_i);
            camera_staging[env_i] = rlt::make_camera_for_state(env, rlt::get_ref(device, parameters, env_i), s);
        }
    };
    auto upload_cameras = [&]() {
        rlt::Tensor<typename decltype(env.renderer->cameras)::SPEC> camera_alias;
        camera_alias._data = camera_staging.data();
        rlt::copy(device, env.renderer->device, camera_alias, rlt::cameras(device, *env.renderer));
    };

    auto do_video_output = [&](TI step_i) -> int {
#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
        (void)step_i;
        return 0;
#else
        if (mp4_pipe == nullptr) return 0;
        constexpr size_t pixel_count = (size_t)NUM_ENVS * SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        static std::vector<uint32_t> frame_staging(pixel_count);
        rlt::Tensor<typename decltype(env.renderer->frame_buffer)::SPEC> frame_alias;
        frame_alias._data = frame_staging.data();
        rlt::copy(env.renderer->device, device, rlt::frame_buffer(device, *env.renderer), frame_alias);
        const uint32_t* per_camera_rgba_ptr = frame_staging.data();
        for (TI camera_i = 0; camera_i < NUM_ENVS; camera_i++) {
            const TI col = camera_i % SPEC::RAYTRACING_SPEC::GRID_COLS;
            const TI row = camera_i / SPEC::RAYTRACING_SPEC::GRID_COLS;
            const TI offset_x = col * SPEC::CAM_WIDTH;
            const TI offset_y = row * SPEC::CAM_HEIGHT;
            for (TI y = 0; y < SPEC::CAM_HEIGHT; y++) {
                const uint32_t* src = per_camera_rgba_ptr + static_cast<size_t>(camera_i) * SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT + static_cast<size_t>(y) * SPEC::CAM_WIDTH;
                uint32_t* dst = megaframe_rgba.data() + static_cast<size_t>(offset_y + y) * (SPEC::RAYTRACING_SPEC::GRID_COLS * SPEC::CAM_WIDTH) + offset_x;
                std::memcpy(dst, src, static_cast<size_t>(SPEC::CAM_WIDTH) * sizeof(uint32_t));
            }
        }
        const size_t bytes = megaframe_rgba.size() * sizeof(uint32_t);
        const size_t written = fwrite(megaframe_rgba.data(), 1, bytes, mp4_pipe);
        if (written != bytes) {
            std::cerr << "Failed writing frame " << step_i << " to ffmpeg." << std::endl;
            pclose(mp4_pipe);
            mp4_pipe = nullptr;
            return 1;
        }
        return 0;
#endif
    };

    rlt::synchronize(device, *env.renderer);
    auto t0 = std::chrono::steady_clock::now();

    // Pipelined loop: overlap CPU camera computation with GPU rendering
    compute_states_and_cameras(0);
    upload_cameras();

    for (TI step_i = 0; step_i < STEPS; step_i++) {
        if (no_probe) {
#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
            rlt::render_launch(device, *env.renderer);
#elif RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
            rlt::render_launch(device, *env.renderer);
#else
            rlt::render_launch(device, *env.renderer);
#endif
        }
        else {
            rlt::render_launch(device, *env.renderer);
        }

        // Overlap: compute next frame's cameras while GPU renders current frame
        if (step_i + 1 < STEPS) {
            compute_states_and_cameras(step_i + 1);
        }

        if (no_probe) {
#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
            rlt::render_sync(device, *env.renderer);
#elif RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
            rlt::render_sync(device, *env.renderer);
#else
            rlt::render_sync(device, *env.renderer);
#endif
        }
        else {
            rlt::render_sync(device, *env.renderer);
        }

        if (int err = do_video_output(step_i)) return err;

        // Upload next frame's cameras (GPU is idle now, buffer is safe to overwrite)
        if (step_i + 1 < STEPS) {
            upload_cameras();
        }
    }
    rlt::synchronize(device, *env.renderer);
    auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const double fps = (static_cast<double>(NUM_ENVS) * STEPS) / elapsed;
    std::cout << "raytracing_example: " << NUM_ENVS << " envs, " << STEPS << " batched observe steps" << std::endl;
    std::cout << "elapsed: " << elapsed << " s, effective frame throughput: " << fps << " frames/s" << std::endl;

#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    if (mp4_pipe != nullptr) {
        const int ffmpeg_status = pclose(mp4_pipe);
        mp4_pipe = nullptr;
        if (ffmpeg_status != 0) {
            std::cerr << "ffmpeg exited with status " << ffmpeg_status << std::endl;
            return 1;
        }
        std::cout << "video written: raytracing_example_megaframe.mp4" << std::endl;
    }
#endif

#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_DEPTH
    rlt::save_image(device, *env.renderer, "raytracing_example_grid.png");
#endif
#if RL_TOOLS_RENDERING_RAYTRACING_EXAMPLE_OUTPUT_MODE != RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
    rlt::save_depth_image(device, *env.renderer, "raytracing_example_depth.png");
    rlt::save_depth(device, *env.renderer, "raytracing_example_depth.bin");
#endif

    rlt::free(device, env);
    rlt::free(device, pixels);
    rlt::free(device, states);
    rlt::free(device, parameters);
    return 0;
}
