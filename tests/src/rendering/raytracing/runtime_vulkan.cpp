#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <metra/metra.h>

#include "render_copy.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <vector>

using namespace rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

namespace {
    using DEVICE = devices::DefaultCPU;
    using BACKEND = rrt::backends::Vulkan;
    using RUNTIME = rrt::Runtime<BACKEND>;

    template <bool RGB>
    struct Config: rrt::config::Default<float, DEVICE::index_t>{
        static constexpr DEVICE::index_t CAM_WIDTH = 1, CAM_HEIGHT = 1, NUM_CAMERAS = 1, NUM_PROBES = 0;
        using OUTPUTS = typename rl_tools::rendering::raytracing::outputs::Concat<
            rl_tools::rendering::raytracing::outputs::If<(RGB), rl_tools::rendering::raytracing::outputs::RGB<>>,
            rl_tools::rendering::raytracing::outputs::If<(!RGB), rl_tools::rendering::raytracing::outputs::Depth<>>
        >::type;
        using SHADING = rrt::Low;
    };

    template <bool RGB>
    using SPEC = rrt::Specification<Config<RGB>>;

    template <typename SPEC>
    void check_frame(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, const rendering::Bundle<float>& bundle){
        init(device, renderer, bundle);
        const float position[3] = {0, 0, 0};
        const float target[3] = {3, 0, 0};
        const float up[3] = {0, 0, 1};
        const auto camera = make_camera_data(position, target, up, 80.0f, 1.0f);
        golden::copy_in(device, renderer.device, &camera, cameras(device, renderer));
        render(device, renderer);
        if constexpr(SPEC::HAS_RGB){
            std::vector<uint32_t> pixels;
            golden::copy_out(renderer.device, device, output(device, renderer, rl_tools::rendering::raytracing::outputs::RGB<>{}), pixels);
            EXPECT_EQ(pixels, std::vector<uint32_t>{0xff0000ffu});
        }
        else{
            std::vector<float> depth;
            golden::copy_out(renderer.device, device, output(device, renderer, rl_tools::rendering::raytracing::outputs::Depth<>{}), depth);
            ASSERT_EQ(depth.size(), 1);
            EXPECT_FLOAT_EQ(depth[0], 3.0f);
        }
    }

    template <typename SPEC>
    void check_cycle(DEVICE& device, RUNTIME& runtime, const rendering::Bundle<float>& bundle, bool render_frame){
        rrt::Renderer<SPEC, BACKEND> renderer;
        malloc(device, renderer, runtime);
        EXPECT_EQ(renderer.backend->physical_device, runtime.state.physical_device);
        EXPECT_EQ(renderer.backend->owned_runtime.state.instance, VK_NULL_HANDLE);
        if(render_frame){
            check_frame(device, renderer, bundle);
        }
        free(device, renderer);
    }
}

TEST(RENDERING_RAYTRACING_VULKAN_RUNTIME, REPEATED_RENDERERS_KEEP_THE_SELECTED_DEVICE){
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    RUNTIME runtime;
    malloc(device, runtime);
    const auto instance = runtime.state.instance;
    const auto physical_device = runtime.state.physical_device;
    ASSERT_NE(instance, VK_NULL_HANDLE);
    ASSERT_NE(physical_device, VK_NULL_HANDLE);

    rendering::Bundle<float> bundle;
    rrt::Mesh triangle;
    triangle.vertices = {3, -1, -1, 3, 1, -1, 3, 0, 1};
    triangle.indices = {0, 1, 2};
    triangle.color[0] = 1;
    triangle.color[1] = 0;
    triangle.color[2] = 0;
    add(device, bundle.scene, triangle);
    bundle.metadata.max_ray_length = 10;

    const auto start = std::chrono::steady_clock::now();
    for(int cycle = 0; cycle < 100; cycle++){
        SCOPED_TRACE(cycle);
        if(cycle == 50){
            rrt::Renderer<SPEC<true>, BACKEND> standalone;
            malloc(device, standalone);
            EXPECT_NE(standalone.backend->owned_runtime.state.instance, VK_NULL_HANDLE);
            EXPECT_NE(standalone.backend->owned_runtime.state.instance, instance);
            check_frame(device, standalone, bundle);
            free(device, standalone);
        }
        const bool render_frame = cycle < 2 || cycle == 50 || cycle == 51 || cycle >= 98;
        if(cycle % 2 == 0){
            check_cycle<SPEC<true>>(device, runtime, bundle, render_frame);
        }
        else{
            check_cycle<SPEC<false>>(device, runtime, bundle, render_frame);
        }
        EXPECT_EQ(runtime.state.instance, instance);
        EXPECT_EQ(runtime.state.physical_device, physical_device);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    free(device, runtime);
    EXPECT_EQ(runtime.state.instance, VK_NULL_HANDLE);
    EXPECT_EQ(runtime.state.physical_device, VK_NULL_HANDLE);
    metra::log("raytracing/runtime/vulkan/100-cycles-seconds", elapsed);
}
