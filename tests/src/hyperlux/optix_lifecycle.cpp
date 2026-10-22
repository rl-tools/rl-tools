#include <rl_tools/operations/cpu.h>
#include <rl_tools/hyperlux/operations_cpu_mux.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>

namespace {
    using namespace rl_tools;
    namespace rrt = hyperlux;
    using DEVICE = devices::DefaultCPU;
    using TI = DEVICE::index_t;
    struct Config: rrt::config::Default<float, TI> {
        using SHADING = rrt::High;
        static constexpr TI CAM_WIDTH = 64, CAM_HEIGHT = 48, NUM_CAMERAS = 1;
    };
    using SPEC = rrt::Specification<Config>;
    using BACKEND = rrt::backends::Optix;
    static_assert(SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING, "PBR geometry references the world group");
    constexpr int CYCLES = 8;
    constexpr std::size_t TOLERANCE_BYTES = 4 << 20;

    std::size_t device_memory_used() {
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::size_t free_bytes = 0, total_bytes = 0;
        EXPECT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
        return total_bytes - free_bytes;
    }

    hyperlux::Bundle<float> scene(DEVICE& device) {
        hyperlux::Bundle<float> bundle;
        rrt::Mesh triangle;
        triangle.vertices = {3, -100, -100, 3, 100, -100, 3, 0, 100};
        triangle.indices = {0, 1, 2};
        add(device, bundle.scene, triangle);
        bundle.metadata.max_ray_length = 20;
        bundle.metadata.content_hash = "optix-lifecycle-triangle";
        return bundle;
    }

    template <typename CYCLE>
    void expect_released(CYCLE cycle) {
        cycle();
        const std::size_t baseline = device_memory_used();
        for(int cycle_i = 0; cycle_i < CYCLES; cycle_i++) {
            cycle();
        }
        EXPECT_LE(device_memory_used(), baseline + TOLERANCE_BYTES);
    }
}

TEST(HYPERLUX_OPTIX_LIFECYCLE, RendererFreeReleasesScene) {
    DEVICE device;
    init(device);
    const auto bundle = scene(device);
    expect_released([&] {
        rl_tools::HyperLux<SPEC, BACKEND> renderer;
        malloc(device, renderer);
        init(device, renderer, bundle);
        free(device, renderer);
    });
}

TEST(HYPERLUX_OPTIX_LIFECYCLE, LibraryFreeReleasesScenes) {
    DEVICE device;
    init(device);
    const auto bundle = scene(device);
    expect_released([&] {
        rrt::AssetLibrary<SPEC, BACKEND> library;
        malloc(device, library);
        auto inserted = bundle;
        const TI scene_id = insert(device, library, inserted);
        rl_tools::HyperLux<SPEC, BACKEND> renderer;
        malloc(device, renderer, library);
        init(device, renderer, library, scene_id);
        free(device, renderer);
        free(device, library);
    });
}
