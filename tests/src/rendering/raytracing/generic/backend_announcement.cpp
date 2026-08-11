#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>

#include <gtest/gtest.h>

#include <string>

namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

namespace {
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;

    struct CONFIG: rrt::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = 1;
        static constexpr TI CAM_HEIGHT = 1;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = 1;
        static constexpr bool OUTPUT_RGB = false;
        static constexpr bool OUTPUT_DEPTH = true;
        using SHADING = rrt::Low;
    };

    using SPEC = rrt::Specification<CONFIG>;
    using RENDERER = rrt::Renderer<SPEC, rrt::backends::Generic>;
}

TEST(RENDERING_RAYTRACING_BACKEND_ANNOUNCEMENT, CANONICAL_NAMES){
    EXPECT_STREQ(rrt::backends::name<rrt::backends::None>(), "none");
    EXPECT_STREQ(rrt::backends::name<rrt::backends::Generic>(), "generic");
    EXPECT_STREQ(rrt::backends::name<rrt::backends::Optix>(), "optix");
    EXPECT_STREQ(rrt::backends::name<rrt::backends::Metal>(), "metal");
    EXPECT_STREQ(rrt::backends::name<rrt::backends::Vulkan>(), "vulkan");
}

TEST(RENDERING_RAYTRACING_BACKEND_ANNOUNCEMENT, EACH_MALLOC_WRITES_TO_STDERR){
    DEVICE device;
    rlt::init(device);
    RENDERER first;
    RENDERER second;

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    rlt::malloc(device, first);
    rlt::malloc(device, second);
    const std::string stderr_output = testing::internal::GetCapturedStderr();
    const std::string stdout_output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(
        stderr_output,
        "#rl_tools::rendering::raytracing: backend=generic\n"
        "#rl_tools::rendering::raytracing: backend=generic\n"
    );
    EXPECT_TRUE(stdout_output.empty());

    rlt::free(device, second);
    rlt::free(device, first);
}
