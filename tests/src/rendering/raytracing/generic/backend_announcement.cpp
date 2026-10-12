#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#include <rl_tools/rendering/raytracing/save_cpu.h>

#include <gtest/gtest.h>

#include <string>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;
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
        using OUTPUTS = rro::List<rro::Depth<>, rro::Probes>;
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
    EXPECT_STREQ(rrt::backends::name<rrt::backends::Webgpu>(), "webgpu");
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

TEST(RENDERING_RAYTRACING_VERBOSITY, LEVEL_GATES_MESSAGES){
    using Verbosity = rlt::devices::logging::Verbosity;
    DEVICE device;
    rlt::init(device);
    RENDERER renderer;

    device.logger.verbosity = Verbosity::warning;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    rlt::malloc(device, renderer);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "");

    device.logger.verbosity = Verbosity::info;
    testing::internal::CaptureStderr();
    rlt::generate_probe_directions(device, renderer);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");

    device.logger.verbosity = Verbosity::debug;
    testing::internal::CaptureStderr();
    rlt::generate_probe_directions(device, renderer);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "#rl_tools::rendering::raytracing: Generated 1 probe directions per camera\n");

    rlt::free(device, renderer);
}

TEST(RENDERING_RAYTRACING_VERBOSITY, QUIET_PROBE_EXPORT_PRESERVES_DATA){
    const auto directory = std::filesystem::temp_directory_path() / "rl_tools_verbosity_probe_export";
    std::filesystem::create_directories(directory);
    const std::string quiet = (directory / "quiet.bin").string();
    const std::string verbose = (directory / "verbose.bin").string();
    DEVICE device;
    const rrt::CollisionResult probes[] = {{2.5f, 1}};
    device.logger.verbosity = rlt::devices::logging::Verbosity::warning;
    testing::internal::CaptureStderr();
    rrt::detail::write_probes_bin_and_log<SPEC>(device, probes, quiet.c_str());
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
    device.logger.verbosity = rlt::devices::logging::Verbosity::debug;
    testing::internal::CaptureStderr();
    rrt::detail::write_probes_bin_and_log<SPEC>(device, probes, verbose.c_str());
    EXPECT_NE(testing::internal::GetCapturedStderr().find("COLLISION PROBE RESULTS"), std::string::npos);

    const auto read = [](const std::string& path){
        std::ifstream input(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const auto quiet_data = read(quiet);
    EXPECT_EQ(quiet_data.size(), 2 * sizeof(int) + sizeof(probes));
    EXPECT_EQ(quiet_data, read(verbose));
    std::filesystem::remove(quiet);
    std::filesystem::remove(verbose);
    std::filesystem::remove(directory);
}
