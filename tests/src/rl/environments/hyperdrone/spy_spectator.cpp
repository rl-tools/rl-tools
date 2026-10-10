#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cpu_spectator.h>

#include <gtest/gtest.h>
#include <chrono>
#include <stdexcept>

namespace {
    namespace rt = rl_tools::rendering::raytracing;
    namespace spy = rl_tools::rl::environments::hyperdrone::spy;
    using namespace rl_tools;

    struct SourceConfig: rt::config::Default<float, size_t> {
        using SHADING = rt::Low;
        static constexpr size_t NUM_OVERLAYS = 2;
        static constexpr size_t MAX_OVERLAY_INSTANCES = 1;
        static constexpr size_t MAX_OVERLAYS_PER_CAMERA = 1;
    };

    struct Config: spy::SpectatorConfig<SourceConfig> {
        static constexpr size_t CAM_WIDTH = 32;
        static constexpr size_t CAM_HEIGHT = 24;
    };

    rt::Object triangle(float red, float green, float blue) {
        rt::Mesh mesh;
        mesh.vertices = {3, -1, -1, 3, 1, -1, 3, 0, 1};
        mesh.indices = {0, 1, 2};
        mesh.color[0] = red;
        mesh.color[1] = green;
        mesh.color[2] = blue;
        rt::Object object;
        object.meshes.push_back(mesh);
        return object;
    }

    class HyperDroneSpySpectator: public ::testing::Test {
    protected:
        devices::DefaultCPU device;
        spy::Spectator<Config> spectator;
        rt::Scene scene;
        rt::Scene other_scene;
        rt::AssetPool pool;

        void SetUp() override {
            for(auto object: {triangle(0.8f, 0.1f, 0.1f), triangle(0.1f, 0.8f, 0.1f)}) {
                rt::ObjectAssembly assembly;
                assembly.objects.push_back(object);
                assembly.parts.push_back({0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}});
                pool.assemblies.push_back(assembly);
            }
        }

        void TearDown() override {
            stop_spy_spectator(device, spectator);
        }

        spy::SpectatorJob<float> job(bool active = true, uint32_t object = 0) {
            spy::SpectatorJob<float> result;
            result.scene = &scene;
            result.pool = &pool;
            result.metadata.max_ray_length = 20;
            const float position[3] = {0, 0, 0};
            const float forward[3] = {1, 0, 0};
            const float up[3] = {0, 0, 1};
            result.camera = make_camera_data(position, forward, up, 80.0f, static_cast<float>(Config::CAM_WIDTH) / Config::CAM_HEIGHT);
            result.overlay.push_back({object, active, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}});
            result.member = 17;
            result.instance = 9;
            result.scene_version = 23;
            result.captured_at = 24.25;
            return result;
        }

        void submit_job(spy::SpectatorJob<float> request) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while(!submit_spy_spectator(device, spectator, request)) {
                if(std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("Spectator request was not accepted");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        spy::SpectatorImage await_result() {
            spy::SpectatorImage result;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while(!poll_spy_spectator(device, spectator, result)) {
                if(std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("Spectator rendering timed out");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return result;
        }

        spy::SpectatorImage render_job(spy::SpectatorJob<float> request) {
            submit_job(std::move(request));
            return await_result();
        }
    };

    TEST_F(HyperDroneSpySpectator, ResolvedObjectReplacementTransformsAndActiveMasks) {
        start_spy_spectator(device, spectator);
        const auto background = render_job(job(false));
        const auto red = render_job(job(true, 0));
        const auto green = render_job(job(true, 1));
        ASSERT_EQ(red.pixels.size(), Config::CAM_WIDTH * Config::CAM_HEIGHT);
        ASSERT_EQ(green.pixels.size(), red.pixels.size());
        constexpr size_t center = Config::CAM_HEIGHT / 2 * Config::CAM_WIDTH + Config::CAM_WIDTH / 2;
        EXPECT_NE(red.pixels, background.pixels);
        EXPECT_GT(red.pixels[center] & 0xffu, (red.pixels[center] >> 8) & 0xffu);
        EXPECT_GT((green.pixels[center] >> 8) & 0xffu, green.pixels[center] & 0xffu);
        auto translated = job(true, 1);
        translated.overlay[0].transform[7] = 10;
        EXPECT_EQ(render_job(translated).pixels, background.pixels);
        EXPECT_EQ(render_job(job(false, 1)).pixels, background.pixels);
        EXPECT_EQ(render_job(job(true, 0)).pixels, red.pixels);
    }

    TEST_F(HyperDroneSpySpectator, SceneSwitchPreservesPoolObjectNumbering) {
        other_scene.objects.push_back(triangle(0.1f, 0.1f, 0.8f));
        other_scene.instances.push_back({0, {1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1, 0}, false});
        start_spy_spectator(device, spectator);
        const auto red = render_job(job(true, 0));
        const auto green = render_job(job(true, 1));
        auto switched = job(true, 1);
        switched.scene = &other_scene;
        switched.scene_version++;
        const auto switched_red = render_job(switched);
        EXPECT_EQ(switched_red.scene_version, switched.scene_version);
        EXPECT_EQ(switched_red.pixels, red.pixels);
        switched.overlay[0].object = 2;
        EXPECT_EQ(render_job(switched).pixels, green.pixels);
        EXPECT_EQ(render_job(job(true, 0)).pixels, red.pixels);
        stop_spy_spectator(device, spectator);
    }

    TEST_F(HyperDroneSpySpectator, BoundedSubmissionResultAndRestart) {
        auto request = job();
        spy::SpectatorImage result;
        EXPECT_FALSE(submit_spy_spectator(device, spectator, request));
        EXPECT_FALSE(poll_spy_spectator(device, spectator, result));
        for(size_t repetition = 0; repetition < 2; repetition++) {
            start_spy_spectator(device, spectator);
            EXPECT_THROW(start_spy_spectator(device, spectator), std::logic_error);
            submit_job(job());
            EXPECT_FALSE(submit_spy_spectator(device, spectator, request));
            result = await_result();
            EXPECT_EQ(result.member, request.member);
            EXPECT_EQ(result.instance, request.instance);
            EXPECT_EQ(result.scene_version, request.scene_version);
            EXPECT_DOUBLE_EQ(result.captured_at, request.captured_at);
            EXPECT_EQ(result.pixels.size(), Config::CAM_WIDTH * Config::CAM_HEIGHT);
            EXPECT_FALSE(poll_spy_spectator(device, spectator, result));
            stop_spy_spectator(device, spectator);
            stop_spy_spectator(device, spectator);
            EXPECT_FALSE(submit_spy_spectator(device, spectator, request));
        }
        start_spy_spectator(device, spectator);
        submit_job(job());
        stop_spy_spectator(device, spectator);
        EXPECT_FALSE(poll_spy_spectator(device, spectator, result));
    }

    TEST_F(HyperDroneSpySpectator, WorkerErrorPropagatesAndCanRestart) {
        start_spy_spectator(device, spectator);
        submit_job({});
        EXPECT_THROW(await_result(), std::invalid_argument);
        auto request = job();
        EXPECT_THROW(submit_spy_spectator(device, spectator, request), std::invalid_argument);
        stop_spy_spectator(device, spectator);
        start_spy_spectator(device, spectator);
        EXPECT_EQ(render_job(job()).pixels.size(), Config::CAM_WIDTH * Config::CAM_HEIGHT);
    }
}
