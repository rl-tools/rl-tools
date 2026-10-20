#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/spy/operations_cpu_spectator.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
    namespace rt = rl_tools::rendering::raytracing;
    namespace spy = rt::spy;
    using namespace rl_tools;

    struct SourceConfig: rt::config::Default<float, size_t> {
        using SHADING = rt::Low;
        static constexpr size_t NUM_OVERLAYS = 2;
        static constexpr size_t MAX_OVERLAY_INSTANCES = 1;
        static constexpr size_t MAX_OVERLAYS_PER_CAMERA = 1;
    };

    struct Config: spy::SpectatorConfig<SourceConfig> {
        static constexpr size_t NUM_OVERLAYS = 1;
        static constexpr size_t MAX_OVERLAYS_PER_CAMERA = 1;
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

    class RaytracingSpySpectator: public ::testing::Test {
    protected:
        devices::DefaultCPU device;
        spy::Spectator<Config> spectator;
        rt::Scene scene;
        rt::Scene other_scene;
        rt::AssetPool pool;

        void SetUp() override {
            for(auto object: {triangle(0.8f, 0.1f, 0.1f), triangle(0.1f, 0.8f, 0.1f)}) {
                rt::ObjectAssembly assembly;
                rl_tools::add(device, assembly, object, rl_tools::rendering::raytracing::detail::IDENTITY_TRANSFORM);
                pool.assemblies.push_back(assembly);
            }
        }

        void TearDown() override {
            stop_spy_spectator(device, spectator);
        }

        spy::SpectatorJob<rt::Specification<Config>> job(bool active = true, uint32_t object = 0) {
            spy::SpectatorJob<rt::Specification<Config>> result;
            result.scene = &scene;
            result.pool = &pool;
            result.metadata.max_ray_length = 20;
            const float position[3] = {0, 0, 0};
            const float forward[3] = {1, 0, 0};
            const float up[3] = {0, 0, 1};
            result.camera = make_camera_data(position, forward, up, 80.0f, static_cast<float>(Config::CAM_WIDTH) / Config::CAM_HEIGHT);
            result.overlays[0].slots[0] = {active ? 1u : 0u, object, rendering::segmentation::INHERIT};
            const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            for(size_t element = 0; element < 12; element++) {
                result.overlays[0].rows[element] = identity[element];
            }
            result.num_overlays = 1;
            result.overlay_indices[0] = rt::OverlayIndex{0};
            result.view = 179;
            result.scene_version = 23;
            result.captured_at = 24.25;
            return result;
        }

        void submit_job(spy::SpectatorJob<rt::Specification<Config>> request) {
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

        spy::SpectatorImage render_job(spy::SpectatorJob<rt::Specification<Config>> request) {
            submit_job(std::move(request));
            return await_result();
        }
    };

    TEST_F(RaytracingSpySpectator, ResolvedObjectReplacementTransformsAndActiveMasks) {
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
        translated.overlays[0].rows[7] = 10;
        EXPECT_EQ(render_job(translated).pixels, background.pixels);
        EXPECT_EQ(render_job(job(false, 1)).pixels, background.pixels);
        EXPECT_EQ(render_job(job(true, 0)).pixels, red.pixels);
    }

    TEST_F(RaytracingSpySpectator, SceneSwitchPreservesPoolObjectNumbering) {
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
        switched.overlays[0].slots[0].object = 2;
        EXPECT_EQ(render_job(switched).pixels, green.pixels);
        EXPECT_EQ(render_job(job(true, 0)).pixels, red.pixels);
        stop_spy_spectator(device, spectator);
    }

    TEST_F(RaytracingSpySpectator, BoundedSubmissionResultAndRestart) {
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
            EXPECT_EQ(result.view, request.view);
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

    TEST_F(RaytracingSpySpectator, WorkerErrorPropagatesAndCanRestart) {
        start_spy_spectator(device, spectator);
        submit_job({});
        EXPECT_THROW(await_result(), std::invalid_argument);
        auto request = job();
        EXPECT_THROW(submit_spy_spectator(device, spectator, request), std::invalid_argument);
        stop_spy_spectator(device, spectator);
        start_spy_spectator(device, spectator);
        EXPECT_EQ(render_job(job()).pixels.size(), Config::CAM_WIDTH * Config::CAM_HEIGHT);
    }

    struct MultipleOverlayConfig: Config {
        static constexpr size_t NUM_OVERLAYS = 3;
        static constexpr size_t MAX_OVERLAYS_PER_CAMERA = 2;
    };

    TEST_F(RaytracingSpySpectator, MultipleOverlayAttachmentsAndRemoval) {
        using SPEC = rt::Specification<MultipleOverlayConfig>;
        spy::SpectatorBuffer<MultipleOverlayConfig> buffer;
        spy::SpectatorJob<SPEC> request;
        spy::SpectatorImage image;
        const auto original = job();
        request.camera = original.camera;
        request.metadata = original.metadata;
        request.num_overlays = 2;
        request.overlay_indices = {rt::OverlayIndex{2}, rt::OverlayIndex{0}};
        for(size_t slot = 0; slot < 2; slot++) {
            request.overlays[slot].slots[0] = {1, static_cast<unsigned int>(slot), rendering::segmentation::INHERIT};
            std::copy_n(rt::detail::IDENTITY_TRANSFORM, 12, request.overlays[slot].rows);
            request.overlays[slot].rows[7] = slot == 0 ? -1.25f : 1.25f;
        }
        malloc(device, buffer, spy_spectator_device_id(device, rt::backends::Default{}));
        malloc(buffer.device, buffer.renderer);
        init(buffer.device, buffer.renderer, scene, pool, request.metadata);
        render_spy_spectator(buffer.device, buffer, request, image);
        EXPECT_EQ(buffer.renderer.attachments[0], 2u);
        EXPECT_EQ(buffer.renderer.attachments[1], 0u);
        const auto both = image.pixels;
        request.num_overlays = 1;
        render_spy_spectator(buffer.device, buffer, request, image);
        EXPECT_EQ(buffer.renderer.attachments[0], 2u);
        EXPECT_EQ(buffer.renderer.attachments[1], buffer.renderer.INVALID_OVERLAY);
        const auto red = image.pixels;
        EXPECT_NE(both, red);
        request.overlay_indices[0] = rt::OverlayIndex{0};
        request.overlays[0] = request.overlays[1];
        render_spy_spectator(buffer.device, buffer, request, image);
        EXPECT_NE(image.pixels, red);
        EXPECT_NE(image.pixels, both);
        request.num_overlays = 0;
        render_spy_spectator(buffer.device, buffer, request, image);
        EXPECT_NE(image.pixels, red);
        EXPECT_EQ(buffer.renderer.attachments[0], buffer.renderer.INVALID_OVERLAY);
        request.num_overlays = 3;
        EXPECT_THROW(copy_spy_overlays(buffer.device, buffer.renderer, request), std::invalid_argument);
        request.num_overlays = 1;
        request.overlay_indices[0] = rt::OverlayIndex{3};
        EXPECT_THROW(copy_spy_overlays(buffer.device, buffer.renderer, request), std::invalid_argument);
        free(buffer.device, buffer);
    }
}
