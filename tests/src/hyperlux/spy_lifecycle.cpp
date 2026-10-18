#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/hyperlux/spy/operations_cpu.h>
#include <rl_tools/hyperlux/primitives/operations_cpu.h>
#include <metra/metra.h>

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>

namespace rl_tools::hyperlux::spy::ui {
    struct Window {
        Input input;
        std::vector<uint32_t> pixels;
        size_t size = 0;
        size_t uploads = 0;
    };
    bool fail_create = false;
    size_t created = 0;
    size_t destroyed = 0;

    Window* create(const char*, const std::vector<std::string>& labels, int, int, int, int, int width, int height, Frontend) {
        if(fail_create) {
            throw std::runtime_error("Simulated window creation failure");
        }
        EXPECT_TRUE(labels.empty());
        auto* window = new Window;
        window->size = static_cast<size_t>(width) * height;
        created++;
        return window;
    }

    Input update(Window* window, double, double, const char*) {
        return window->input;
    }

    void upload_atlas(Window*, const uint32_t*) {
        ADD_FAILURE() << "Standalone spy must not upload an atlas";
    }

    void upload_spectator(Window* window, const uint32_t* pixels) {
        window->pixels.assign(pixels, pixels + window->size);
        window->uploads++;
    }

    void destroy(Window* window) {
        if(window != nullptr) {
            destroyed++;
            delete window;
        }
    }
}

namespace {
    using namespace rl_tools;
    namespace rt = hyperlux;
    namespace ui = rt::spy::ui;

    struct SourceConfig: rt::config::Default<float, size_t> {
        using SHADING = rt::Low;
        using OUTPUTS = rt::outputs::List<rt::outputs::Depth<>>;
        static constexpr size_t CAM_WIDTH = 4;
        static constexpr size_t CAM_HEIGHT = 3;
        static constexpr size_t NUM_CAMERAS = 2;
        static constexpr size_t NUM_OVERLAYS = 3;
        static constexpr size_t MAX_OVERLAY_INSTANCES = 2;
        static constexpr size_t MAX_OVERLAYS_PER_CAMERA = 2;
    };
    using Source = rl_tools::HyperLux<rt::Specification<SourceConfig>>;
    struct Config: rt::spy::Config<Source> {
        struct SPECTATOR_CONFIG: rt::spy::SpectatorConfig<SourceConfig> {
            static constexpr size_t CAM_WIDTH = 32;
            static constexpr size_t CAM_HEIGHT = 24;
        };
        static constexpr double REFRESH_INTERVAL = 0;
        static constexpr double UI_REFRESH_INTERVAL = 0;
    };

    class HyperLuxSpyLifecycle: public ::testing::Test {
    protected:
        devices::DefaultCPU device;
        Source source;
        rt::Scene scene;
        rt::AssetPool pool;
        hyperlux::SceneMetadata<float> metadata;
        rt::Spy<Config> viewer;
        rt::OverlayPlacement placement;
        std::array<rt::Camera<float>, 2> camera_data;

        void SetUp() override {
            ui::fail_create = false;
            ui::created = 0;
            ui::destroyed = 0;
            metadata.max_ray_length = 20;
            const float size[3] = {1, 1, 1};
            const float color[3] = {1, 0.2f, 0};
            const auto asset = add(device, pool, make_box(device, size, color));
            malloc(device, source);
            init(device, source, scene, pool, metadata);
            const float origin[3] = {0, 0, 0};
            const float forward[3] = {1, 0, 0};
            const float up[3] = {0, 0, 1};
            camera_data[0] = make_camera_data(origin, forward, up, 80.0f, 4.0f / 3.0f);
            camera_data[1] = camera_data[0];
            camera_data[0].pos[2] = 5;
            Tensor<typename decltype(source.cameras)::SPEC> camera_alias;
            camera_alias._data = camera_data.data();
            copy(device, source.device, camera_alias, cameras(device, source));
            rt::Pose pose;
            pose.base[3] = 3;
            placement = spawn(device, source, rt::OverlayIndex{2}, asset, pose);
            pose.base[7] = 10;
            spawn(device, source, rt::OverlayIndex{0}, asset, pose);
            attach(device, source, 1, rt::OverlayIndex{2});
            attach(device, source, 1, rt::OverlayIndex{0});
            update(device, source);
            malloc(device, viewer);
        }

        void TearDown() override {
            free(device, viewer);
            free(device, source);
        }

        void expect_source_cameras_unchanged() {
            std::array<rt::Camera<float>, 2> copied;
            Tensor<typename decltype(source.cameras)::SPEC> alias;
            alias._data = copied.data();
            copy(source.device, device, cameras(device, source), alias);
            EXPECT_EQ(std::memcmp(copied.data(), camera_data.data(), sizeof(camera_data)), 0);
        }

        void await_uploads(size_t count) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            while(viewer.state->window->uploads < count) {
                ASSERT_TRUE(update(device, viewer, source));
                ASSERT_LT(std::chrono::steady_clock::now(), deadline);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    TEST_F(HyperLuxSpyLifecycle, CapturesSelectedCameraPublishedPosesAndAttachmentOrder) {
        init(device, viewer, source, scene, pool, metadata, 1);
        auto& state = *viewer.state;
        capture_spy(device, state);
        const auto& job = state.buffer.job;
        EXPECT_EQ(job.scene, &scene);
        EXPECT_EQ(job.pool, &pool);
        EXPECT_EQ(job.num_overlays, 2u);
        EXPECT_EQ(job.overlay_indices[0].index, 2u);
        EXPECT_EQ(job.overlay_indices[1].index, 0u);
        EXPECT_EQ(job.overlays[0].slots[0].active, 1u);
        EXPECT_FLOAT_EQ(job.overlays[0].rows[3], 3);
        EXPECT_FLOAT_EQ(job.overlays[1].rows[7], 10);
        EXPECT_FLOAT_EQ(state.position[2], 0);
        rt::Pose pose;
        pose.base[3] = 7;
        set_pose(device, source, rt::OverlayIndex{2}, placement, pose);
        capture_spy(device, state);
        EXPECT_FLOAT_EQ(job.overlays[0].rows[3], 3);
        update(device, source);
        capture_spy(device, state);
        EXPECT_FLOAT_EQ(job.overlays[0].rows[3], 7);
        detach(device, source, 1, rt::OverlayIndex{2});
        capture_spy(device, state);
        EXPECT_EQ(job.num_overlays, 1u);
        EXPECT_EQ(job.overlay_indices[0].index, 0u);
        expect_source_cameras_unchanged();
    }

    TEST_F(HyperLuxSpyLifecycle, RendersWithoutSourceRgbNavigatesAndCloses) {
        const auto started = std::chrono::steady_clock::now();
        init(device, viewer, source, scene, pool, metadata, 1);
        await_uploads(1);
        auto* window = viewer.state->window;
        const auto original = window->pixels;
        EXPECT_EQ(original.size(), Config::SPECTATOR_CONFIG::CAM_WIDTH * Config::SPECTATOR_CONFIG::CAM_HEIGHT);
        despawn(device, source, rt::OverlayIndex{2}, placement);
        update(device, source);
        await_uploads(window->uploads + 2);
        EXPECT_NE(window->pixels, original);
        window->input.forward = 1;
        window->input.yaw_delta = 0.1f;
        viewer.state->last_update -= 0.1;
        EXPECT_TRUE(update(device, viewer, source));
        EXPECT_GT(viewer.state->position[0], 0);
        EXPECT_GT(viewer.state->yaw, 0);
        expect_source_cameras_unchanged();
        window->input.closed = true;
        EXPECT_FALSE(update(device, viewer, source));
        EXPECT_EQ(viewer.state, nullptr);
        EXPECT_EQ(ui::created, ui::destroyed);
        EXPECT_FALSE(update(device, viewer, source));
        metra::log("hyperlux/spy/lifecycle/seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    }

    TEST_F(HyperLuxSpyLifecycle, RejectsInvalidCameraWrongThreadAndCleansUpFailure) {
        EXPECT_THROW(init(device, viewer, source, scene, pool, metadata, 2), std::out_of_range);
        ui::fail_create = true;
        EXPECT_THROW(init(device, viewer, source, scene, pool, metadata), std::runtime_error);
        free(device, viewer);
        EXPECT_EQ(viewer.state, nullptr);
        ui::fail_create = false;
        malloc(device, viewer);
        init(device, viewer, source, scene, pool, metadata);
        std::thread other([&] {
            EXPECT_THROW(update(device, viewer, source), std::logic_error);
            EXPECT_THROW(free(device, viewer), std::runtime_error);
        });
        other.join();
        Source different_source;
        EXPECT_THROW(update(device, viewer, different_source), std::logic_error);
        EXPECT_THROW(malloc(device, viewer), std::logic_error);
        EXPECT_THROW(init(device, viewer, source, scene, pool, metadata), std::logic_error);
        free(device, viewer);
        EXPECT_EQ(ui::created, ui::destroyed);
    }

    struct StaticSourceConfig: rt::config::Default<float, size_t> {
        using SHADING = rt::Low;
        using OUTPUTS = rt::outputs::List<rt::outputs::Depth<>>;
        static constexpr size_t CAM_WIDTH = 4;
        static constexpr size_t CAM_HEIGHT = 3;
        static constexpr size_t NUM_CAMERAS = 1;
    };
    using StaticSource = rl_tools::HyperLux<rt::Specification<StaticSourceConfig>>;
    struct StaticConfig: rt::spy::Config<StaticSource> {
        struct SPECTATOR_CONFIG: rt::spy::SpectatorConfig<StaticSourceConfig> {
            static constexpr size_t CAM_WIDTH = 16;
            static constexpr size_t CAM_HEIGHT = 12;
        };
        static constexpr double REFRESH_INTERVAL = 0;
        static constexpr double UI_REFRESH_INTERVAL = 0;
    };

    TEST(HyperLuxSpyNoOverlays, RendersStaticDepthOnlySource) {
        devices::DefaultCPU device;
        StaticSource source;
        rt::Spy<StaticConfig> viewer;
        rt::Scene scene;
        rt::AssetPool pool;
        hyperlux::SceneMetadata<float> metadata;
        metadata.max_ray_length = 20;
        malloc(device, source);
        init(device, source, scene, pool, metadata);
        const float position[3] = {0, 0, 0};
        const float look[3] = {1, 0, 0};
        const float up[3] = {0, 0, 1};
        auto camera = make_camera_data(position, look, up, 80.0f, 4.0f / 3.0f);
        Tensor<typename decltype(source.cameras)::SPEC> alias;
        alias._data = &camera;
        copy(device, source.device, alias, cameras(device, source));
        malloc(device, viewer);
        init(device, viewer, source, scene, pool, metadata);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while(viewer.state->window->uploads == 0 && std::chrono::steady_clock::now() < deadline) {
            EXPECT_TRUE(update(device, viewer, source));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(viewer.state->buffer.job.num_overlays, 0u);
        EXPECT_GT(viewer.state->window->uploads, 0u);
        free(device, viewer);
        free(device, source);
    }
}
