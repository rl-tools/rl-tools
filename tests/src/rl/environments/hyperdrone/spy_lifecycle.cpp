#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cpu.h>

#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace rl_tools::rl::environments::hyperdrone::spy::ui {
    struct Window {
        Input input;
        size_t camera_agents;
        size_t cameras_per_agent;
        size_t atlas_size;
        size_t spectator_size;
        std::vector<uint32_t> atlas;
        std::vector<uint32_t> spectator;
        size_t spectator_uploads = 0;
        double atlas_age = -1;
        double spectator_age = -1;
    };
    bool fail_create = false;
    size_t created = 0;
    size_t destroyed = 0;

    Window* create(size_t, size_t, size_t camera_agents, size_t cameras_per_agent, int atlas_width, int atlas_height, int, int, int spectator_width, int spectator_height) {
        if(fail_create) {
            throw std::runtime_error("Simulated window creation failure");
        }
        auto* window = new Window;
        window->camera_agents = camera_agents;
        window->cameras_per_agent = cameras_per_agent;
        window->atlas_size = static_cast<size_t>(atlas_width) * atlas_height;
        window->spectator_size = static_cast<size_t>(spectator_width) * spectator_height;
        created++;
        return window;
    }

    Input update(Window* window, double atlas_age, double spectator_age, const char*) {
        window->atlas_age = atlas_age;
        window->spectator_age = spectator_age;
        return window->input;
    }

    void upload_atlas(Window* window, const uint32_t* pixels) {
        window->atlas.assign(pixels, pixels + window->atlas_size);
    }

    void upload_spectator(Window* window, const uint32_t* pixels) {
        window->spectator.assign(pixels, pixels + window->spectator_size);
        window->spectator_uploads++;
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
    namespace hd = rl::environments::hyperdrone;
    namespace rt = rendering::raytracing;
    namespace ui = hd::spy::ui;

    struct WorldConfig: hd::presets::X500FPV<float, size_t> {
        using SHADING = rt::Low;
        static constexpr size_t INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr size_t N_AGENTS = 2;
        static constexpr size_t N_CAMERA_AGENTS = 1;
        static constexpr size_t CAM_WIDTH = 4;
        static constexpr size_t CAM_HEIGHT = 3;
        static constexpr size_t HISTORY_LENGTH = 1;
        static constexpr size_t MAX_ENTITY_SLOTS_PER_INSTANCE = 16;
        using CAMERA_PROFILE = rendering::raytracing::outputs::List<rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>>;
        using CAMERA_OUTPUTS = hd::CameraOutputs<CAMERA_PROFILE, CAMERA_PROFILE>;
        using OBSERVATION = hd::ImageObservation<0, 1>;
        static constexpr size_t NUM_PROBES = 1;
        static constexpr bool SELF_VISIBLE = true;
        static constexpr bool ENABLE_MOTION_BLUR = false;
    };
    using World = hd::World<WorldConfig>;
    using Environment = hd::MultiEnvironment<World, 1>;
    struct Config: hd::spy::Config<Environment> {
        struct SPECTATOR_CONFIG: hd::spy::SpectatorConfig<World::RENDERER_CONFIG> {
            static constexpr size_t CAM_WIDTH = 16;
            static constexpr size_t CAM_HEIGHT = 12;
        };
        static constexpr size_t ATLAS_WIDTH = 8;
        static constexpr size_t ATLAS_HEIGHT = 4;
        static constexpr double REFRESH_INTERVAL = 0;
        static constexpr double UI_REFRESH_INTERVAL = 0;
    };

    class HyperDroneSpyLifecycle: public ::testing::Test {
    protected:
        devices::DefaultCPU device;
        Environment environment{};
        hd::Spy<Config> viewer;
        std::array<float, World::HISTORY_SPEC::SIZE> history{};
        std::array<rt::Camera<float>, World::INSTANCES * World::N_VIEWS> camera_data{};

        void SetUp() override {
            ui::fail_create = false;
            ui::created = 0;
            ui::destroyed = 0;
            auto& world = environment.environments[0];
            world.history._data = history.data();
            world.slots.resize(2);
            environment.shared.library.scenes.resize(2);
            for(size_t camera = 0; camera < camera_data.size(); camera++) {
                const float position[3] = {0, float(camera / World::N_VIEWS) * 8 + float(camera % World::N_VIEWS) * 2, 0};
                const float look[3] = {1, position[1], 0};
                const float up[3] = {0, 0, 1};
                camera_data[camera] = make_camera_data(position, look, up, 80.0f, 4.0f / 3.0f);
            }
            for(size_t scene_i = 0; scene_i < 2; scene_i++) {
                auto& scene = environment.shared.library.scenes[scene_i];
                rt::Mesh mesh;
                mesh.vertices = {3, -1, -1, 3, 1, -1, 3, 0, 1};
                mesh.indices = {0, 1, 2};
                mesh.color[0] = scene_i == 0 ? 0.8f : 0.1f;
                mesh.color[1] = scene_i == 0 ? 0.1f : 0.8f;
                mesh.color[2] = 0.1f;
                rt::Object object;
                object.meshes.push_back(mesh);
                scene.objects.push_back(object);
                scene.instances.push_back({0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, true});
                auto& slot = world.slots[scene_i];
                slot.scene_id = scene_i;
                slot.metadata.max_ray_length = 20;
                malloc(device, slot.renderer);
                init(device, slot.renderer, scene, environment.shared.library.pool, slot.metadata);
                Tensor<typename decltype(slot.renderer.cameras)::SPEC> camera_alias;
                camera_alias._data = camera_data.data();
                copy(device, slot.renderer.device, camera_alias, cameras(device, slot.renderer));
            }
            publish_scene(0);
        }

        void TearDown() override {
            free(device, viewer);
            for(auto& slot: environment.environments[0].slots) {
                if(slot.renderer.backend != nullptr) {
                    free(device, slot.renderer);
                }
            }
        }

        void publish_scene(size_t scene) {
            auto& world = environment.environments[0];
            world.active_slot = scene;
            world.scene_version++;
            auto& renderer = world.slots[scene].renderer;
            render(device, renderer);
            Tensor<rl_tools::rendering::raytracing::outputs::TensorSpecification<World::RENDERER_SPEC, rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>>> observation_alias;
            observation_alias._data = history.data();
            copy(renderer.device, device, output(device, renderer, rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>{}), observation_alias);
            world.history_step++;
            world.render_pending = false;
        }

        void await_image(size_t instance, size_t version) {
            const auto previous = viewer.state->window->spectator_uploads;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            do {
                if(!update(device, viewer, environment)) {
                    throw std::runtime_error("Spy closed before publishing an image");
                }
                if(std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("Spy did not publish the selected view");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while(viewer.state->window->spectator_uploads == previous || viewer.state->buffer.image.instance != instance || viewer.state->buffer.image.scene_version != version || viewer.state->reset_camera);
        }
    };

    TEST_F(HyperDroneSpyLifecycle, NavigationSelectionSceneChangeAndClosePreserveSource) {
        const auto before = history;
        const auto sequence = environment.environments[0].history_step;
        malloc(device, viewer);
        init(device, viewer, environment);
        auto* window = viewer.state->window;
        EXPECT_EQ(window->camera_agents, World::N_CAMERA_AGENTS);
        EXPECT_EQ(window->cameras_per_agent, World::N_CAMERAS_PER_AGENT);
        await_image(0, 1);
        ASSERT_EQ(window->atlas.size(), Config::ATLAS_WIDTH * Config::ATLAS_HEIGHT);
        ASSERT_EQ(window->spectator.size(), Config::SPECTATOR_CONFIG::CAM_WIDTH * Config::SPECTATOR_CONFIG::CAM_HEIGHT);
        const auto red = window->spectator;
        window->input.right = 1;
        window->input.yaw_delta = 0.25f;
        viewer.state->last_update = spy_time() - 0.05;
        ASSERT_TRUE(update(device, viewer, environment));
        const auto moved = viewer.state->views[0];
        EXPECT_LT(moved.position[1], -0.1f);
        EXPECT_FLOAT_EQ(moved.yaw, 0.25f);
        window->input = {};
        window->input.camera = World::N_VIEWS;
        await_image(1, 1);
        EXPECT_FLOAT_EQ(viewer.state->views[1].position[1], 8);
        EXPECT_NE(window->spectator, red);
        window->input.camera = 0;
        await_image(0, 1);
        EXPECT_FLOAT_EQ(viewer.state->views[0].position[1], moved.position[1]);
        EXPECT_FLOAT_EQ(viewer.state->views[0].yaw, moved.yaw);
        window->input.reset_camera = true;
        await_image(0, 1);
        window->input.reset_camera = false;
        EXPECT_FLOAT_EQ(viewer.state->views[0].position[1], 0);
        EXPECT_FLOAT_EQ(viewer.state->views[0].yaw, 0);
        EXPECT_EQ(history, before);
        EXPECT_EQ(environment.environments[0].history_step, sequence);
        window->input.camera = 1;
        window->input.reset_camera = true;
        await_image(0, 1);
        EXPECT_FLOAT_EQ(viewer.state->views[0].position[1], 2);
        window->input = {};
        std::array<rt::Camera<float>, World::INSTANCES * World::N_VIEWS> after{};
        auto& renderer = environment.environments[0].slots[0].renderer;
        Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
        camera_alias._data = after.data();
        copy(renderer.device, device, cameras(device, renderer), camera_alias);
        EXPECT_EQ(std::memcmp(after.data(), camera_data.data(), sizeof(after)), 0);
        publish_scene(1);
        ASSERT_TRUE(update(device, viewer, environment));
        EXPECT_LT(window->spectator_age, 0);
        await_image(0, 2);
        EXPECT_NE(window->spectator, red);
        EXPECT_EQ(viewer.state->displayed_frames[0].scene_version, 2u);
        window->input.closed = true;
        EXPECT_FALSE(update(device, viewer, environment));
        EXPECT_EQ(viewer.state, nullptr);
        EXPECT_EQ(ui::created, ui::destroyed);
        EXPECT_FALSE(update(device, viewer, environment));
        free(device, viewer);
        malloc(device, viewer);
        init(device, viewer, environment);
        await_image(0, 2);
    }

    TEST_F(HyperDroneSpyLifecycle, FailureCleanupAndApiOwnership) {
        EXPECT_FALSE(update(device, viewer, environment));
        EXPECT_THROW(init(device, viewer, environment), std::logic_error);
        malloc(device, viewer);
        EXPECT_THROW(malloc(device, viewer), std::logic_error);
        ui::fail_create = true;
        EXPECT_THROW(init(device, viewer, environment), std::runtime_error);
        free(device, viewer);
        EXPECT_EQ(viewer.state, nullptr);
        EXPECT_EQ(ui::created, ui::destroyed);
        ui::fail_create = false;
        malloc(device, viewer);
        init(device, viewer, environment);
        EXPECT_THROW(init(device, viewer, environment), std::logic_error);
        Environment other{};
        EXPECT_THROW(update(device, viewer, other), std::logic_error);
        std::exception_ptr error;
        std::thread wrong_thread([&] {
            try {
                update(device, viewer, environment);
            }
            catch(...) {
                error = std::current_exception();
            }
        });
        wrong_thread.join();
        EXPECT_NE(error, nullptr);
        await_image(0, 1);
    }
}
