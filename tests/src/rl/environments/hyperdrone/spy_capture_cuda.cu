#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/spy/capture_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/spy/capture_cuda.h>
#include <gtest/gtest.h>
#include <metra/metra.h>
#include <array>
#include <chrono>
#include <thread>

namespace {
    using namespace rl_tools;
    namespace hd = rl::environments::hyperdrone;
    namespace rt = rendering::raytracing;
    struct WorldConfig: hd::presets::X500FPV<float, size_t> {
        using OBSERVATION = hd::NoObservation;
        static constexpr size_t N_CAMERA_AGENTS = 1;
        using SHADING = rt::Low;
        static constexpr size_t INSTANCES_PER_WORLD = 3;
        static constexpr size_t CAM_WIDTH = 7;
        static constexpr size_t CAM_HEIGHT = 5;
        static constexpr size_t HISTORY_LENGTH = 1;
        static constexpr size_t MAX_ENTITY_SLOTS_PER_INSTANCE = 2;
        using CAMERA_PROFILE = rendering::raytracing::outputs::List<rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>>;
        using CAMERA_OUTPUTS = hd::CameraOutputs<CAMERA_PROFILE, CAMERA_PROFILE>;
        static constexpr size_t NUM_PROBES = 1;
        static constexpr bool SELF_VISIBLE = false;
        static constexpr bool ENABLE_MOTION_BLUR = false;
    };
    using World = hd::World<WorldConfig>;
    using Environment = rl::environments::HyperDrone<World, 2>;
    struct Config: hd::spy::Config<Environment> {
        static constexpr size_t ATLAS_WIDTH = 17;
        static constexpr size_t ATLAS_HEIGHT = 11;
    };

    TEST(HYPERDRONE_SPY_CUDA, CaptureOwnsSnapshotBeforeSourceOverwrite) {
        devices::DefaultCPU cpu;
        devices::DefaultCUDA device;
        ASSERT_EQ(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), cudaSuccess);
        Environment environment{};
        hd::spy::State<Config> state;
        state.columns = 4;
        state.rows = 3;
        state.buffer.selected_camera = 9;
        state.buffer.selected_valid = true;
        malloc_spy_capture(device, state, rt::backends::Optix{});
        constexpr size_t HISTORY_SIZE = rt::outputs::TensorSpecification<World::RENDERER_SPEC, Config::RGB_FLOAT>::SIZE;
        std::array<std::array<float, HISTORY_SIZE>, 2> histories;
        std::array<float*, 2> device_histories{};
        rt::Scene scene;
        rt::AssetPool pool;
        rendering::SceneMetadata<float> metadata;
        metadata.max_ray_length = 10;
        for(size_t member = 0; member < 2; member++) {
            auto& world = environment.worlds[member];
            world.slots.emplace_back();
            world.history_step = 1;
            state.buffer.frames[member].valid = true;
            for(size_t i = 0; i < HISTORY_SIZE; i++) {
                histories[member][i] = float((i * 17 + member * 29) % 301) / 255.0f - 0.1f;
            }
            auto& renderer = world.slots[0].renderer;
            malloc(cpu, renderer);
            init(cpu, renderer, scene, pool, metadata);
            device_histories[member] = data(output(cpu, renderer, Config::RGB_FLOAT{}));
        }
        auto& selected = environment.worlds[1].slots[0].renderer;
        rt::Camera<float> source_camera{};
        source_camera.pos[0] = 37;
        source_camera.pos[1] = -4;
        source_camera.dir_du[2] = 0.25f;
        const rt::backends::optix::OverlaySlotStructure source_structure[2] = {{1, 13}, {0, 19}};
        const float source_transforms[24] = {1,0,0,3, 0,1,0,4, 0,0,1,5, 1,0,0,6, 0,1,0,7, 0,0,1,8};
        for(bool cuda_history : {true, false}) {
            for(size_t member = 0; member < 2; member++) {
                ASSERT_EQ(cudaMemcpy(device_histories[member], histories[member].data(), HISTORY_SIZE * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
            }
            pack_spy_atlas_cpu(cpu, state, environment);
            const std::vector<uint32_t> expected(state.buffer.atlas, state.buffer.atlas + Config::ATLAS_WIDTH * Config::ATLAS_HEIGHT);
            ASSERT_EQ(cudaMemcpy(data(cameras(cpu, selected)) + 3, &source_camera, sizeof(source_camera), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(overlay_structure(cpu, selected) + 2, source_structure, sizeof(source_structure), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(data(selected.transforms) + 24, source_transforms, sizeof(source_transforms), cudaMemcpyHostToDevice), cudaSuccess);
            if(cuda_history) {
                capture_spy(device, state, environment, rt::backends::Optix{});
            }
            else {
                capture_spy(cpu, state, environment, rt::backends::Optix{});
            }
            const auto producer = cuda_history ? device.stream : stream(cpu, selected);
            ASSERT_EQ(cudaMemsetAsync(data(cameras(cpu, selected)) + 3, 0, sizeof(source_camera), producer), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(overlay_structure(cpu, selected) + 2, 0, sizeof(source_structure), producer), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(data(selected.transforms) + 24, 0, sizeof(source_transforms), producer), cudaSuccess);
            for(auto* history : device_histories) {
                ASSERT_EQ(cudaMemsetAsync(history, 0, HISTORY_SIZE * sizeof(float), producer), cudaSuccess);
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            bool ready = false;
            while(!(ready = poll_spy_capture(device, state, rt::backends::Optix{})) && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_TRUE(ready);
            EXPECT_EQ(std::vector<uint32_t>(state.buffer.atlas, state.buffer.atlas + expected.size()), expected);
            EXPECT_EQ(std::memcmp(&state.buffer.geometry.camera, &source_camera, sizeof(source_camera)), 0);
            for(size_t slot = 0; slot < 2; slot++) {
                EXPECT_EQ(state.buffer.geometry.overlay.slots[slot].active, source_structure[slot].active);
                EXPECT_EQ(state.buffer.geometry.overlay.slots[slot].object, source_structure[slot].object);
                EXPECT_EQ(std::memcmp(state.buffer.geometry.overlay.rows + slot * 12, source_transforms + slot * 12, 12 * sizeof(float)), 0);
            }
            ASSERT_EQ(cudaStreamSynchronize(producer), cudaSuccess);
            state.buffer.frames[0].valid = false;
        }
        free_spy_capture(device, state, rt::backends::Optix{});
        free_spy_capture(device, state, rt::backends::Optix{});
        for(size_t member = 0; member < 2; member++) {
            free(cpu, environment.worlds[member].slots[0].renderer);
        }
        cudaStreamDestroy(device.stream);
    }

    struct CaptureWorldConfig: WorldConfig {
        static constexpr size_t INSTANCES_PER_WORLD = 64;
        using CAMERA_OUTPUTS = hd::CameraOutputs<CAMERA_PROFILE>;
        static constexpr size_t CAM_WIDTH = 64;
        static constexpr size_t CAM_HEIGHT = 64;
    };
    using CaptureWorld = hd::World<CaptureWorldConfig>;
    using CaptureEnvironment = rl::environments::HyperDrone<CaptureWorld, 4>;
    using CaptureConfig = hd::spy::Config<CaptureEnvironment>;

    TEST(HYPERDRONE_SPY_CUDA, CameraAtlasParityAndTiming) {
        devices::DefaultCPU cpu;
        devices::DefaultCUDA device;
        ASSERT_EQ(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), cudaSuccess);
        CaptureEnvironment environment{};
        hd::spy::State<CaptureConfig> state;
        state.columns = 16;
        state.rows = 16;
        malloc_spy_capture(device, state, rt::backends::Optix{});
        constexpr size_t HISTORY_SIZE = rt::outputs::TensorSpecification<CaptureWorld::RENDERER_SPEC, CaptureConfig::RGB_FLOAT>::SIZE;
        std::array<std::vector<float>, 4> histories;
        std::array<float*, 4> device_histories{};
        for(size_t member = 0; member < 4; member++) {
            histories[member].resize(HISTORY_SIZE);
            for(size_t i = 0; i < HISTORY_SIZE; i++) {
                histories[member][i] = float((i * 17 + member * 29) % 257) / 256.0f;
            }
            auto& world = environment.worlds[member];
            world.slots.emplace_back();
            auto& renderer = world.slots[0].renderer;
            rt::Scene scene;
            rt::AssetPool pool;
            rendering::SceneMetadata<float> metadata;
            metadata.max_ray_length = 10;
            malloc(cpu, renderer);
            init(cpu, renderer, scene, pool, metadata);
            world.history_step = 1;
            state.buffer.frames[member].valid = true;
            device_histories[member] = data(output(cpu, renderer, CaptureConfig::RGB_FLOAT{}));
            ASSERT_EQ(cudaMemcpy(device_histories[member], histories[member].data(), HISTORY_SIZE * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        }
        constexpr size_t REPETITIONS = 20;
        using Clock = std::chrono::steady_clock;
        const auto cpu_started = Clock::now();
        for(size_t repetition = 0; repetition < REPETITIONS; repetition++) {
            pack_spy_atlas_cpu(cpu, state, environment);
        }
        const double cpu_seconds = std::chrono::duration<double>(Clock::now() - cpu_started).count() / REPETITIONS;
        const std::vector<uint32_t> expected(state.buffer.atlas, state.buffer.atlas + CaptureConfig::ATLAS_WIDTH * CaptureConfig::ATLAS_HEIGHT);
        double enqueue_seconds = 0, complete_seconds = 0;
        for(size_t repetition = 0; repetition <= REPETITIONS; repetition++) {
            const auto started = Clock::now();
            capture_spy(device, state, environment, rt::backends::Optix{});
            const double enqueued = std::chrono::duration<double>(Clock::now() - started).count();
            const auto deadline = started + std::chrono::seconds(5);
            bool ready = false;
            while(!(ready = poll_spy_capture(device, state, rt::backends::Optix{})) && Clock::now() < deadline) {
                std::this_thread::yield();
            }
            ASSERT_TRUE(ready);
            if(repetition > 0) {
                enqueue_seconds += enqueued / REPETITIONS;
                complete_seconds += std::chrono::duration<double>(Clock::now() - started).count() / REPETITIONS;
            }
        }
        EXPECT_EQ(std::vector<uint32_t>(state.buffer.atlas, state.buffer.atlas + expected.size()), expected);
        metra::log("hyperdrone/spy/capture/cpu_readback_pack/seconds", cpu_seconds);
        metra::log("hyperdrone/spy/capture/cuda/enqueue_seconds", enqueue_seconds);
        metra::log("hyperdrone/spy/capture/cuda/complete_seconds", complete_seconds);
        std::cout << "256 x 64x64 cameras -> 1024x1024 atlas: CPU pack " << cpu_seconds * 1000 << " ms, CUDA enqueue " << enqueue_seconds * 1000 << " ms, CUDA completion " << complete_seconds * 1000 << " ms\n";
        free_spy_capture(device, state, rt::backends::Optix{});
        for(auto& world : environment.worlds) {
            free(cpu, world.slots[0].renderer);
        }
        cudaStreamDestroy(device.stream);
    }
}
