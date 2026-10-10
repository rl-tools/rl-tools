#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cpu.h>
#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace {
    using namespace rl_tools;
    namespace hd = rl::environments::hyperdrone;
    struct WorldConfig: hd::presets::X500FPV<float, size_t> {
        static constexpr size_t INSTANCES_PER_ENVIRONMENT = 3;
        static constexpr size_t CAM_WIDTH = 2;
        static constexpr size_t CAM_HEIGHT = 2;
        static constexpr size_t HISTORY_LENGTH = 2;
        static constexpr size_t MAX_ENTITY_SLOTS_PER_INSTANCE = 0;
        static constexpr bool SELF_VISIBLE = false;
    };
    using World = hd::World<WorldConfig>;
    using Environment = hd::MultiEnvironment<World, 2>;
    struct Config: hd::spy::Config<Environment> {
        static constexpr size_t ATLAS_WIDTH = 7;
        static constexpr size_t ATLAS_HEIGHT = 5;
    };

    TEST(HYPERDRONE_SPY, CanonicalHistoryAndSceneIdentity) {
        devices::DefaultCPU device;
        Environment environment{};
        hd::spy::State<Config> state;
        std::array<uint32_t, 35> atlas{};
        std::array<std::array<float, 72>, 2> history{};
        state.buffer.atlas = atlas.data();
        state.columns = 3;
        state.rows = 2;
        environment.shared.library.scenes.emplace_back();
        environment.shared.library.scenes.emplace_back();
        for(size_t member = 0; member < 2; member++) {
            auto& world = environment.environments[member];
            world.slots.emplace_back();
            world.slots[0].scene_id = member;
            world.history._data = history[member].data();
            world.history_step = 2;
            world.scene_version = member + 1;
            for(size_t camera = 0; camera < 3; camera++) {
                for(size_t pixel = 0; pixel < 4; pixel++) {
                    float* rgb = history[member].data() + 36 + camera * 12 + pixel * 3;
                    rgb[0] = float(member);
                    rgb[1] = float(camera) * 0.5f;
                    rgb[2] = float(pixel) / 3.0f;
                }
            }
        }
        const auto before = history;
        state.selected_camera = 4;
        prepare_spy_capture(state, environment, 10);
        pack_spy_atlas_cpu(state, environment);
        EXPECT_EQ(state.buffer.job.scene, &environment.shared.library.scenes[1]);
        EXPECT_EQ(state.buffer.job.member, 1u);
        EXPECT_EQ(state.buffer.job.instance, 1u);
        EXPECT_EQ(state.buffer.job.scene_version, 2u);
        EXPECT_EQ(atlas[0], 0xff000000u);
        EXPECT_EQ(atlas[2], 0xff550000u);
        EXPECT_EQ(atlas[3], 0xff008000u);
        EXPECT_EQ(atlas[4 * 7 + 6], 0xffffffffu);
        EXPECT_EQ(history, before);
        EXPECT_EQ(environment.environments[0].history_step, 2u);
        EXPECT_EQ(environment.environments[1].history_step, 2u);
        prepare_spy_capture(state, environment, 20);
        EXPECT_EQ(state.buffer.frames[1].captured_at, 10);
        environment.environments[1].scene_version++;
        environment.environments[1].render_pending = true;
        prepare_spy_capture(state, environment, 30);
        pack_spy_atlas_cpu(state, environment);
        EXPECT_FALSE(state.buffer.selected_valid);
        EXPECT_FALSE(state.buffer.frames[1].valid);
        EXPECT_EQ(atlas[4 * 7 + 6], 0xff202020u);
        environment.environments[1].render_pending = false;
        environment.environments[1].history_step++;
        prepare_spy_capture(state, environment, 40);
        pack_spy_atlas_cpu(state, environment);
        EXPECT_TRUE(state.buffer.selected_valid);
        EXPECT_EQ(state.buffer.job.scene_version, 3u);
        EXPECT_EQ(state.buffer.frames[1].captured_at, 40);
        EXPECT_EQ(atlas[4 * 7 + 6], 0xff000000u);
        EXPECT_EQ(history, before);
    }
}
