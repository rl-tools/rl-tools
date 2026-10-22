#include "../../../../../interface/python/hyperdrone/hyperdrone/_native/environment/environment.h"
#include "../../../utils/utils.h"
#include <gtest/gtest.h>
#include <cstring>

namespace shim = HYPERLUX_MODULE_NAME;

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE, USER_TASK_HEADER){
    const std::string root = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    shim::Environment env;
    env.init({root + "/ProcTHOR-Train-1.glb"}, {root + "/x500.glb"}, {env.parameters_json()}, {{"orbiter", root + "/x500.glb"}}, 0);
    const auto layout = shim::observation_layout();
    ASSERT_EQ(layout.shape.size(), 3);
    ASSERT_EQ(layout.blocks.size(), 1);
    EXPECT_EQ(layout.blocks[0].name, "image");
    env.reset(nullptr, !shim::CUDA_WORLD);
    std::vector<float> observations(shim::TOTAL * shim::WORLD::OBSERVATION_DIM);
    std::vector<float> actions(shim::TOTAL * shim::WORLD::ACTION_DIM, 0);
    std::vector<float> rewards(shim::TOTAL);
    env.observe(observations.data(), !shim::CUDA_WORLD);
    env.step(actions.data(), !shim::CUDA_WORLD);
    env.rewards(rewards.data(), !shim::CUDA_WORLD);
    for(float value : observations){
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        EXPECT_NE(bits & 0x7f800000u, 0x7f800000u);
    }
    for(float value : rewards){
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        EXPECT_NE(bits & 0x7f800000u, 0x7f800000u);
    }
}
