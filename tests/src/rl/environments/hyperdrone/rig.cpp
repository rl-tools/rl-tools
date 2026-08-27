#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/rig/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string DRONE_PATH = "";
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using RIG = rlt::rl::environments::hyperdrone::rig::Rotorcraft<T, TI, 4>;

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, INIT_X500){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::rendering::raytracing::ObjectAssembly assembly;
    ASSERT_TRUE((rlt::load<rlt::rendering::raytracing::High, true>(device, assembly, DRONE_PATH)));
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly));
    EXPECT_EQ(rig.body_part, 0);
    EXPECT_EQ(rig.num_props, 4);
    T direction_sum = 0;
    for(TI prop_i = 0; prop_i < rig.num_props; prop_i++){
        EXPECT_GT(rig.prop_parts[prop_i], 0);
        EXPECT_LT(rig.prop_parts[prop_i], assembly.parts.size());
        EXPECT_TRUE(rig.prop_directions[prop_i] == (T)1 || rig.prop_directions[prop_i] == (T)-1);
        // X configuration: hubs sit off both axes
        EXPECT_GT(std::abs(rig.prop_pivots[prop_i][0]), (T)1e-4);
        EXPECT_GT(std::abs(rig.prop_pivots[prop_i][1]), (T)1e-4);
        EXPECT_EQ(rig.prop_directions[prop_i], rig.prop_pivots[prop_i][0] * rig.prop_pivots[prop_i][1] > 0 ? (T)1 : (T)-1);
        direction_sum += rig.prop_directions[prop_i];
    }
    EXPECT_EQ(direction_sum, (T)0);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, REJECTS_NON_RIG){
    DEVICE device;
    RIG rig;
    rlt::rendering::raytracing::ObjectAssembly empty;
    EXPECT_FALSE(rlt::init(device, rig, empty));

    rlt::rendering::raytracing::ObjectAssembly wrong_root;
    wrong_root.objects.resize(1);
    wrong_root.objects[0].name = "prop_0";
    wrong_root.parts.resize(1);
    wrong_root.parts[0].object = 0;
    EXPECT_FALSE(rlt::init(device, rig, wrong_root));
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, BODY_TRANSFORM){
    DEVICE device;
    const T orientation[4] = {1, 0, 0, 0};
    const T position[3] = {1, 2, 3};
    const T translation[3] = {10, 20, 30};
    float transform[12];
    rlt::rl::environments::hyperdrone::rig::make_body_transform(device, orientation, position, translation, (T)1, (T)0, transform);
    const float expected[12] = {1,0,0,11, 0,1,0,22, 0,0,1,33};
    for(TI element_i = 0; element_i < 12; element_i++){
        EXPECT_FLOAT_EQ(transform[element_i], expected[element_i]);
    }
    // yaw by 90 degrees: x -> y
    rlt::rl::environments::hyperdrone::rig::make_body_transform(device, orientation, position, translation, (T)0, (T)1, transform);
    EXPECT_FLOAT_EQ(transform[0 * 4 + 3], 10 - 2);
    EXPECT_FLOAT_EQ(transform[1 * 4 + 3], 20 + 1);
    EXPECT_FLOAT_EQ(transform[2 * 4 + 3], 30 + 3);
    EXPECT_FLOAT_EQ(transform[0 * 4 + 0], 0);
    EXPECT_FLOAT_EQ(transform[1 * 4 + 0], 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
