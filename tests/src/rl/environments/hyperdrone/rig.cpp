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
using RIG = rlt::rl::environments::hyperdrone::rig::Rotorcraft<TI, 4>;

namespace {
    rlt::rendering::raytracing::Object quad(const char* name, float center_x, float center_y, float half_size){
        rlt::rendering::raytracing::Mesh mesh;
        mesh.vertices = {
            center_x - half_size, center_y - half_size, 0,
            center_x + half_size, center_y - half_size, 0,
            center_x + half_size, center_y + half_size, 0,
            center_x - half_size, center_y + half_size, 0,
        };
        mesh.indices = {0, 1, 2, 0, 2, 3};
        rlt::rendering::raytracing::Object object;
        object.name = name;
        object.meshes.push_back(mesh);
        return object;
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, INIT_X500){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::rendering::raytracing::ObjectAssembly assembly;
    ASSERT_TRUE((rlt::load<rlt::rendering::raytracing::High, true>(device, assembly, DRONE_PATH)));
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly));
    EXPECT_EQ(rig.num_links, 5);
    EXPECT_EQ(rig.num_coordinates, 4);
    EXPECT_EQ(rig.num_parts, assembly.parts.size());
    EXPECT_EQ(rig.parts[0].link, 0);
    T direction_sum = 0;
    for(TI link_i = 1; link_i < rig.num_links; link_i++){
        const auto& link = rig.links[link_i];
        EXPECT_EQ(link.parent, 0);
        EXPECT_TRUE(link.joint == rlt::rendering::rig::Joint::REVOLUTE);
        EXPECT_TRUE(link.axis[2] == (T)1 || link.axis[2] == (T)-1);
        TI part_i = 0;
        while(part_i < rig.num_parts && rig.parts[part_i].link != link_i){
            part_i++;
        }
        ASSERT_LT(part_i, rig.num_parts);
        // X configuration: hubs sit off both axes, the spin direction follows the pivot quadrant
        const T pivot_x = -rig.parts[part_i].offset[3], pivot_y = -rig.parts[part_i].offset[7];
        EXPECT_GT(std::abs(pivot_x), (T)1e-4);
        EXPECT_GT(std::abs(pivot_y), (T)1e-4);
        EXPECT_EQ(link.axis[2], pivot_x * pivot_y > 0 ? (T)1 : (T)-1);
        direction_sum += link.axis[2];
    }
    EXPECT_EQ(direction_sum, (T)0);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, SPIN_ABOUT_PIVOT){
    DEVICE device;
    rlt::rendering::raytracing::ObjectAssembly assembly;
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float lifted[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0.05f};
    rlt::add(device, assembly, quad("body", 0, 0, 0.1f), identity);
    rlt::add(device, assembly, quad("prop_0", 0.1f, 0.1f, 0.05f), lifted);
    rlt::add(device, assembly, quad("prop_1", -0.1f, 0.1f, 0.05f), lifted);
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly));
    ASSERT_EQ(rig.num_coordinates, 2);
    EXPECT_EQ(rig.links[1].axis[2], (T)1);
    EXPECT_EQ(rig.links[2].axis[2], (T)-1);

    const float base[12] = {1,0,0,2, 0,1,0,0, 0,0,1,1};
    const float phases[RIG::SPEC::MAX_COORDINATES] = {(float)M_PI / 2, (float)M_PI / 2};
    rlt::rendering::rig::Buffer<RIG::SPEC> buffer;
    float poses[3][12];
    rlt::evaluate(device, rig, base, phases, buffer, &poses[0][0]);
    for(int element = 0; element < 12; element++){
        EXPECT_FLOAT_EQ(poses[0][element], base[element]);
    }
    // the hub stays fixed: the pivot maps to its rest position under the body pose, the prop
    // turns a quarter turn in its quadrant's direction
    const float hub[3] = {0.1f, 0.1f, 0};
    const float blade_tip[3] = {0.15f, 0.1f, 0};
    float hub_world[3], tip_world[3];
    rlt::rendering::transform_point(poses[1], hub, hub_world);
    rlt::rendering::transform_point(poses[1], blade_tip, tip_world);
    EXPECT_NEAR(hub_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(hub_world[2], 1.05f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.15f, 1e-5f);
    const float hub_mirror[3] = {-0.1f, 0.1f, 0};
    const float tip_mirror[3] = {-0.05f, 0.1f, 0};
    rlt::rendering::transform_point(poses[2], hub_mirror, hub_world);
    rlt::rendering::transform_point(poses[2], tip_mirror, tip_world);
    EXPECT_NEAR(hub_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.05f, 1e-5f);
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

    rlt::rendering::raytracing::ObjectAssembly too_many;
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    rlt::add(device, too_many, quad("body", 0, 0, 0.1f), identity);
    for(int prop_i = 0; prop_i < 5; prop_i++){
        rlt::add(device, too_many, quad("prop_", 0.1f, 0.1f, 0.05f), identity);
    }
    EXPECT_FALSE(rlt::init(device, rig, too_many));
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
