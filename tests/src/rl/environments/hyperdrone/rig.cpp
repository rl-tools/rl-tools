#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/rig/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
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

    // body plus two props at (0.1, 0.1) and (-0.1, 0.1), lifted by 0.05
    rlt::rendering::raytracing::ObjectAssembly two_prop_assembly(DEVICE& device){
        rlt::rendering::raytracing::ObjectAssembly assembly;
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        const float lifted[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0.05f};
        rlt::add(device, assembly, quad("body", 0, 0, 0.1f), identity);
        rlt::add(device, assembly, quad("prop_0", 0.1f, 0.1f, 0.05f), lifted);
        rlt::add(device, assembly, quad("prop_1", -0.1f, 0.1f, 0.05f), lifted);
        return assembly;
    }

    void hub_in_assembly(const rlt::rendering::raytracing::ObjectAssembly& assembly, const RIG& rig, TI part_i, float hub[3]){
        const float pivot[3] = {-rig.parts[part_i].offset[3], -rig.parts[part_i].offset[7], -rig.parts[part_i].offset[11]};
        rlt::rendering::transform_point(assembly.parts[part_i].transform, pivot, hub);
    }

    rlt::rendering::ObjectAssembly transformed_quadrotor(DEVICE& device){
        using rlt::add;
        rlt::rendering::ObjectAssembly assembly;
        const float body[12] = {1,0,0,0.03f, 0,1,0,-0.02f, 0,0,1,0.01f};
        add(device, assembly, quad("body", 0, 0, 0.1f), body);
        const float hubs[4][2] = {{0.2f,-0.2f}, {-0.2f,-0.2f}, {-0.2f,0.2f}, {0.2f,0.2f}};
        for(TI rotor = 0; rotor < 4; rotor++){
            const float transform[12] = {0,-2,0,hubs[rotor][0] - 0.05f, 2,0,0,hubs[rotor][1] - 0.03f, 0,0,2,0.04f};
            add(device, assembly, quad(("prop_" + std::to_string(rotor)).c_str(), 0.015f, -0.025f, 0.025f), transform);
        }
        return assembly;
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, AUTHORED_TRANSFORMS_AND_NOMINAL_WHEELBASE){
    using rlt::evaluate;
    using rlt::init;
    DEVICE device;
    const auto assembly = transformed_quadrotor(device);
    RIG physical, fitted;
    ASSERT_TRUE(init(device, physical, assembly));
    EXPECT_EQ(physical.wheelbase, 0);
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    ASSERT_TRUE(init(device, fitted, assembly, rotors, torques, TI(4)));
    EXPECT_NEAR(fitted.wheelbase, std::sqrt(0.08f), 1e-6f);
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float phases[4] = {};
    rlt::rendering::rig::Buffer<RIG::SPEC> buffer;
    float poses[5][12];
    evaluate(device, physical, identity, phases, buffer, &poses[0][0]);
    for(TI part = 0; part < 5; part++){
        for(TI element = 0; element < 12; element++){
            EXPECT_NEAR(poses[part][element], assembly.parts[part].transform[element], 1e-6f);
        }
    }
    evaluate(device, fitted, identity, phases, buffer, &poses[0][0]);
    const float pivot[3] = {0.015f,-0.025f,0};
    for(TI rotor = 0; rotor < 4; rotor++){
        EXPECT_EQ(fitted.parts[rotor + 1].link, rotor + 1);
        float hub[3];
        rlt::rendering::transform_point(poses[rotor + 1], pivot, hub);
        EXPECT_NEAR(hub[0], rotors[rotor][0], 1e-6f);
        EXPECT_NEAR(hub[1], rotors[rotor][1], 1e-6f);
        EXPECT_NEAR(hub[2], 0.02f, 1e-6f);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, PLACEMENT_SCALE_PRESERVES_BODY_POSE_AND_SPIN_PIVOTS){
    using rlt::evaluate;
    using rlt::init;
    using rlt::rotor_scale;
    using rlt::sample;
    DEVICE device;
    const auto assembly = transformed_quadrotor(device);
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    RIG rig;
    ASSERT_TRUE(init(device, rig, assembly, rotors, torques, TI(4)));
    const auto original = rig;
    const float base_open[12] = {1,0,0,1, 0,1,0,2, 0,0,1,3};
    const float base_close[12] = {0,-1,0,2, 1,0,0,3, 0,0,1,4};
    const float phases_open[4] = {};
    const float phases_close[4] = {0.7f,1.1f,-0.4f,2.2f};
    const float pivot[3] = {0.015f,-0.025f,0};
    rlt::rendering::rig::Buffer<RIG::SPEC> buffer;
    for(T factor : {0.5f,1.5f}){
        T current_rotors[4][3];
        for(TI rotor = 0; rotor < 4; rotor++){
            for(TI axis = 0; axis < 3; axis++){
                current_rotors[rotor][axis] = factor * rotors[rotor][axis];
            }
        }
        const float scale = rotor_scale(device, rig, current_rotors, TI(4));
        EXPECT_NEAR(scale, factor, 1e-6f);
        float poses[5][12], samples[4][5][12];
        sample(device, rig, base_open, base_close, phases_open, phases_close, TI(4), buffer, &samples[0][0][0], TI(5 * 12), &poses[0][0], scale);
        for(TI sample_i = 0; sample_i <= 4; sample_i++){
            const float* transforms = sample_i == 4 ? &poses[0][0] : &samples[sample_i][0][0];
            float base[12];
            const float time = sample_i == 4 ? 1.0f : (float(sample_i) + 0.5f) / 4;
            rlt::rendering::screw_interpolate(base_open, base_close, time, base);
            const float body_origin[3] = {factor * 0.015f, factor * -0.01f, factor * 0.005f};
            float expected_body[3];
            rlt::rendering::transform_point(base, body_origin, expected_body);
            for(TI axis = 0; axis < 3; axis++){
                EXPECT_NEAR(transforms[axis * 4 + 3], expected_body[axis], 1e-5f);
                float squared_norm = 0;
                for(TI row = 0; row < 3; row++){
                    squared_norm += transforms[row * 4 + axis] * transforms[row * 4 + axis];
                }
                EXPECT_NEAR(std::sqrt(squared_norm), factor * 0.5f, 1e-6f);
            }
            for(TI rotor = 0; rotor < 4; rotor++){
                const float local_hub[3] = {current_rotors[rotor][0], current_rotors[rotor][1], factor * 0.02f};
                float expected_hub[3], actual_hub[3];
                rlt::rendering::transform_point(base, local_hub, expected_hub);
                rlt::rendering::transform_point(transforms + (rotor + 1) * 12, pivot, actual_hub);
                for(TI axis = 0; axis < 3; axis++){
                    EXPECT_NEAR(actual_hub[axis], expected_hub[axis], 1e-5f);
                }
            }
        }
    }
    float original_poses[5][12], final_poses[5][12];
    evaluate(device, original, base_close, phases_close, buffer, &original_poses[0][0]);
    evaluate(device, rig, base_close, phases_close, buffer, &final_poses[0][0]);
    for(TI part = 0; part < 5; part++){
        for(TI element = 0; element < 12; element++){
            EXPECT_EQ(original_poses[part][element], final_poses[part][element]);
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, MOTOR_LAYOUT_WARNING_AFTER_WHEELBASE_FIT){
    using rlt::init;
    DEVICE device;
    const T square[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const TI permutation[4] = {2,0,3,1};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    for(T deviation : {0.0f, 0.0499f, 0.0501f}){
        auto assembly = transformed_quadrotor(device);
        const T angle = std::atan(T(1)) + 2 * std::asin(deviation);
        const T x = std::sqrt(T(0.08)) * std::cos(angle);
        const T y = std::sqrt(T(0.08)) * std::sin(angle);
        for(TI rotor = 0; rotor < 4; rotor++){
            assembly.parts[rotor + 1].transform[3] += std::copysign(x, square[rotor][0]) - 2 * square[rotor][0];
            assembly.parts[rotor + 1].transform[7] += std::copysign(y, square[rotor][1]) - 2 * square[rotor][1];
        }
        for(T factor : {0.25f, 2.0f}){
            T rotors[4][3];
            for(TI rotor = 0; rotor < 4; rotor++){
                for(TI axis = 0; axis < 3; axis++){
                    rotors[rotor][axis] = factor * square[permutation[rotor]][axis];
                }
            }
            RIG rig;
            testing::internal::CaptureStderr();
            const bool initialized = init(device, rig, assembly, rotors, torques, TI(4));
            const auto warning = testing::internal::GetCapturedStderr();
            ASSERT_TRUE(initialized);
            EXPECT_NEAR(rig.rotor_position_deviation, deviation, 1e-6f);
            EXPECT_EQ(warning.empty(), deviation <= rlt::rl::environments::hyperdrone::rig::ROTOR_POSITION_WARNING_THRESHOLD);
            if(deviation > 0.05f){
                EXPECT_NE(warning.find("% of wheelbase"), std::string::npos);
                EXPECT_NE(warning.find("warning threshold 5%"), std::string::npos);
            }
            for(TI rotor = 0; rotor < 4; rotor++){
                EXPECT_EQ(rig.parts[permutation[rotor] + 1].link, rotor + 1);
            }
            ASSERT_TRUE(init(device, rig, assembly));
            EXPECT_EQ(rig.rotor_position_deviation, 0);
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, REJECTS_INVALID_WHEELBASE_GEOMETRY){
    using rlt::init;
    using rlt::rotor_wheelbase;
    DEVICE device;
    const auto assembly = transformed_quadrotor(device);
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    RIG rig;
    T invalid_rotors[4][3] = {};
    EXPECT_EQ(rotor_wheelbase(device, invalid_rotors, TI(4)), 0);
    EXPECT_FALSE(init(device, rig, assembly, invalid_rotors, torques, TI(4)));
    const unsigned int infinity_bits = 0x7f800000u;
    const unsigned int nan_bits = 0x7fc00000u;
    static_assert(sizeof(T) == sizeof(infinity_bits));
    std::memcpy(&invalid_rotors[0][0], &infinity_bits, sizeof(T));
    EXPECT_EQ(rotor_wheelbase(device, invalid_rotors, TI(4)), 0);
    EXPECT_FALSE(init(device, rig, assembly, invalid_rotors, torques, TI(4)));
    auto invalid = assembly;
    std::memcpy(&invalid.objects[1].meshes[0].vertices[0], &nan_bits, sizeof(float));
    EXPECT_FALSE(rlt::rendering::rig::detail::is_finite(invalid.objects[1].meshes[0].vertices[0]));
    EXPECT_FALSE(init(device, rig, invalid));
    EXPECT_FALSE(init(device, rig, invalid, rotors, torques, TI(4)));
    invalid = assembly;
    std::memcpy(&invalid.parts[1].transform[3], &infinity_bits, sizeof(float));
    EXPECT_FALSE(rlt::rendering::rig::detail::is_finite(invalid.parts[1].transform[3]));
    EXPECT_FALSE(init(device, rig, invalid));
    EXPECT_FALSE(init(device, rig, invalid, rotors, torques, TI(4)));
    invalid = assembly;
    for(TI part = 2; part < 5; part++){
        invalid.parts[part] = invalid.parts[1];
    }
    EXPECT_FALSE(init(device, rig, invalid, rotors, torques, TI(4)));
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
        // Quad-X: hubs sit off both body axes, front-right (+x, -y) spins CCW (+z)
        float hub[3];
        hub_in_assembly(assembly, rig, part_i, hub);
        EXPECT_GT(std::abs(hub[0]), (T)1e-4);
        EXPECT_GT(std::abs(hub[1]), (T)1e-4);
        EXPECT_EQ(link.axis[2], hub[0] * hub[1] < 0 ? (T)1 : (T)-1);
        direction_sum += link.axis[2];
    }
    EXPECT_EQ(direction_sum, (T)0);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, MATCH_X500_ROTORS){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::rendering::raytracing::ObjectAssembly assembly;
    ASSERT_TRUE((rlt::load<rlt::rendering::raytracing::High, true>(device, assembly, DRONE_PATH)));
    constexpr auto dynamics = rlt::rl::environments::l2f::parameters::dynamics::x500::remap<T, TI>;
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly, dynamics.rotor_positions, dynamics.rotor_torque_directions, (TI)4));
    ASSERT_EQ(rig.num_coordinates, 4);
    // the asset's prop_0..3 sit at rotors 0, 2, 3, 1 of the x500 model
    const TI expected_rotor_of_part[5] = {0, 0, 2, 3, 1};
    EXPECT_EQ(rig.parts[0].link, 0);
    for(TI part_i = 1; part_i < 5; part_i++){
        const TI rotor_i = expected_rotor_of_part[part_i];
        EXPECT_EQ(rig.parts[part_i].link, 1 + rotor_i);
        float hub[3];
        hub_in_assembly(assembly, rig, part_i, hub);
        EXPECT_GT(hub[0] * dynamics.rotor_positions[rotor_i][0], 0);
        EXPECT_GT(hub[1] * dynamics.rotor_positions[rotor_i][1], 0);
        EXPECT_EQ(rig.links[1 + rotor_i].axis[2], -dynamics.rotor_torque_directions[rotor_i][2]);
    }
    // the convention-only rig agrees on the spin of every prop
    RIG heuristic;
    ASSERT_TRUE(rlt::init(device, heuristic, assembly));
    for(TI part_i = 1; part_i < 5; part_i++){
        EXPECT_EQ(heuristic.links[heuristic.parts[part_i].link].axis[2], rig.links[rig.parts[part_i].link].axis[2]);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, SPIN_ABOUT_PIVOT){
    DEVICE device;
    const auto assembly = two_prop_assembly(device);
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly));
    ASSERT_EQ(rig.num_coordinates, 2);
    // Quad-X: (+x, +y) is front-left and spins CW, (-x, +y) rear-left spins CCW
    EXPECT_EQ(rig.links[1].axis[2], (T)-1);
    EXPECT_EQ(rig.links[2].axis[2], (T)1);

    const float base[12] = {1,0,0,2, 0,1,0,0, 0,0,1,1};
    const float phases[RIG::SPEC::MAX_COORDINATES] = {(float)M_PI / 2, (float)M_PI / 2};
    rlt::rendering::rig::Buffer<RIG::SPEC> buffer;
    float poses[3][12];
    rlt::evaluate(device, rig, base, phases, buffer, &poses[0][0]);
    for(int element = 0; element < 12; element++){
        EXPECT_FLOAT_EQ(poses[0][element], base[element]);
    }
    // the hub stays fixed: the pivot maps to its rest position under the body pose, the prop
    // turns a quarter turn in its direction
    const float hub[3] = {0.1f, 0.1f, 0};
    const float blade_tip[3] = {0.15f, 0.1f, 0};
    float hub_world[3], tip_world[3];
    rlt::rendering::transform_point(poses[1], hub, hub_world);
    rlt::rendering::transform_point(poses[1], blade_tip, tip_world);
    EXPECT_NEAR(hub_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(hub_world[2], 1.05f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.05f, 1e-5f);
    const float hub_mirror[3] = {-0.1f, 0.1f, 0};
    const float tip_mirror[3] = {-0.05f, 0.1f, 0};
    rlt::rendering::transform_point(poses[2], hub_mirror, hub_world);
    rlt::rendering::transform_point(poses[2], tip_mirror, tip_world);
    EXPECT_NEAR(hub_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.15f, 1e-5f);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, MATCH_ROTORS){
    DEVICE device;
    const auto assembly = two_prop_assembly(device);
    // rotors listed in the opposite order of the props, torques against the Quad-X spins
    const T rotor_positions[2][3] = {{-0.1, 0.1, 0}, {0.1, 0.1, 0}};
    const T rotor_torque_directions[2][3] = {{0, 0, 1}, {0, 0, -1}};
    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, assembly, rotor_positions, rotor_torque_directions, (TI)2));
    ASSERT_EQ(rig.num_coordinates, 2);
    EXPECT_EQ(rig.parts[1].link, 2);
    EXPECT_EQ(rig.parts[2].link, 1);
    EXPECT_EQ(rig.links[1].axis[2], (T)-1);
    EXPECT_EQ(rig.links[2].axis[2], (T)1);
    // coordinate i is rotor i: a quarter turn of rotor 1 moves prop_0, not prop_1
    const float base[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float phases[RIG::SPEC::MAX_COORDINATES] = {0, (float)M_PI / 2};
    rlt::rendering::rig::Buffer<RIG::SPEC> buffer;
    float poses[3][12];
    rlt::evaluate(device, rig, base, phases, buffer, &poses[0][0]);
    const float tip[3] = {-0.05f, 0.1f, 0};
    float tip_world[3];
    rlt::rendering::transform_point(poses[2], tip, tip_world);
    EXPECT_NEAR(tip_world[0], -0.05f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.1f, 1e-5f);
    const float tip_moving[3] = {0.15f, 0.1f, 0};
    rlt::rendering::transform_point(poses[1], tip_moving, tip_world);
    EXPECT_NEAR(tip_world[0], 0.1f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.15f, 1e-5f);

    const T one_rotor[1][3] = {{0.1, 0.1, 0}};
    const T one_torque[1][3] = {{0, 0, 1}};
    EXPECT_FALSE(rlt::init(device, rig, assembly, one_rotor, one_torque, (TI)1));
    const T coincident[2][3] = {{0.1, 0.1, 0}, {0.1, 0.1, 0}};
    EXPECT_FALSE(rlt::init(device, rig, assembly, coincident, rotor_torque_directions, (TI)2));
    const T no_torque[2][3] = {{0, 0, 0}, {0, 0, -1}};
    EXPECT_FALSE(rlt::init(device, rig, assembly, rotor_positions, no_torque, (TI)2));
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

// every rigged assembly of the test corpus follows the body/prop_* convention with four props
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG, INIT_RIGGED_CORPUS){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    const std::string directory = DRONE_PATH.substr(0, DRONE_PATH.rfind('/'));
    for(const char* name : {"arpl.glb", "crazyflie.glb", "crazyflie_brushless.glb", "soft.glb", "x500.glb"}){
        rlt::rendering::raytracing::ObjectAssembly assembly;
        ASSERT_TRUE((rlt::load<rlt::rendering::raytracing::High, true>(device, assembly, directory + "/" + name))) << name;
        RIG rig;
        ASSERT_TRUE(rlt::init(device, rig, assembly)) << name;
        EXPECT_EQ(rig.num_links, 5) << name;
        EXPECT_EQ(rig.num_coordinates, 4) << name;
        EXPECT_EQ(rig.num_parts, 5) << name;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
