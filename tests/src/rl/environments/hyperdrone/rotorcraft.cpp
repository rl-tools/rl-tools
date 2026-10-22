#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/hyperlux/operations_cpu_mux.h>
#include <rl_tools/hyperlux/datasets/glb/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/rotorcraft/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string DRONE_PATH = "";
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using ROTORCRAFT = rlt::rl::environments::hyperdrone::Rotorcraft;
using ASSEMBLY = rlt::hyperlux::ObjectAssembly;

namespace {
    rlt::hyperlux::Object quad(const char* name, float center_x, float center_y, float half_size){
        rlt::hyperlux::Mesh mesh;
        mesh.vertices = {
            center_x - half_size, center_y - half_size, 0,
            center_x + half_size, center_y - half_size, 0,
            center_x + half_size, center_y + half_size, 0,
            center_x - half_size, center_y + half_size, 0,
        };
        mesh.indices = {0, 1, 2, 0, 2, 3};
        rlt::hyperlux::Object object;
        object.name = name;
        object.meshes.push_back(mesh);
        return object;
    }

    // body plus two props at (0.1, 0.1) and (-0.1, 0.1), lifted by 0.05
    rlt::hyperlux::ObjectAssembly two_prop_assembly(DEVICE& device){
        rlt::hyperlux::ObjectAssembly assembly;
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        const float lifted[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0.05f};
        rlt::add(device, assembly, quad("body", 0, 0, 0.1f), identity);
        rlt::add(device, assembly, quad("prop_0", 0.1f, 0.1f, 0.05f), lifted);
        rlt::add(device, assembly, quad("prop_1", -0.1f, 0.1f, 0.05f), lifted);
        return assembly;
    }

    void hub_in_assembly(const ASSEMBLY& assembly, TI part_i, float hub[3]){
        rlt::hyperlux::transform_point(assembly.parts[part_i].transform, assembly.parts[part_i].joint.pivot, hub);
    }

    struct SinkSpec {
        static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MOTION_BLUR_SAMPLES = 4;
        static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
    };
    constexpr TI SINK_SLOTS = SinkSpec::NUM_OVERLAYS * SinkSpec::MAX_OVERLAY_INSTANCES;

    // the rows set_pose writes for an assembly: the close state and MOTION_BLUR_SAMPLES samples
    struct Rows {
        std::vector<rlt::hyperlux::AssetPart> parts;
        std::vector<float> close = std::vector<float>(SINK_SLOTS * 12), samples = std::vector<float>(SinkSpec::MOTION_BLUR_SAMPLES * SINK_SLOTS * 12);
        explicit Rows(const ASSEMBLY& assembly){
            rlt::hyperlux::detail::flatten_parts(assembly, 0, parts);
        }
        rlt::hyperlux::PoseSink<SinkSpec> sink(){
            rlt::hyperlux::PoseSink<SinkSpec> result;
            result.transforms = close.data();
            result.transforms_motion = samples.data();
            result.parts = parts.data();
            return result;
        }
        rlt::hyperlux::OverlayPlacement placement() const {
            return {0, parts.size(), 0};
        }
        const float* part(TI part_i) const {
            return close.data() + part_i * 12;
        }
        const float* sample(TI sample_i, TI part_i) const {
            return samples.data() + (sample_i * SINK_SLOTS + part_i) * 12;
        }
    };
    Rows pose(DEVICE& device, const ASSEMBLY& assembly, const float base[12], float scale, const float* coordinates){
        Rows rows(assembly);
        rlt::set_pose(device, rows.sink(), rlt::hyperlux::OverlayIndex{0}, rows.placement(), rlt::make_pose(base, scale), coordinates);
        return rows;
    }

    rlt::hyperlux::ObjectAssembly transformed_quadrotor(DEVICE& device){
        using rlt::add;
        rlt::hyperlux::ObjectAssembly assembly;
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, AUTHORED_TRANSFORMS_AND_WHEELBASE_FIT){
    using rlt::init;
    DEVICE device;
    auto physical_assembly = transformed_quadrotor(device);
    ASSERT_TRUE(rlt::articulate(device, physical_assembly));
    auto fitted_assembly = physical_assembly;
    ROTORCRAFT physical, fitted;
    ASSERT_TRUE(init(device, physical, physical_assembly));
    EXPECT_NEAR(physical.wheelbase, std::sqrt(0.32f), 1e-6f);
    EXPECT_EQ(physical.scale, 1);
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    ASSERT_TRUE(init(device, fitted, fitted_assembly, rotors, torques, TI(4)));
    EXPECT_NEAR(fitted.wheelbase, std::sqrt(0.32f), 1e-6f);
    EXPECT_NEAR(fitted.scale, 0.5f, 1e-6f);
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float phases[4] = {};
    const auto neutral = pose(device, physical_assembly, identity, 1, phases);
    for(TI part = 0; part < 5; part++){
        for(TI element = 0; element < 12; element++){
            EXPECT_NEAR(neutral.part(part)[element], physical_assembly.parts[part].transform[element], 1e-6f);
        }
    }
    const auto rows = pose(device, fitted_assembly, identity, fitted.scale, phases);
    const float pivot[3] = {0.015f,-0.025f,0};
    for(TI rotor = 0; rotor < 4; rotor++){
        EXPECT_EQ(fitted_assembly.parts[rotor + 1].joint.coordinate, rotor);
        EXPECT_EQ(fitted_assembly.parts[rotor + 1].joint.axis[2], 1);
        EXPECT_EQ(physical.prop_rotor[rotor], rotor);
        EXPECT_EQ(fitted.prop_rotor[rotor], rotor);
        EXPECT_EQ(fitted.prop_spin[rotor], torques[rotor][2] > 0 ? -1.0f : 1.0f);
        EXPECT_EQ(physical.prop_spin[rotor], fitted.prop_spin[rotor]);
        float hub[3];
        rlt::hyperlux::transform_point(rows.part(rotor + 1), pivot, hub);
        EXPECT_NEAR(hub[0], rotors[rotor][0], 1e-6f);
        EXPECT_NEAR(hub[1], rotors[rotor][1], 1e-6f);
        EXPECT_NEAR(hub[2], 0.02f, 1e-6f);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, ROTOR_SCALE_PRESERVES_BODY_POSE_AND_SPIN_PIVOTS){
    using rlt::init;
    using rlt::rotor_scale;
    DEVICE device;
    auto assembly = transformed_quadrotor(device);
    ASSERT_TRUE(rlt::articulate(device, assembly));
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    ROTORCRAFT rotorcraft;
    ASSERT_TRUE(init(device, rotorcraft, assembly, rotors, torques, TI(4)));
    ASSERT_EQ(rlt::num_coordinates(device, assembly), 4);
    const float base_open[12] = {1,0,0,1, 0,1,0,2, 0,0,1,3};
    const float base_close[12] = {0,-1,0,2, 1,0,0,3, 0,0,1,4};
    const float phases_open[4] = {};
    const float phases_close[4] = {0.7f,1.1f,-0.4f,2.2f};
    const float pivot[3] = {0.015f,-0.025f,0};
    for(T factor : {0.5f,1.5f}){
        T current_rotors[4][3];
        for(TI rotor = 0; rotor < 4; rotor++){
            for(TI axis = 0; axis < 3; axis++){
                current_rotors[rotor][axis] = factor * rotors[rotor][axis];
            }
        }
        const float scale = rotor_scale(device, rotorcraft, current_rotors, TI(4));
        EXPECT_NEAR(scale, factor * 0.5f, 1e-6f);
        Rows rows(assembly);
        rlt::set_pose(device, rows.sink(), rlt::hyperlux::OverlayIndex{0}, rows.placement(), rlt::make_pose(base_open, scale), rlt::make_pose(base_close, scale), phases_open, phases_close);
        for(TI sample_i = 0; sample_i <= 4; sample_i++){
            const float* transforms = sample_i == 4 ? rows.part(0) : rows.sample(sample_i, 0);
            float base[12];
            const float time = sample_i == 4 ? 1.0f : (float(sample_i) + 0.5f) / 4;
            rlt::hyperlux::screw_interpolate(base_open, base_close, time, base);
            const float body_origin[3] = {factor * 0.015f, factor * -0.01f, factor * 0.005f};
            float expected_body[3];
            rlt::hyperlux::transform_point(base, body_origin, expected_body);
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
                rlt::hyperlux::transform_point(base, local_hub, expected_hub);
                rlt::hyperlux::transform_point(transforms + (rotor + 1) * 12, pivot, actual_hub);
                for(TI axis = 0; axis < 3; axis++){
                    EXPECT_NEAR(actual_hub[axis], expected_hub[axis], 1e-5f);
                }
            }
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, MOTOR_LAYOUT_WARNING_AFTER_WHEELBASE_FIT){
    using rlt::init;
    DEVICE device;
    const T square[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const TI permutation[4] = {2,0,3,1};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    constexpr T threshold = ROTORCRAFT::ROTOR_POSITION_WARNING_THRESHOLD;
    std::ostringstream threshold_message;
    threshold_message << "warning threshold " << 100 * threshold << "%";
    for(T deviation : {0.0f, threshold - 0.0001f, threshold + 0.0001f}){
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
            auto fitted = assembly;
            ASSERT_TRUE(rlt::articulate(device, fitted));
            ROTORCRAFT rotorcraft;
            testing::internal::CaptureStderr();
            const bool initialized = init(device, rotorcraft, fitted, rotors, torques, TI(4));
            const auto warning = testing::internal::GetCapturedStderr();
            ASSERT_TRUE(initialized);
            EXPECT_NEAR(rotorcraft.rotor_position_deviation, deviation, 1e-6f);
            EXPECT_EQ(warning.empty(), deviation <= threshold);
            if(deviation > threshold){
                EXPECT_NE(warning.find("% of wheelbase"), std::string::npos);
                EXPECT_NE(warning.find(threshold_message.str()), std::string::npos);
            }
            for(TI rotor = 0; rotor < 4; rotor++){
                EXPECT_EQ(fitted.parts[permutation[rotor] + 1].joint.coordinate, permutation[rotor]);
                EXPECT_EQ(rotorcraft.prop_rotor[permutation[rotor]], rotor);
            }
            ASSERT_TRUE(init(device, rotorcraft, fitted));
            EXPECT_EQ(rotorcraft.rotor_position_deviation, 0);
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, REJECTS_INVALID_WHEELBASE_GEOMETRY){
    using rlt::init;
    using rlt::rotor_wheelbase;
    DEVICE device;
    const auto assembly = transformed_quadrotor(device);
    const T rotors[4][3] = {{0.1f,-0.1f,0}, {-0.1f,-0.1f,0}, {-0.1f,0.1f,0}, {0.1f,0.1f,0}};
    const T torques[4][3] = {{0,0,-1}, {0,0,1}, {0,0,-1}, {0,0,1}};
    ROTORCRAFT rotorcraft;
    const auto rejects = [&](ASSEMBLY candidate, const T (*candidate_rotors)[3]){
        const auto original = candidate;
        if(!rlt::articulate(device, candidate)){
            for(TI part = 0; part < candidate.parts.size(); part++){
                EXPECT_TRUE(candidate.parts[part].joint.type == original.parts[part].joint.type) << "a rejected assembly stays unchanged";
            }
            return true;
        }
        return candidate_rotors == nullptr ? !init(device, rotorcraft, candidate) : !init(device, rotorcraft, candidate, candidate_rotors, torques, TI(4));
    };
    T invalid_rotors[4][3] = {};
    EXPECT_EQ(rotor_wheelbase(device, invalid_rotors, TI(4)), 0);
    EXPECT_TRUE(rejects(assembly, invalid_rotors));
    const unsigned int infinity_bits = 0x7f800000u;
    const unsigned int nan_bits = 0x7fc00000u;
    static_assert(sizeof(T) == sizeof(infinity_bits));
    std::memcpy(&invalid_rotors[0][0], &infinity_bits, sizeof(T));
    EXPECT_EQ(rotor_wheelbase(device, invalid_rotors, TI(4)), 0);
    EXPECT_TRUE(rejects(assembly, invalid_rotors));
    auto invalid = assembly;
    std::memcpy(&invalid.objects[1].meshes[0].vertices[0], &nan_bits, sizeof(float));
    EXPECT_FALSE(rlt::hyperlux::is_finite(invalid.objects[1].meshes[0].vertices[0]));
    EXPECT_TRUE(rejects(invalid, nullptr));
    EXPECT_TRUE(rejects(invalid, rotors));
    invalid = assembly;
    std::memcpy(&invalid.parts[1].transform[3], &infinity_bits, sizeof(float));
    EXPECT_FALSE(rlt::hyperlux::is_finite(invalid.parts[1].transform[3]));
    EXPECT_TRUE(rejects(invalid, nullptr));
    EXPECT_TRUE(rejects(invalid, rotors));
    invalid = assembly;
    for(TI part = 2; part < 5; part++){
        invalid.parts[part] = invalid.parts[1];
    }
    EXPECT_TRUE(rejects(invalid, rotors));
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, INIT_X500){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    ASSEMBLY assembly;
    ASSERT_TRUE((rlt::load<rlt::hyperlux::High, true>(device, assembly, DRONE_PATH)));
    ASSERT_TRUE(rlt::articulate(device, assembly));
    ROTORCRAFT rotorcraft;
    ASSERT_TRUE(rlt::init(device, rotorcraft, assembly));
    EXPECT_EQ(rlt::num_coordinates(device, assembly), 4);
    EXPECT_TRUE(assembly.parts[0].joint.type == rlt::hyperlux::Joint::Type::FIXED);
    EXPECT_EQ(rotorcraft.num_props, 4);
    T direction_sum = 0;
    for(TI part_i = 1; part_i < assembly.parts.size(); part_i++){
        const auto& joint = assembly.parts[part_i].joint;
        const TI prop = part_i - 1;
        EXPECT_TRUE(joint.type == rlt::hyperlux::Joint::Type::REVOLUTE);
        EXPECT_EQ(joint.coordinate, prop);
        EXPECT_EQ(joint.axis[2], (T)1);
        EXPECT_EQ(rotorcraft.prop_rotor[prop], prop);
        // Quad-X: hubs sit off both body axes, front-right (+x, -y) spins CCW (+z)
        float hub[3];
        hub_in_assembly(assembly, part_i, hub);
        EXPECT_GT(std::abs(hub[0]), (T)1e-4);
        EXPECT_GT(std::abs(hub[1]), (T)1e-4);
        EXPECT_EQ(rotorcraft.prop_spin[prop], hub[0] * hub[1] < 0 ? (T)1 : (T)-1);
        direction_sum += rotorcraft.prop_spin[prop];
    }
    EXPECT_EQ(direction_sum, (T)0);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, MATCH_X500_ROTORS){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    ASSEMBLY assembly;
    ASSERT_TRUE((rlt::load<rlt::hyperlux::High, true>(device, assembly, DRONE_PATH)));
    ASSERT_TRUE(rlt::articulate(device, assembly));
    auto heuristic = assembly;
    constexpr auto dynamics = rlt::rl::environments::l2f::parameters::dynamics::x500::real<T, TI>;
    ROTORCRAFT rotorcraft;
    ASSERT_TRUE(rlt::init(device, rotorcraft, assembly, dynamics.rotor_positions, dynamics.rotor_torque_directions, (TI)4));
    ASSERT_EQ(rlt::num_coordinates(device, assembly), 4);
    // the asset's prop_0..3 sit at rotors 0, 2, 3, 1 of the x500 model
    const TI expected_rotor_of_part[5] = {0, 0, 2, 3, 1};
    EXPECT_TRUE(assembly.parts[0].joint.type == rlt::hyperlux::Joint::Type::FIXED);
    for(TI part_i = 1; part_i < 5; part_i++){
        const TI rotor_i = expected_rotor_of_part[part_i];
        const TI prop = part_i - 1;
        const auto& joint = assembly.parts[part_i].joint;
        EXPECT_EQ(joint.coordinate, prop);
        EXPECT_EQ(joint.axis[2], (T)1);
        EXPECT_EQ(rotorcraft.prop_rotor[prop], rotor_i);
        float hub[3];
        hub_in_assembly(assembly, part_i, hub);
        EXPECT_GT(hub[0] * dynamics.rotor_positions[rotor_i][0], 0);
        EXPECT_GT(hub[1] * dynamics.rotor_positions[rotor_i][1], 0);
        EXPECT_EQ(rotorcraft.prop_spin[prop], -dynamics.rotor_torque_directions[rotor_i][2]);
    }
    // the joints are the asset's own and the convention-only rotorcraft agrees on every spin
    ROTORCRAFT quad_x;
    ASSERT_TRUE(rlt::init(device, quad_x, heuristic));
    for(TI part_i = 1; part_i < 5; part_i++){
        EXPECT_EQ(heuristic.parts[part_i].joint.coordinate, assembly.parts[part_i].joint.coordinate);
        EXPECT_EQ(quad_x.prop_spin[part_i - 1], rotorcraft.prop_spin[part_i - 1]);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, SPIN_ABOUT_PIVOT){
    DEVICE device;
    auto assembly = two_prop_assembly(device);
    ASSERT_TRUE(rlt::articulate(device, assembly));
    ROTORCRAFT rotorcraft;
    ASSERT_TRUE(rlt::init(device, rotorcraft, assembly));
    ASSERT_EQ(rlt::num_coordinates(device, assembly), 2);
    // Quad-X: (+x, +y) is front-left and spins CW, (-x, +y) rear-left spins CCW
    EXPECT_EQ(rotorcraft.prop_spin[0], (T)-1);
    EXPECT_EQ(rotorcraft.prop_spin[1], (T)1);

    const float base[12] = {1,0,0,2, 0,1,0,0, 0,0,1,1};
    const float phases[2] = {(float)M_PI / 2, (float)M_PI / 2};
    float joint_coordinates[2];
    rlt::coordinates(device, rotorcraft, phases, joint_coordinates);
    const auto rows = pose(device, assembly, base, 1, joint_coordinates);
    for(int element = 0; element < 12; element++){
        EXPECT_FLOAT_EQ(rows.part(0)[element], base[element]);
    }
    // the hub stays fixed: the pivot maps to its rest position under the body pose, the prop
    // turns a quarter turn in its direction
    const float hub[3] = {0.1f, 0.1f, 0};
    const float blade_tip[3] = {0.15f, 0.1f, 0};
    float hub_world[3], tip_world[3];
    rlt::hyperlux::transform_point(rows.part(1), hub, hub_world);
    rlt::hyperlux::transform_point(rows.part(1), blade_tip, tip_world);
    EXPECT_NEAR(hub_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(hub_world[2], 1.05f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 2.1f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.05f, 1e-5f);
    const float hub_mirror[3] = {-0.1f, 0.1f, 0};
    const float tip_mirror[3] = {-0.05f, 0.1f, 0};
    rlt::hyperlux::transform_point(rows.part(2), hub_mirror, hub_world);
    rlt::hyperlux::transform_point(rows.part(2), tip_mirror, tip_world);
    EXPECT_NEAR(hub_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(hub_world[1], 0.1f, 1e-5f);
    EXPECT_NEAR(tip_world[0], 1.9f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.15f, 1e-5f);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, MATCH_ROTORS){
    DEVICE device;
    auto two_props = two_prop_assembly(device);
    ASSERT_TRUE(rlt::articulate(device, two_props));
    auto assembly = two_props;
    // rotors listed in the opposite order of the props, torques against the Quad-X spins
    const T rotor_positions[2][3] = {{-0.1, 0.1, 0}, {0.1, 0.1, 0}};
    const T rotor_torque_directions[2][3] = {{0, 0, 1}, {0, 0, -1}};
    ROTORCRAFT rotorcraft;
    ASSERT_TRUE(rlt::init(device, rotorcraft, assembly, rotor_positions, rotor_torque_directions, (TI)2));
    ASSERT_EQ(rlt::num_coordinates(device, assembly), 2);
    EXPECT_EQ(assembly.parts[1].joint.coordinate, 0);
    EXPECT_EQ(assembly.parts[2].joint.coordinate, 1);
    EXPECT_EQ(rotorcraft.prop_rotor[0], 1);
    EXPECT_EQ(rotorcraft.prop_rotor[1], 0);
    EXPECT_EQ(rotorcraft.prop_spin[0], (T)1);
    EXPECT_EQ(rotorcraft.prop_spin[1], (T)-1);
    // prop_0 follows rotor 1: a quarter turn of rotor 1 moves prop_0, not prop_1
    const float base[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float phases[2] = {0, (float)M_PI / 2};
    float joint_coordinates[2];
    rlt::coordinates(device, rotorcraft, phases, joint_coordinates);
    const auto rows = pose(device, assembly, base, 1, joint_coordinates);
    const float tip[3] = {-0.05f, 0.1f, 0};
    float tip_world[3];
    rlt::hyperlux::transform_point(rows.part(2), tip, tip_world);
    EXPECT_NEAR(tip_world[0], -0.05f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.1f, 1e-5f);
    const float tip_moving[3] = {0.15f, 0.1f, 0};
    rlt::hyperlux::transform_point(rows.part(1), tip_moving, tip_world);
    EXPECT_NEAR(tip_world[0], 0.1f, 1e-5f);
    EXPECT_NEAR(tip_world[1], 0.15f, 1e-5f);

    const T one_rotor[1][3] = {{0.1, 0.1, 0}};
    const T one_torque[1][3] = {{0, 0, 1}};
    assembly = two_props;
    EXPECT_FALSE(rlt::init(device, rotorcraft, assembly, one_rotor, one_torque, (TI)1));
    const T coincident[2][3] = {{0.1, 0.1, 0}, {0.1, 0.1, 0}};
    EXPECT_FALSE(rlt::init(device, rotorcraft, assembly, coincident, rotor_torque_directions, (TI)2));
    const T no_torque[2][3] = {{0, 0, 0}, {0, 0, -1}};
    EXPECT_FALSE(rlt::init(device, rotorcraft, assembly, rotor_positions, no_torque, (TI)2));
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, CONVENTION){
    DEVICE device;
    ROTORCRAFT rotorcraft;
    ASSEMBLY empty;
    EXPECT_FALSE(rlt::articulate(device, empty));
    EXPECT_FALSE(rlt::init(device, rotorcraft, empty));

    ASSEMBLY wrong_root;
    wrong_root.objects.resize(1);
    wrong_root.objects[0].name = "prop_0";
    wrong_root.parts.resize(1);
    wrong_root.parts[0].object = 0;
    EXPECT_FALSE(rlt::articulate(device, wrong_root));
    EXPECT_FALSE(rlt::init(device, rotorcraft, wrong_root));

    // any number of props articulates (a fit to a dynamics model needs one per rotor)
    ASSEMBLY hexacopter;
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    rlt::add(device, hexacopter, quad("body", 0, 0, 0.1f), identity);
    for(int prop_i = 0; prop_i < 6; prop_i++){
        rlt::add(device, hexacopter, quad("prop_", 0.1f * std::cos(prop_i + 0.5f), 0.1f * std::sin(prop_i + 0.5f), 0.05f), identity);
    }
    // init reads the joints and never writes them: a copy pooled before articulate stays rigid
    const auto rigid = hexacopter;
    EXPECT_FALSE(rlt::init(device, rotorcraft, rigid));
    ASSERT_TRUE(rlt::articulate(device, hexacopter));
    EXPECT_TRUE(rlt::init(device, rotorcraft, hexacopter));
    EXPECT_EQ(rotorcraft.num_props, 6);
    EXPECT_EQ(rlt::num_coordinates(device, hexacopter), 6);
    EXPECT_EQ(rlt::num_coordinates(device, rigid), 0);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, BODY_TRANSFORM){
    DEVICE device;
    const T orientation[4] = {1, 0, 0, 0};
    const T position[3] = {1, 2, 3};
    const T translation[3] = {10, 20, 30};
    float transform[12];
    rlt::rl::environments::hyperdrone::make_body_transform(device, orientation, position, translation, (T)1, (T)0, transform);
    const float expected[12] = {1,0,0,11, 0,1,0,22, 0,0,1,33};
    for(TI element_i = 0; element_i < 12; element_i++){
        EXPECT_FLOAT_EQ(transform[element_i], expected[element_i]);
    }
    // yaw by 90 degrees: x -> y
    rlt::rl::environments::hyperdrone::make_body_transform(device, orientation, position, translation, (T)0, (T)1, transform);
    EXPECT_FLOAT_EQ(transform[0 * 4 + 3], 10 - 2);
    EXPECT_FLOAT_EQ(transform[1 * 4 + 3], 20 + 1);
    EXPECT_FLOAT_EQ(transform[2 * 4 + 3], 30 + 3);
    EXPECT_FLOAT_EQ(transform[0 * 4 + 0], 0);
    EXPECT_FLOAT_EQ(transform[1 * 4 + 0], 1);
}

// every rigged assembly of the test corpus follows the body/prop_* convention with four props
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT, INIT_RIGGED_CORPUS){
    if(DRONE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    const std::string directory = DRONE_PATH.substr(0, DRONE_PATH.rfind('/'));
    for(const char* name : {"arpl.glb", "crazyflie.glb", "crazyflie_brushless.glb", "soft.glb", "x500.glb"}){
        ASSEMBLY assembly;
        ASSERT_TRUE((rlt::load<rlt::hyperlux::High, true>(device, assembly, directory + "/" + name))) << name;
        ASSERT_TRUE(rlt::articulate(device, assembly)) << name;
        ROTORCRAFT rotorcraft;
        ASSERT_TRUE(rlt::init(device, rotorcraft, assembly)) << name;
        EXPECT_EQ(rlt::num_coordinates(device, assembly), 4) << name;
        EXPECT_EQ(assembly.parts.size(), 5) << name;
        rlt::hyperlux::AssetPool pool;
        EXPECT_TRUE(rlt::can_add(device, pool, assembly)) << name;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
