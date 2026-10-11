#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/rig/operations_cpu.h>
#include <rl_tools/rendering/raytracing/operations_cpu_common.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <cstdint>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using SPEC = rlt::rendering::rig::Specification<T, TI, 4, 4>;
using Model = rlt::rendering::rig::Model<SPEC>;
using Joint = rlt::rendering::rig::Joint;

namespace {
    void translation(float x, float y, float z, float out[12]){
        const float transform[12] = {1,0,0,x, 0,1,0,y, 0,0,1,z};
        std::memcpy(out, transform, sizeof(transform));
    }
    void rotation_x(float angle, float out[12]){
        const float c = std::cos(angle), s = std::sin(angle);
        const float transform[12] = {1,0,0,0, 0,c,-s,0, 0,s,c,0};
        std::memcpy(out, transform, sizeof(transform));
    }
    void rotation_z(float angle, float out[12]){
        const float c = std::cos(angle), s = std::sin(angle);
        const float transform[12] = {c,-s,0,0, s,c,0,0, 0,0,1,0};
        std::memcpy(out, transform, sizeof(transform));
    }
    void expect_transform_near(const float actual[12], const float expected[12], float tolerance){
        for(int element = 0; element < 12; element++){
            EXPECT_NEAR(actual[element], expected[element], tolerance) << "element " << element;
        }
    }
    // revolute about x at an off-origin pivot, prismatic along y below it, spherical below that;
    // parts map geometry onto links out of order
    Model chain(DEVICE& device){
        Model model;
        model.num_links = 3;
        model.num_parts = 4;
        model.links[0].joint = Joint::REVOLUTE;
        model.links[0].axis[0] = 2;
        model.links[0].axis[2] = 0;
        translation(0, 2, 3, model.links[0].origin);
        model.links[1].parent = 0;
        model.links[1].joint = Joint::PRISMATIC;
        model.links[1].axis[1] = 3;
        model.links[1].axis[2] = 0;
        translation(1, 0, 0, model.links[1].origin);
        model.links[2].parent = 1;
        model.links[2].joint = Joint::SPHERICAL;
        translation(0, 0, 1, model.links[2].origin);
        model.parts[0].link = 1;
        model.parts[1].link = 0;
        translation(0, -2, -3, model.parts[1].offset);
        model.parts[2].link = 2;
        translation(2, 0, 0, model.parts[2].offset);
        model.parts[3].link = 0;
        translation(0, 0, 2, model.parts[3].offset);
        EXPECT_TRUE(rlt::init(device, model));
        return model;
    }
}

TEST(RL_TOOLS_RENDERING_RIG, FORWARD_KINEMATICS){
    DEVICE device;
    Model model = chain(device);
    ASSERT_EQ(model.num_coordinates, 6);
    EXPECT_EQ(model.links[0].coordinate, 0);
    EXPECT_EQ(model.links[1].coordinate, 1);
    EXPECT_EQ(model.links[2].coordinate, 2);
    EXPECT_FLOAT_EQ(model.links[0].axis[0], 1);
    EXPECT_FLOAT_EQ(model.links[1].axis[1], 1);

    const float quarter = (float)M_PI / 2;
    const float coordinates[6] = {quarter, 0.75f, std::sqrt(0.5f), 0, 0, std::sqrt(0.5f)};
    float base[12];
    translation(4, 0, 0, base);
    rlt::rendering::rig::Buffer<SPEC> buffer;
    float poses[4][12];
    rlt::evaluate(device, model, base, coordinates, buffer, &poses[0][0]);

    float frame[12], link0[12], link1[12], link2[12], expected[12], turn[12];
    rlt::rendering::compose_transforms(base, model.links[0].origin, frame);
    rotation_x(quarter, turn);
    rlt::rendering::compose_transforms(frame, turn, link0);
    rlt::rendering::compose_transforms(link0, model.links[1].origin, frame);
    float slide[12];
    translation(0, 0.75f, 0, slide);
    rlt::rendering::compose_transforms(frame, slide, link1);
    rlt::rendering::compose_transforms(link1, model.links[2].origin, frame);
    rotation_z(quarter, turn);
    rlt::rendering::compose_transforms(frame, turn, link2);

    expect_transform_near(poses[0], link1, 1e-6f);
    rlt::rendering::compose_transforms(link0, model.parts[1].offset, expected);
    expect_transform_near(poses[1], expected, 1e-6f);
    rlt::rendering::compose_transforms(link2, model.parts[2].offset, expected);
    expect_transform_near(poses[2], expected, 1e-6f);
    rlt::rendering::compose_transforms(link0, model.parts[3].offset, expected);
    expect_transform_near(poses[3], expected, 1e-6f);

    // the pivot of the revolute link stays put: part 1 cancels the origin, so its frame's pivot
    // point (0, 2, 3) lands where the base puts it
    const float pivot[3] = {0, 2, 3};
    float pivot_world[3];
    rlt::rendering::transform_point(poses[1], pivot, pivot_world);
    EXPECT_NEAR(pivot_world[0], 4, 1e-6f);
    EXPECT_NEAR(pivot_world[1], 2, 1e-6f);
    EXPECT_NEAR(pivot_world[2], 3, 1e-6f);
}

TEST(RL_TOOLS_RENDERING_RIG, INTERPOLATION_KEEPS_REVOLUTIONS_AND_SLERPS){
    DEVICE device;
    Model model = chain(device);
    const float open[6] = {0, 0, 1, 0, 0, 0};
    const float close[6] = {4 * (float)M_PI, 1, -1, 0, 0, -1}; // the quaternion is the negated 90 degree z turn
    float interpolated[6];
    rlt::interpolate(device, model, open, close, 0.125f, interpolated);
    EXPECT_NEAR(interpolated[0], (float)M_PI / 2, 1e-5f);
    EXPECT_NEAR(interpolated[1], 0.125f, 1e-6f);
    rlt::interpolate(device, model, open, close, 0.5f, interpolated);
    const float half_angle = (float)M_PI / 8;
    EXPECT_NEAR(interpolated[2], std::cos(half_angle), 1e-5f);
    EXPECT_NEAR(interpolated[5], std::sin(half_angle), 1e-5f);
    EXPECT_NEAR(interpolated[3], 0, 1e-6f);
    EXPECT_NEAR(interpolated[4], 0, 1e-6f);
}

TEST(RL_TOOLS_RENDERING_RIG, SAMPLE_MATCHES_INTERPOLATED_EVALUATION){
    DEVICE device;
    Model model = chain(device);
    const float open[6] = {0, 0, 1, 0, 0, 0};
    const float close[6] = {7, 0.5f, std::sqrt(0.5f), 0, 0, std::sqrt(0.5f)};
    float base_open[12], base_close[12];
    translation(4, 0, 0, base_open);
    float turn[12];
    rotation_z(0.6f, turn);
    float shift[12];
    translation(5, 1, 0, shift);
    rlt::rendering::compose_transforms(shift, turn, base_close);
    constexpr TI SAMPLES = 4;
    constexpr TI STRIDE = 4 * 12 + 24;
    float samples[SAMPLES * STRIDE], close_poses[4][12];
    rlt::rendering::rig::Buffer<SPEC> buffer;
    rlt::sample(device, model, base_open, base_close, open, close, SAMPLES, buffer, samples, STRIDE, &close_poses[0][0]);
    for(TI sample = 0; sample < SAMPLES; sample++){
        const float t = ((float)sample + 0.5f) / (float)SAMPLES;
        float base[12], coordinates[6], expected[4][12];
        rlt::rendering::screw_interpolate(base_open, base_close, t, base);
        rlt::interpolate(device, model, open, close, t, coordinates);
        rlt::evaluate(device, model, base, coordinates, buffer, &expected[0][0]);
        for(TI part = 0; part < 4; part++){
            expect_transform_near(samples + sample * STRIDE + part * 12, expected[part], 1e-6f);
        }
    }
    float expected_close[4][12];
    rlt::evaluate(device, model, base_close, close, buffer, &expected_close[0][0]);
    for(TI part = 0; part < 4; part++){
        expect_transform_near(close_poses[part], expected_close[part], 0);
    }
}

TEST(RL_TOOLS_RENDERING_RIG, ASSEMBLY_REST_FRAMES){
    DEVICE device;
    rlt::rendering::ObjectAssembly assembly;
    rlt::rendering::Object object;
    float rest[12];
    translation(1, 0, 0, rest);
    rlt::add(device, assembly, object, rest);
    float child_rest[12], turn[12];
    rotation_z(0.3f, turn);
    float shift[12];
    translation(1, 2, 0, shift);
    rlt::rendering::compose_transforms(shift, turn, child_rest);
    rlt::add(device, assembly, object, child_rest);
    assembly.parts[1].parent = 0;
    Model model;
    ASSERT_TRUE(rlt::init(device, model, assembly));
    EXPECT_EQ(model.num_links, 2);
    EXPECT_EQ(model.links[1].parent, 0);
    EXPECT_EQ(model.num_coordinates, 0);
    float pose[12];
    translation(3, 0, 1, pose);
    rlt::rendering::rig::Buffer<SPEC> buffer;
    float poses[2][12];
    const float no_coordinates[SPEC::MAX_COORDINATES] = {0};
    rlt::evaluate(device, model, pose, no_coordinates, buffer, &poses[0][0]);
    float expected[12];
    rlt::rendering::compose_transforms(pose, rest, expected);
    expect_transform_near(poses[0], expected, 0);
    rlt::rendering::compose_transforms(pose, child_rest, expected);
    expect_transform_near(poses[1], expected, 1e-6f);

    assembly.parts[0].parent = 1; // a parent declared after its child
    EXPECT_FALSE(rlt::init(device, model, assembly));
}

TEST(RL_TOOLS_RENDERING_RIG, REJECTS_INVALID_MODELS){
    DEVICE device;
    Model empty;
    EXPECT_FALSE(rlt::init(device, empty));

    Model forward_parent = chain(device);
    forward_parent.links[0].parent = 1;
    EXPECT_FALSE(rlt::init(device, forward_parent));

    Model self_parent = chain(device);
    self_parent.links[1].parent = 1;
    EXPECT_FALSE(rlt::init(device, self_parent));

    Model zero_axis = chain(device);
    zero_axis.links[0].axis[0] = 0;
    zero_axis.links[0].axis[1] = 0;
    zero_axis.links[0].axis[2] = 0;
    EXPECT_FALSE(rlt::init(device, zero_axis));

    Model dangling_part = chain(device);
    dangling_part.parts[2].link = 3;
    EXPECT_FALSE(rlt::init(device, dangling_part));

    Model non_finite = chain(device);
    const std::uint32_t infinity_bits = 0x7f800000u;
    std::memcpy(&non_finite.links[2].origin[3], &infinity_bits, sizeof(infinity_bits));
    EXPECT_FALSE(rlt::rendering::rig::detail::is_finite(non_finite.links[2].origin[3]));
    EXPECT_FALSE(rlt::init(device, non_finite));

    Model too_many = chain(device);
    too_many.num_links = 5;
    EXPECT_FALSE(rlt::init(device, too_many));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
