#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
    using namespace rl_tools;
    namespace rrt = rendering::raytracing;
    using DEVICE = devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using Matrix4 = std::array<double, 16>;

    template <bool DYNAMIC_MOTION_BLUR>
    struct Config: rrt::config::Default<float, TI> {
        static constexpr TI CAM_WIDTH = 8, CAM_HEIGHT = 8, NUM_CAMERAS = 1;
        static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
        static constexpr bool ENABLE_MOTION_BLUR = DYNAMIC_MOTION_BLUR;
        static constexpr TI MOTION_BLUR_SAMPLES = DYNAMIC_MOTION_BLUR ? 4 : 1;
        static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = DYNAMIC_MOTION_BLUR;
    };
    using STATIC_SPEC = rrt::Specification<Config<false>>;
    using MOTION_SPEC = rrt::Specification<Config<true>>;

    Matrix4 as_matrix(const float transform[12]){
        Matrix4 out{};
        for(int element = 0; element < 12; element++){
            out[element] = transform[element];
        }
        out[15] = 1;
        return out;
    }
    Matrix4 operator*(const Matrix4& a, const Matrix4& b){
        Matrix4 out{};
        for(int row = 0; row < 4; row++){
            for(int column = 0; column < 4; column++){
                for(int k = 0; k < 4; k++){
                    out[row * 4 + column] += a[row * 4 + k] * b[k * 4 + column];
                }
            }
        }
        return out;
    }
    Matrix4 translation(double x, double y, double z){
        return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
    }
    Matrix4 rotation_x(double angle){
        const double c = std::cos(angle), s = std::sin(angle);
        return {1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1};
    }
    Matrix4 rotation_z(double angle){
        const double c = std::cos(angle), s = std::sin(angle);
        return {c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }
    Matrix4 inverse_rigid(const Matrix4& m){
        Matrix4 out{};
        for(int row = 0; row < 3; row++){
            for(int column = 0; column < 3; column++){
                out[row * 4 + column] = m[column * 4 + row];
            }
            out[row * 4 + 3] = -(m[row] * m[3] + m[4 + row] * m[7] + m[8 + row] * m[11]);
        }
        out[15] = 1;
        return out;
    }
    std::array<float, 12> rows(const Matrix4& m){
        std::array<float, 12> out;
        for(int element = 0; element < 12; element++){
            out[element] = (float)m[element];
        }
        return out;
    }
    void expect_row_near(const float* actual, const Matrix4& expected, double tolerance){
        for(int element = 0; element < 12; element++){
            EXPECT_NEAR(actual[element], expected[element], tolerance) << "element " << element;
        }
    }

    rrt::Object cube(float half_extent){
        rrt::Mesh mesh;
        for(int corner = 0; corner < 8; corner++){
            mesh.vertices.push_back(corner & 1 ? half_extent : -half_extent);
            mesh.vertices.push_back(corner & 2 ? half_extent : -half_extent);
            mesh.vertices.push_back(corner & 4 ? half_extent : -half_extent);
        }
        mesh.indices = {0, 1, 3, 0, 3, 2, 4, 7, 5, 4, 6, 7};
        rrt::Object object;
        object.meshes.push_back(mesh);
        return object;
    }

    // revolute arm about x through a pivot off its origin (coordinate 1), a slider riding it
    // along y (coordinate 0), a fixed cap riding the slider and an independent fixed plate
    const Matrix4 ARM_REST = translation(0, 2, 3), SLIDER_REST = translation(1, 2, 3) * rotation_z(0.5), CAP_REST = translation(1, 2, 4), PLATE_REST = rotation_z(-0.4) * translation(0, 0, -1);
    constexpr float ARM_PIVOT[3] = {0, -2, -3};

    rrt::ObjectAssembly chain(DEVICE& device){
        rrt::ObjectAssembly assembly;
        const auto arm = add(device, assembly, cube(1), rows(ARM_REST).data());
        const auto slider = add(device, assembly, cube(.5f), rows(SLIDER_REST).data());
        const auto cap = add(device, assembly, cube(.25f), rows(CAP_REST).data());
        add(device, assembly, cube(2), rows(PLATE_REST).data());
        assembly.parts[slider].parent = arm;
        assembly.parts[cap].parent = slider;
        auto& hinge = assembly.parts[arm].joint;
        hinge.type = rendering::Joint::Type::REVOLUTE;
        hinge.axis[0] = 2;
        hinge.axis[2] = 0;
        std::memcpy(hinge.pivot, ARM_PIVOT, sizeof(hinge.pivot));
        hinge.coordinate = 1;
        auto& rail = assembly.parts[slider].joint;
        rail.type = rendering::Joint::Type::PRISMATIC;
        rail.axis[1] = 3;
        rail.axis[2] = 0;
        rail.coordinate = 0;
        return assembly;
    }

    rendering::Bundle<float> scene(DEVICE& device){
        rendering::Bundle<float> bundle;
        add(device, bundle.scene, cube(1));
        rendering::datasets::compute_bounds(device, bundle);
        return bundle;
    }

    template <typename SPEC>
    struct Harness {
        DEVICE device;
        rendering::Bundle<float> bundle;
        rrt::AssetPool pool;
        rrt::Renderer<SPEC, rrt::backends::Generic> renderer;
        Harness(const std::vector<rrt::ObjectAssembly>& assets){
            init(device);
            device.logger.verbosity = devices::logging::Verbosity::warning;
            bundle = scene(device);
            for(const auto& asset : assets){
                add(device, pool, asset);
            }
            malloc(device, renderer);
            init(device, renderer, bundle, pool);
        }
        ~Harness(){
            free(device, renderer);
        }
        const float* row(const rrt::OverlayPlacement& placement, TI part){
            update(device, renderer);
            return data(renderer.transforms) + (placement.first_slot + part) * 12;
        }
        const float* sample_row(const rrt::OverlayPlacement& placement, TI sample, TI part){
            update(device, renderer);
            return data(renderer.transforms_motion) + (sample * SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part) * 12;
        }
    };

    Matrix4 scaled(const Matrix4& base, double scale){
        Matrix4 out = base;
        for(int row = 0; row < 3; row++){
            for(int column = 0; column < 3; column++){
                out[row * 4 + column] *= scale;
            }
        }
        return out;
    }

    std::array<Matrix4, 4> expected_chain(const Matrix4& base, double scale, const float coordinates[2]){
        const Matrix4 placement = scaled(base, scale);
        const Matrix4 hinge = translation(ARM_PIVOT[0], ARM_PIVOT[1], ARM_PIVOT[2]) * rotation_x(coordinates[1]) * translation(-ARM_PIVOT[0], -ARM_PIVOT[1], -ARM_PIVOT[2]);
        const Matrix4 arm = placement * ARM_REST * hinge;
        const Matrix4 slider = arm * inverse_rigid(ARM_REST) * SLIDER_REST * translation(0, coordinates[0], 0);
        const Matrix4 cap = slider * inverse_rigid(SLIDER_REST) * CAP_REST;
        return {arm, slider, cap, placement * PLATE_REST};
    }
}

TEST(RENDERING_RAYTRACING_POSE, JOINTS_COMPOSE_THROUGH_THE_PART_TREE){
    DEVICE device;
    Harness<STATIC_SPEC> harness({chain(device)});
    EXPECT_EQ(harness.renderer.assets[0].num_coordinates, 2u);
    const auto placement = spawn(harness.device, harness.renderer, rrt::OverlayIndex{0}, rrt::AssetHandle{0});
    const Matrix4 base = translation(4, -1, .5) * rotation_z(0.3);
    const float coordinates[2] = {0.75f, 1.2f};
    set_pose(harness.device, harness.renderer, rrt::OverlayIndex{0}, placement, make_pose(rows(base).data(), 1.5f, coordinates));
    const auto expected = expected_chain(base, 1.5, coordinates);
    for(TI part = 0; part < 4; part++){
        expect_row_near(harness.row(placement, part), expected[part], 1e-5);
    }
}

TEST(RENDERING_RAYTRACING_POSE, RIGID_PARTS_COMPOSE_BASE_AND_REST_EXACTLY){
    DEVICE device;
    auto assembly = chain(device);
    assembly.parts[0].joint = {};
    assembly.parts[1].joint = {};
    Harness<STATIC_SPEC> harness({assembly});
    const float base[12] = {0, -1, 0, 2, 1, 0, 0, 3, 0, 0, 1, 4};
    const auto placement = spawn(harness.device, harness.renderer, rrt::OverlayIndex{0}, rrt::AssetHandle{0}, make_pose(base));
    for(TI part = 0; part < 4; part++){
        float expected[12];
        compose_transforms(base, assembly.parts[part].transform, expected);
        EXPECT_EQ(std::memcmp(harness.row(placement, part), expected, sizeof(expected)), 0) << "part " << part;
    }
}

// two full turns of the hinge across the shutter: a shortest-arc interpolation would not move it
TEST(RENDERING_RAYTRACING_POSE, SHUTTER_PAIR_INTERPOLATES_BASE_AND_KEEPS_REVOLUTIONS){
    DEVICE device;
    Harness<MOTION_SPEC> harness({chain(device)});
    const auto placement = spawn(harness.device, harness.renderer, rrt::OverlayIndex{0}, rrt::AssetHandle{0});
    const auto open_base = rows(translation(1, 0, 0) * rotation_z(0.2));
    const auto close_base = rows(translation(3, 1, 0) * rotation_z(0.6));
    const float open_coordinates[2] = {0, 0}, close_coordinates[2] = {0.5f, (float)(4 * std::acos(-1.0))};
    set_pose(harness.device, harness.renderer, rrt::OverlayIndex{0}, placement, make_pose(open_base.data(), 1, open_coordinates), make_pose(close_base.data(), 2, close_coordinates));
    for(TI sample = 0; sample < MOTION_SPEC::MOTION_BLUR_SAMPLES; sample++){
        const float t = ((float)sample + 0.5f) / (float)MOTION_SPEC::MOTION_BLUR_SAMPLES;
        float base[12];
        rendering::screw_interpolate(open_base.data(), close_base.data(), t, base);
        const float coordinates[2] = {t * close_coordinates[0], t * close_coordinates[1]};
        const auto expected = expected_chain(as_matrix(base), 1 + t, coordinates);
        for(TI part = 0; part < 4; part++){
            expect_row_near(harness.sample_row(placement, sample, part), expected[part], 1e-4);
        }
    }
    const auto close = expected_chain(as_matrix(close_base.data()), 2, close_coordinates);
    for(TI part = 0; part < 4; part++){
        expect_row_near(harness.row(placement, part), close[part], 1e-4);
    }
}

TEST(RENDERING_RAYTRACING_POSE, STATIC_POSE_FILLS_EVERY_MOTION_SAMPLE){
    DEVICE device;
    Harness<MOTION_SPEC> harness({chain(device)});
    const auto placement = spawn(harness.device, harness.renderer, rrt::OverlayIndex{0}, rrt::AssetHandle{0});
    const float coordinates[2] = {0.3f, -0.8f};
    set_pose(harness.device, harness.renderer, rrt::OverlayIndex{0}, placement, make_pose(rows(translation(2, 0, 1)).data(), 1, coordinates));
    for(TI sample = 0; sample < MOTION_SPEC::MOTION_BLUR_SAMPLES; sample++){
        for(TI part = 0; part < 4; part++){
            EXPECT_EQ(std::memcmp(harness.sample_row(placement, sample, part), harness.row(placement, part), 12 * sizeof(float)), 0);
        }
    }
}

TEST(RENDERING_RAYTRACING_POSE, REPLACE_HANDS_OUT_THE_NEW_ASSETS_PLACEMENT){
    DEVICE device;
    auto other = chain(device);
    other.parts[0].joint.axis[0] = 0;
    other.parts[0].joint.axis[2] = 1;
    Harness<STATIC_SPEC> harness({chain(device), other});
    const auto placement = spawn(harness.device, harness.renderer, rrt::OverlayIndex{0}, rrt::AssetHandle{0});
    const auto replaced = replace(harness.device, harness.renderer, rrt::OverlayIndex{0}, placement, rrt::AssetHandle{1});
    EXPECT_EQ(replaced.first_slot, placement.first_slot);
    EXPECT_FALSE(rrt::detail::placement_valid(harness.renderer, rrt::OverlayIndex{0}, placement));
    EXPECT_TRUE(rrt::detail::placement_valid(harness.renderer, rrt::OverlayIndex{0}, replaced));
}

TEST(RENDERING_RAYTRACING_POSE, CAN_ADD_VALIDATES_PARTS_AND_JOINTS){
    DEVICE device;
    init(device);
    rrt::AssetPool pool;
    EXPECT_TRUE(can_add(device, pool, chain(device)));
    EXPECT_FALSE(can_add(device, pool, rrt::ObjectAssembly{}));
    auto coupled = chain(device);
    coupled.parts[0].joint.coordinate = 0;
    EXPECT_TRUE(can_add(device, pool, coupled));
    EXPECT_EQ(num_coordinates(device, coupled), 1u);
    auto gap = chain(device);
    gap.parts[0].joint.coordinate = 2;
    EXPECT_FALSE(can_add(device, pool, gap));
    auto zero_axis = chain(device);
    zero_axis.parts[1].joint.axis[1] = 0;
    EXPECT_FALSE(can_add(device, pool, zero_axis));
    auto infinite_pivot = chain(device);
    infinite_pivot.parts[0].joint.pivot[1] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(can_add(device, pool, infinite_pivot));
    auto child_first = chain(device);
    child_first.parts[0].parent = 1;
    EXPECT_FALSE(can_add(device, pool, child_first));
    auto broken_frame = chain(device);
    broken_frame.parts[3].transform[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(can_add(device, pool, broken_frame));
}
