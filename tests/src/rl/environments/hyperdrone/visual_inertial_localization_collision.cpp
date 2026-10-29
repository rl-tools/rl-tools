#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/collision_operations_cpu.h>

#include <gtest/gtest.h>
#include <array>
#include <limits>

namespace rlt = rl_tools;
namespace collision = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::collision;
using rlt::init;
using rlt::free;
using rlt::collision_free;
using rlt::bezier_collision_free;
using CollisionScene = collision::Scene<collision::Specification<double, std::size_t>>;
using Point = std::array<double, 3>;

namespace {
    template <typename T, typename BITS>
    void set_bits(T& target, BITS value) {
        static_assert(sizeof(T) == sizeof(BITS));
        volatile BITS opaque = value;
        const BITS bits = opaque;
        std::memcpy(&target, &bits, sizeof(bits));
    }

    rlt::hyperlux::Scene triangle_scene(const std::array<float, 9>& vertices = {0, 0, 0, 2, 0, 0, 0, 2, 0}) {
        rlt::hyperlux::Scene source;
        source.objects.resize(1);
        source.objects[0].meshes.resize(1);
        auto& mesh = source.objects[0].meshes[0];
        mesh.vertices.assign(vertices.begin(), vertices.end());
        mesh.indices = {0, 1, 2};
        source.instances.push_back({0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, true});
        return source;
    }

    struct HyperdroneVisualInertialLocalizationCollision : ::testing::Test {
        rlt::devices::DefaultCPU device;
        CollisionScene scene;
        void load(const rlt::hyperlux::Scene& source = triangle_scene()) {
            ASSERT_TRUE(init(device, scene, source));
        }
        bool clear(Point from, Point to, double radius = 0) {
            return collision_free(device, scene, from.data(), to.data(), radius);
        }
        void TearDown() override {
            free(device, scene);
        }
    };
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, ThinWallBetweenClearEndpoints) {
    load();
    EXPECT_TRUE(clear({0.5, 0.5, -1}, {0.5, 0.5, -1}, 0.1));
    EXPECT_TRUE(clear({0.5, 0.5, 1}, {0.5, 0.5, 1}, 0.1));
    EXPECT_FALSE(clear({0.5, 0.5, -1}, {0.5, 0.5, 1}));
    EXPECT_FALSE(clear({0.5, 0.5, 1}, {0.5, 0.5, -1}, 0.1));
    EXPECT_TRUE(clear({3, 3, -1}, {3, 3, 1}, 0.1));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, SweptVolumeGrazesFaceEdgeAndVertex) {
    load();
    EXPECT_FALSE(clear({0.25, 0.25, 0.1}, {0.5, 0.25, 0.1}, 0.1));
    EXPECT_TRUE(clear({0.25, 0.25, 0.1}, {0.5, 0.25, 0.1}, 0.09));
    EXPECT_FALSE(clear({0.25, -0.1, 0}, {1.5, -0.1, 0}, 0.1));
    EXPECT_TRUE(clear({0.25, -0.1, 0}, {1.5, -0.1, 0}, 0.09));
    EXPECT_FALSE(clear({-0.1, -0.1, 0}, {-0.1, -0.1, 0}, 0.142));
    EXPECT_TRUE(clear({-0.1, -0.1, 0}, {-0.1, -0.1, 0}, 0.14));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, CoplanarSegmentsAndStationarySphere) {
    load();
    EXPECT_FALSE(clear({-1, 0.5, 0}, {3, 0.5, 0}));
    EXPECT_FALSE(clear({0.5, 0.5, 0}, {0.5, 0.5, 0}));
    EXPECT_FALSE(clear({0, 0, 0}, {2, 0, 0}));
    EXPECT_TRUE(clear({-1, -1, 0}, {-0.5, -0.5, 0}, 0.1));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, DegeneratePointAndEdgeRemainObstacles) {
    load(triangle_scene({0, 0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_FALSE(clear({-1, 0, 0}, {1, 0, 0}));
    EXPECT_FALSE(clear({-1, 0.1, 0}, {1, 0.1, 0}, 0.1));
    EXPECT_TRUE(clear({-1, 0.2, 0}, {1, 0.2, 0}, 0.1));
    load(triangle_scene({0, -1, 0, 0, 0, 0, 0, 1, 0}));
    EXPECT_FALSE(clear({-1, 0.5, 0}, {1, 0.5, 0}));
    EXPECT_TRUE(clear({-1, 2, 0}, {1, 2, 0}, 0.1));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, TinyTriangleAndAlmostParallelSegments) {
    load(triangle_scene({0, 0, 0, 1e-9f, 0, 0, 0, 1e-9f, 0}));
    EXPECT_FALSE(clear({2e-10, 2e-10, -1e-9}, {2e-10, 2e-10, 1e-9}));
    EXPECT_TRUE(clear({2e-9, 2e-9, -1e-9}, {2e-9, 2e-9, 1e-9}));
    load();
    EXPECT_FALSE(clear({0.1, 0.1, -1e-12}, {0.5, 0.1, 1e-12}));
    EXPECT_TRUE(clear({0.1, 0.1, 1e-5}, {0.5, 0.1, 1e-5 + 1e-12}));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, AffineInstancesIncludingScaleAndShear) {
    auto source = triangle_scene();
    const float transform[12] = {-2, 0.5f, 0, 3, 0, 3, 0, 4, 0, 0, 0.5f, 5};
    std::copy(transform, transform + 12, source.instances[0].transform);
    source.instances[0].identity = false;
    source.instances.push_back({0, {1, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1, 0}, false});
    load(source);
    EXPECT_FALSE(clear({2, 5, 4}, {2, 5, 6}));
    EXPECT_FALSE(clear({20.5, 0.5, -1}, {20.5, 0.5, 1}));
    EXPECT_TRUE(clear({0.5, 0.5, -1}, {0.5, 0.5, 1}));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, InvalidDataFailsClosed) {
    double nan, infinity;
    set_bits(nan, UINT64_C(0x7ff8000000000000));
    set_bits(infinity, UINT64_C(0x7ff0000000000000));
    EXPECT_FALSE(clear({0, 0, 0}, {1, 1, 1}));
    load();
    EXPECT_FALSE(clear({nan, 0, 0}, {1, 1, 1}));
    EXPECT_FALSE(clear({0, 0, 0}, {infinity, 1, 1}));
    EXPECT_FALSE(clear({0, 0, 5}, {1, 1, 5}, nan));
    EXPECT_FALSE(clear({0, 0, 5}, {1, 1, 5}, infinity));
    EXPECT_FALSE(clear({0, 0, 5}, {1, 1, 5}, -0.1));
    auto source = triangle_scene();
    set_bits(source.objects[0].meshes[0].vertices[0], UINT32_C(0x7fc00000));
    EXPECT_FALSE(init(device, scene, source));
    EXPECT_FALSE(clear({0, 0, 5}, {1, 1, 5}));
    source = triangle_scene();
    source.objects[0].meshes[0].indices[0] = 1000;
    EXPECT_FALSE(init(device, scene, source));
    source = triangle_scene();
    set_bits(source.instances[0].transform[0], UINT32_C(0x7f800000));
    EXPECT_FALSE(init(device, scene, source));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, BvhCoversEveryInstanceAndCanBeReinitialized) {
    auto source = triangle_scene();
    for (int index = 1; index < 100; index++) {
        source.instances.push_back({0, {1, 0, 0, (float)(index * 4), 0, 1, 0, 0, 0, 0, 1, 0}, false});
    }
    load(source);
    EXPECT_GT(scene.nodes.size(), 1u);
    for (int index = 0; index < 100; index++) {
        const double x = index * 4;
        EXPECT_FALSE(clear({x + 0.5, 0.5, -1}, {x + 0.5, 0.5, 1}, 0.1));
        EXPECT_TRUE(clear({x + 2.5, 0.5, -1}, {x + 2.5, 0.5, 1}, 0.1));
    }
    load(rlt::hyperlux::Scene{});
    EXPECT_TRUE(clear({0.5, 0.5, -1}, {0.5, 0.5, 1}));
    free(device, scene);
    EXPECT_FALSE(clear({0.5, 0.5, -1}, {0.5, 0.5, 1}));
}

TEST_F(HyperdroneVisualInertialLocalizationCollision, BezierCertificationChecksTheWholeCurve) {
    load();
    const double straight[6][3] = {{0.5, 0.5, -1}, {0.5, 0.5, -0.6}, {0.5, 0.5, -0.2}, {0.5, 0.5, 0.2}, {0.5, 0.5, 0.6}, {0.5, 0.5, 1}};
    EXPECT_FALSE(bezier_collision_free(device, scene, straight, 0.1));
    const double dipping[4][3] = {{0.25, 0.25, 0.5}, {0.25, 0.25, -0.5}, {0.5, 0.25, -0.5}, {0.5, 0.25, 0.5}};
    EXPECT_TRUE(clear({0.25, 0.25, 0.5}, {0.5, 0.25, 0.5}, 0.1));
    EXPECT_FALSE(bezier_collision_free(device, scene, dipping, 0.1));
    const double above[4][3] = {{0.25, 0.25, 0.5}, {0.25, 0.25, 1}, {0.5, 0.25, 1}, {0.5, 0.25, 0.5}};
    EXPECT_TRUE(bezier_collision_free(device, scene, above, 0.1));
    const float quintic[6][3] = {{0.25f, 0.25f, 0.5f}, {0.25f, 0.25f, 1}, {0.3f, 0.25f, 1}, {0.4f, 0.25f, 1}, {0.5f, 0.25f, 1}, {0.5f, 0.25f, 0.5f}};
    EXPECT_TRUE(bezier_collision_free(device, scene, quintic, 0.1f));
    EXPECT_FALSE(bezier_collision_free(device, scene, above, -0.1));
    EXPECT_FALSE(bezier_collision_free(device, scene, above, 0.1, -0.01));
}
