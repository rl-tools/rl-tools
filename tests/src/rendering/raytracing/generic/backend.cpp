#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_generic.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;
namespace generic = rl_tools::rendering::raytracing::backends::generic;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;

namespace {
    // unit cube (matches rendering::raytracing::constants default cube)
    constexpr T CUBE_VERTICES[8 * 3] = {
        -1, -1, -1,
        +1, -1, -1,
        -1, +1, -1,
        +1, +1, -1,
        -1, -1, +1,
        +1, -1, +1,
        -1, +1, +1,
        +1, +1, +1
    };
    constexpr int CUBE_INDICES[12 * 3] = {
        0, 1, 3,  2, 3, 0,
        5, 7, 6,  5, 6, 4,
        0, 4, 5,  0, 5, 1,
        2, 3, 7,  2, 7, 6,
        1, 5, 7,  1, 7, 3,
        4, 0, 2,  4, 2, 6
    };

    struct ShadingLow{
        static constexpr bool LOAD_TEXTURES = false;
        static constexpr bool NORMAL_SHADING = false;
        static constexpr bool METALLIC_REFLECTIONS = false;
        static constexpr bool SRGB_OUTPUT = false;
        static constexpr bool CHECKER_BACKGROUND = false;
        static constexpr bool PBR_SHADING = false;
        static constexpr bool PUNCTUAL_LIGHT_SHADOWS = false;
    };
    struct ShadingLowChecker: ShadingLow{
        static constexpr bool CHECKER_BACKGROUND = true;
    };

    template <typename T_SHADING, TI T_NUM_PROBES, bool T_ENABLE_ANTI_ALIASING = false, TI T_ANTI_ALIASING_GRID_SIZE = 1>
    struct Spec{
        using T = ::T;
        using TI = ::TI;
        using SHADING = T_SHADING;
        static constexpr TI CAM_WIDTH = 1;
        static constexpr TI CAM_HEIGHT = 1;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = T_NUM_PROBES;
        static constexpr TI GRID_COLS = 1;
        static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
        static constexpr bool ENABLE_MOTION_BLUR = false;
        static constexpr TI MOTION_BLUR_SAMPLES = 1;
        static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
    };

    struct CubeScene{
        generic::MeshView<T, TI> mesh;
        generic::SceneView<T, TI> scene;
        TI triangle_mesh[12];
        TI triangle_local[12];
        generic::BVHNode<T, TI> nodes[24];
        TI primitives[12];
        TI temp_primitives[12];
        T bounds_min[12 * 3];
        T bounds_max[12 * 3];
        T centroids[12 * 3];
        generic::ObjectView<T, TI> object;
        generic::InstanceView<T, TI> instance;
        generic::BVHNode<T, TI> tlas_node;
        TI tlas_primitive;

        CubeScene(DEVICE& device){
            mesh.indices = CUBE_INDICES;
            mesh.vertices = CUBE_VERTICES;
            mesh.color[0] = 0.25f;
            mesh.color[1] = 0.5f;
            mesh.color[2] = 0.75f;
            scene.meshes = &mesh;
            scene.num_meshes = 1;
            for(TI triangle = 0; triangle < 12; triangle++){
                triangle_mesh[triangle] = 0;
                triangle_local[triangle] = triangle;
            }
            scene.triangle_mesh = triangle_mesh;
            scene.triangle_local = triangle_local;
            scene.num_triangles = 12;
            scene.max_depth = 100;
            scene.max_dist = 100;
            scene.miss_color_0[0] = 0.8f; scene.miss_color_0[1] = 0.0f; scene.miss_color_0[2] = 0.0f;
            scene.miss_color_1[0] = 0.8f; scene.miss_color_1[1] = 0.8f; scene.miss_color_1[2] = 0.8f;

            for(TI triangle = 0; triangle < 12; triangle++){
                primitives[triangle] = triangle;
                T triangle_min[3] = {1e30f, 1e30f, 1e30f};
                T triangle_max[3] = {-1e30f, -1e30f, -1e30f};
                generic::expand_triangle_bounds(scene, triangle, triangle_min, triangle_max);
                for(int axis = 0; axis < 3; axis++){
                    bounds_min[3 * triangle + axis] = triangle_min[axis];
                    bounds_max[3 * triangle + axis] = triangle_max[axis];
                    centroids[3 * triangle + axis] = (triangle_min[axis] + triangle_max[axis]) * 0.5f;
                }
            }
            object.nodes = nodes;
            object.primitives = primitives;
            object.num_nodes = generic::build_bvh_nodes(nodes, primitives, temp_primitives, bounds_min, bounds_max, centroids, (TI)12);
            scene.objects = &object;
            scene.num_objects = 1;

            const T identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
            instance.object = 0;
            instance.identity = true;
            for(int element = 0; element < 12; element++){
                instance.object_to_world[element] = identity[element];
                instance.world_to_object[element] = identity[element];
            }
            scene.instances = &instance;
            scene.num_instances = 1;

            tlas_node = nodes[0];
            tlas_node.left_or_first = 0;
            tlas_node.count = 1;
            tlas_primitive = 0;
            scene.tlas_nodes = &tlas_node;
            scene.tlas_primitives = &tlas_primitive;
            scene.num_tlas_nodes = 1;
        }
    };

    rlt::rendering::raytracing::Camera<T> forward_camera(T position_x){
        // center ray = dir_00 + 0.5*dir_du + 0.5*dir_dv = (1, 0, 0)
        return {{position_x, 0, 0}, {1, 0.5f, 0.5f}, {0, -1, 0}, {0, 0, -1}};
    }
}

TEST(RENDERING_RAYTRACING_GENERIC, BVH_BUILD){
    DEVICE device;
    CubeScene cube(device);
    ASSERT_GE(cube.object.num_nodes, (TI)1);
    ASSERT_LE(cube.object.num_nodes, (TI)23);
    // every triangle appears exactly once in the leaf permutation
    bool seen[12] = {};
    for(TI i = 0; i < 12; i++){
        ASSERT_LT(cube.object.primitives[i], (TI)12);
        seen[cube.object.primitives[i]] = true;
    }
    for(TI i = 0; i < 12; i++){
        ASSERT_TRUE(seen[i]);
    }
}

TEST(RENDERING_RAYTRACING_GENERIC, DEPTH_ANALYTIC){
    DEVICE device;
    CubeScene cube(device);
    using SPEC = Spec<ShadingLow, 1>;
    const auto camera = forward_camera(-5);
    float depth = -1;
    cube.scene.cameras_close = &camera;
    cube.scene.cameras_open = &camera;
    cube.scene.depth_buffer = &depth;
    generic::render_frame<DEVICE, SPEC, generic::OutputDepth>(device, cube.scene);
    EXPECT_FLOAT_EQ(depth, 4.0f); // camera at x=-5, face at x=-1
}

TEST(RENDERING_RAYTRACING_GENERIC, COLLISION_ANALYTIC){
    DEVICE device;
    CubeScene cube(device);
    using SPEC = Spec<ShadingLow, 4>;
    // camera at the origin (inside the cube): forward probe and axis probes all hit at distance 1
    const rlt::rendering::raytracing::Camera<T> camera = forward_camera(0);
    const T probe_directions[4 * 3] = {
        1, 0, 0, // probe 0 is overridden by the camera forward direction
        0, 1, 0,
        0, 0, 1,
        -1, 0, 0
    };
    rlt::rendering::raytracing::CollisionResult results[4];
    cube.scene.cameras_close = &camera;
    cube.scene.cameras_open = &camera;
    cube.scene.probe_directions = probe_directions;
    cube.scene.collision_results = results;
    generic::render_collision<DEVICE, SPEC>(device, cube.scene);
    for(TI probe_i = 0; probe_i < 4; probe_i++){
        EXPECT_EQ(results[probe_i].hit, 1) << "probe " << probe_i;
        EXPECT_FLOAT_EQ(results[probe_i].distance, 1.0f) << "probe " << probe_i;
    }

    // camera outside, looking away from the cube: forward probe misses, probe towards the cube hits at 4
    const rlt::rendering::raytracing::Camera<T> camera_away = {{-5, 0, 0}, {-1, 0.5f, 0.5f}, {0, -1, 0}, {0, 0, -1}};
    const T probe_directions_away[4 * 3] = {
        -1, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    cube.scene.cameras_close = &camera_away;
    cube.scene.probe_directions = probe_directions_away;
    generic::render_collision<DEVICE, SPEC>(device, cube.scene);
    EXPECT_EQ(results[0].hit, 0);
    EXPECT_FLOAT_EQ(results[0].distance, 100.0f);
    EXPECT_EQ(results[1].hit, 1);
    EXPECT_FLOAT_EQ(results[1].distance, 4.0f);
    EXPECT_EQ(results[2].hit, 0);
    EXPECT_EQ(results[3].hit, 0);
}

TEST(RENDERING_RAYTRACING_GENERIC, RGB_FLAT_AND_MISS){
    DEVICE device;
    CubeScene cube(device);
    const auto camera = forward_camera(-5);
    unsigned int pixel = 0;
    cube.scene.cameras_close = &camera;
    cube.scene.cameras_open = &camera;
    cube.scene.frame_buffer = &pixel;
    {
        using SPEC = Spec<ShadingLow, 1>;
        generic::render_frame<DEVICE, SPEC, generic::OutputRGB>(device, cube.scene);
        // flat shading: exactly the mesh color, linear packing: (0.25, 0.5, 0.75) -> (64, 128, 192)
        EXPECT_EQ(pixel, 0xFFC08040u);
    }
    {
        // anti-aliasing over a flat surface must not change the result
        using SPEC = Spec<ShadingLow, 1, true, 2>;
        pixel = 0;
        generic::render_frame<DEVICE, SPEC, generic::OutputRGB>(device, cube.scene);
        EXPECT_EQ(pixel, 0xFFC08040u);
    }
    {
        // camera looking away: miss; checker cell (0, 0) selects miss_color_0 = (0.8, 0, 0) -> r = 204
        using SPEC = Spec<ShadingLowChecker, 1>;
        const rlt::rendering::raytracing::Camera<T> camera_away = {{-5, 0, 0}, {-1, 0.5f, 0.5f}, {0, -1, 0}, {0, 0, -1}};
        cube.scene.cameras_close = &camera_away;
        cube.scene.cameras_open = &camera_away;
        pixel = 0;
        generic::render_frame<DEVICE, SPEC, generic::OutputRGB>(device, cube.scene);
        EXPECT_EQ(pixel, 0xFF0000CCu);
    }
}
