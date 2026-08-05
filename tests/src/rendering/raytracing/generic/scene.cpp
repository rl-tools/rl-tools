#include <rl_tools/operations/cpu.h>
// compiled twice: pinned to the generic backend (default), and against the build's active
// backend via the mux — the same backend-agnostic assertions are the cross-backend parity suite
#ifdef RL_TOOLS_RENDERING_RAYTRACING_SCENE_TEST_ACTIVE_BACKEND
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#ifndef RL_TOOLS_SCENE_SUITE
#define RL_TOOLS_SCENE_SUITE RENDERING_RAYTRACING_GENERIC_SCENE
#endif

#include <gtest/gtest.h>

#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;

namespace {
    using SPEC = rlt::rendering::raytracing::Specification<T, TI, 16, 16, 1, 4, rlt::rendering::raytracing::Low>;
    using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

    rlt::rendering::raytracing::Mesh make_cube(T center_x, T half_extent){
        rlt::rendering::raytracing::Mesh mesh;
        mesh.color[0] = 0.25f;
        mesh.color[1] = 0.5f;
        mesh.color[2] = 0.75f;
        for(TI vertex_i = 0; vertex_i < (TI)rlt::rendering::raytracing::constants::NUM_VERTICES; vertex_i++){
            mesh.vertices.push_back(center_x + rlt::rendering::raytracing::constants::default_vertices[vertex_i][0] * half_extent);
            mesh.vertices.push_back(rlt::rendering::raytracing::constants::default_vertices[vertex_i][1] * half_extent);
            mesh.vertices.push_back(rlt::rendering::raytracing::constants::default_vertices[vertex_i][2] * half_extent);
        }
        for(TI triangle_i = 0; triangle_i < (TI)rlt::rendering::raytracing::constants::NUM_INDICES; triangle_i++){
            mesh.indices.push_back(rlt::rendering::raytracing::constants::default_indices[triangle_i][0]);
            mesh.indices.push_back(rlt::rendering::raytracing::constants::default_indices[triangle_i][1]);
            mesh.indices.push_back(rlt::rendering::raytracing::constants::default_indices[triangle_i][2]);
        }
        return mesh;
    }

    template <typename RENDERER_SPEC>
    void set_test_camera(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC>& renderer, const T position[3], const T look_at[3]){
        const T up[3] = {0, 0, 1};
        constexpr T aspect = (T)RENDERER_SPEC::CAM_WIDTH / (T)RENDERER_SPEC::CAM_HEIGHT;
        rlt::set(device, renderer.cameras, rlt::make_camera_data(position, look_at, up, RENDERER_SPEC::COS_FOVY, aspect), (TI)0);
        rlt::set_cameras(device, renderer, renderer.cameras);
    }

    template <typename RENDERER_SPEC>
    void render_pixels(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC>& renderer, const T position[3], const T look_at[3], std::vector<uint32_t>& pixels){
        set_test_camera(device, renderer, position, look_at);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
        pixels.assign(rlt::data(renderer.frame_buffer), rlt::data(renderer.frame_buffer) + RENDERER_SPEC::CAM_PIXELS);
    }

    void render_frame(DEVICE& device, Renderer& renderer, std::vector<uint32_t>& pixels){
        const T position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        render_pixels(device, renderer, position, look_at, pixels);
    }

    // 1x1: the single pixel's ray is exactly the optical axis (dir_00 + 0.5 du + 0.5 dv)
    using DEPTH_SPEC = rlt::rendering::raytracing::Specification<T, TI, 1, 1, 1, 4, rlt::rendering::raytracing::Low, false, 1, false, 1, rlt::rendering::raytracing::OutputMode::DEPTH>;
    using PBR_SPEC = rlt::rendering::raytracing::Specification<T, TI, 32, 32, 1, 4, rlt::rendering::raytracing::VeryHigh>;

    template <typename RENDERER_SPEC>
    float render_center_depth(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC>& renderer){
        const T position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        set_test_camera(device, renderer, position, look_at);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
        const float* depth = rlt::data(renderer.depth_buffer);
        return depth[(RENDERER_SPEC::CAM_HEIGHT / 2) * RENDERER_SPEC::CAM_WIDTH + RENDERER_SPEC::CAM_WIDTH / 2];
    }
}

TEST(RL_TOOLS_SCENE_SUITE, INJECTION){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, make_cube(0, 1));
    ASSERT_EQ(scene.objects.size(), (size_t)1);
    ASSERT_EQ(scene.instances.size(), (size_t)1);

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene);

    std::vector<uint32_t> pixels;
    render_frame(device, renderer, pixels);

    const uint32_t center = pixels[SPEC::CAM_PIXELS / 2 + SPEC::CAM_WIDTH / 2];
    const uint32_t corner = pixels[0];
    EXPECT_NE(center, corner); // cube in the center, miss color at the edge
    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, REINIT_IDENTICAL){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, make_cube(0, 1));
    rlt::add(device, scene, make_cube(2.5f, 0.5f));

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    rlt::init(device, renderer, scene);
    std::vector<uint32_t> first;
    render_frame(device, renderer, first);

    rlt::init(device, renderer, scene);
    std::vector<uint32_t> second;
    render_frame(device, renderer, second);

    ASSERT_EQ(first, second);
    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, REINIT_SCENE_SWAP){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene_a;
    rlt::add(device, scene_a, make_cube(0, 1));
    rlt::rendering::raytracing::Scene scene_b;
    rlt::add(device, scene_b, make_cube(100, 1)); // out of view: renders as all-miss

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    rlt::init(device, renderer, scene_a);
    std::vector<uint32_t> first_a;
    render_frame(device, renderer, first_a);

    rlt::init(device, renderer, scene_b);
    std::vector<uint32_t> swapped;
    render_frame(device, renderer, swapped);
    EXPECT_NE(first_a, swapped);

    rlt::init(device, renderer, scene_a);
    std::vector<uint32_t> second_a;
    render_frame(device, renderer, second_a);
    ASSERT_EQ(first_a, second_a);

    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, INSTANCE_DEPTH_ANALYTIC){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Object cube;
    cube.meshes.push_back(make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<DEPTH_SPEC> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    {
        // translated +1 in x: front face moves from x=-1 to x=0, camera at x=-5
        rlt::rendering::raytracing::Scene scene;
        float transform[12];
        const float position[3] = {1, 0, 0};
        const float orientation_wxyz[4] = {1, 0, 0, 0};
        rlt::make_transform(position, orientation_wxyz, transform);
        rlt::add(device, scene, cube, transform);
        rlt::init(device, renderer, scene);
        EXPECT_NEAR(render_center_depth(device, renderer), 5.0f, 1e-4f);
    }
    {
        // rotated 30 degrees about z: the front edge crosses y=0 at x=-2/sqrt(3)
        rlt::rendering::raytracing::Scene scene;
        float transform[12];
        const float position[3] = {0, 0, 0};
        const float half_angle = 30.0f * 3.14159265f / 180.0f / 2.0f;
        const float orientation_wxyz[4] = {std::cos(half_angle), 0, 0, std::sin(half_angle)};
        rlt::make_transform(position, orientation_wxyz, transform);
        rlt::add(device, scene, cube, transform);
        rlt::init(device, renderer, scene);
        EXPECT_NEAR(render_center_depth(device, renderer), 5.0f - 2.0f / std::sqrt(3.0f), 1e-3f);
    }
    {
        // uniformly scaled by 0.5: front face at x=-0.5
        rlt::rendering::raytracing::Scene scene;
        const float transform[12] = {0.5f,0,0,0, 0,0.5f,0,0, 0,0,0.5f,0};
        rlt::add(device, scene, cube, transform);
        rlt::init(device, renderer, scene);
        EXPECT_NEAR(render_center_depth(device, renderer), 4.5f, 1e-4f);
    }

    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, TWO_INSTANCES_OF_ONE_OBJECT){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Object cube;
    cube.meshes.push_back(make_cube(0, 1));

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    rlt::rendering::raytracing::Scene single;
    rlt::add(device, single, cube);
    rlt::init(device, renderer, single);
    std::vector<uint32_t> frame_single;
    render_frame(device, renderer, frame_single);

    rlt::rendering::raytracing::Scene pair = single;
    float transform[12];
    const float position[3] = {0, 3, 0};
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position, orientation_wxyz, transform);
    rlt::add(device, pair, cube, transform);
    ASSERT_EQ(pair.objects.size(), (size_t)2);
    ASSERT_EQ(pair.instances.size(), (size_t)2);
    rlt::init(device, renderer, pair);
    std::vector<uint32_t> frame_pair;
    render_frame(device, renderer, frame_pair);

    EXPECT_NE(frame_single, frame_pair); // the second instance is visible somewhere
    const size_t center = (SPEC::CAM_HEIGHT / 2) * SPEC::CAM_WIDTH + SPEC::CAM_WIDTH / 2;
    EXPECT_EQ(frame_single[center], frame_pair[center]); // and does not affect the first

    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, BAKED_VS_INSTANCED_ORACLE){
    DEVICE device;
    rlt::init(device);

    float transform[12];
    const float position[3] = {0.3f, 0.4f, 0.2f};
    const float half_angle = 30.0f * 3.14159265f / 180.0f / 2.0f;
    const float orientation_wxyz[4] = {std::cos(half_angle), 0, 0, std::sin(half_angle)};
    rlt::make_transform(position, orientation_wxyz, transform);

    const rlt::rendering::raytracing::Mesh cube_mesh = make_cube(0, 1);
    rlt::rendering::raytracing::Mesh baked_mesh = cube_mesh;
    for(size_t vertex_i = 0; vertex_i + 2 < baked_mesh.vertices.size(); vertex_i += 3){
        const float local[3] = {baked_mesh.vertices[vertex_i], baked_mesh.vertices[vertex_i + 1], baked_mesh.vertices[vertex_i + 2]};
        float world[3];
        rlt::rendering::raytracing::detail::transform_point(transform, local, world);
        baked_mesh.vertices[vertex_i] = world[0];
        baked_mesh.vertices[vertex_i + 1] = world[1];
        baked_mesh.vertices[vertex_i + 2] = world[2];
    }

    rlt::rendering::raytracing::Object cube;
    cube.meshes.push_back(cube_mesh);

    auto compare_scenes = [&](auto& renderer){
        rlt::rendering::raytracing::Scene baked;
        rlt::add(device, baked, baked_mesh);
        baked.lights.push_back({0, {0,0,0}, {-0.57735f, -0.57735f, 0.57735f}, {0.8f, 0.8f, 0.8f}, 0,0,0, 0,0});
        rlt::init(device, renderer, baked);
        std::vector<uint32_t> frame_baked;
        const T camera_position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        render_pixels(device, renderer, camera_position, look_at, frame_baked);

        rlt::rendering::raytracing::Scene instanced;
        rlt::add(device, instanced, cube, transform);
        instanced.lights = baked.lights;
        rlt::init(device, renderer, instanced);
        std::vector<uint32_t> frame_instanced;
        render_pixels(device, renderer, camera_position, look_at, frame_instanced);

        // different float paths (transform in traversal vs baked vertices): tight but not exact
        size_t mismatched = 0;
        int max_channel_diff = 0;
        for(size_t pixel_i = 0; pixel_i < frame_baked.size(); pixel_i++){
            for(int channel = 0; channel < 3; channel++){
                const int a = (int)((frame_baked[pixel_i] >> (8 * channel)) & 0xFF);
                const int b = (int)((frame_instanced[pixel_i] >> (8 * channel)) & 0xFF);
                const int diff = a > b ? a - b : b - a;
                if(diff > max_channel_diff) max_channel_diff = diff;
                if(diff > 2){ mismatched++; break; }
            }
        }
        EXPECT_LE(mismatched, frame_baked.size() / 100); // silhouette pixels may land on different surfaces
        return max_channel_diff;
    };

    Renderer renderer_low;
    rlt::malloc(device, renderer_low);
    rlt::generate_probe_directions(device, renderer_low);
    compare_scenes(renderer_low);
    rlt::free(device, renderer_low);

    rlt::rendering::raytracing::Renderer<PBR_SPEC> renderer_pbr;
    rlt::malloc(device, renderer_pbr);
    rlt::generate_probe_directions(device, renderer_pbr);
    compare_scenes(renderer_pbr);
    rlt::free(device, renderer_pbr);
}

TEST(RL_TOOLS_SCENE_SUITE, OBJECT_LIGHT_FOLLOWS_INSTANCE){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Mesh floor;
    floor.color[0] = 1; floor.color[1] = 1; floor.color[2] = 1;
    const float floor_vertices[4 * 3] = {-10, -10, 0,  10, -10, 0,  10, 10, 0,  -10, 10, 0};
    const int floor_indices[2 * 3] = {0, 1, 2,  0, 2, 3};
    floor.vertices.assign(floor_vertices, floor_vertices + 12);
    floor.indices.assign(floor_indices, floor_indices + 6);

    rlt::rendering::raytracing::Object lamp;
    lamp.meshes.push_back(make_cube(0, 0.2f));
    lamp.lights.push_back({1, {0, 0, 1.5f}, {0, 0, -1}, {5, 5, 5}, 1, 0, 1, 0, 0}); // point light above the lamp body

    rlt::rendering::raytracing::Renderer<PBR_SPEC> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    auto side_difference = [&](float lamp_y){
        rlt::rendering::raytracing::Scene scene;
        rlt::add(device, scene, floor);
        float transform[12];
        const float position[3] = {0, lamp_y, 0.2f};
        const float orientation_wxyz[4] = {1, 0, 0, 0};
        rlt::make_transform(position, orientation_wxyz, transform);
        rlt::add(device, scene, lamp, transform);
        rlt::init(device, renderer, scene);

        const T camera_position[3] = {-6, 0, 4};
        const T look_at[3] = {0, 0, 0};
        std::vector<uint32_t> pixels;
        render_pixels(device, renderer, camera_position, look_at, pixels);

        long left = 0, right = 0;
        for(TI y = 0; y < PBR_SPEC::CAM_HEIGHT; y++){
            for(TI x = 0; x < PBR_SPEC::CAM_WIDTH; x++){
                const long red = (long)(pixels[y * PBR_SPEC::CAM_WIDTH + x] & 0xFF);
                if(x < PBR_SPEC::CAM_WIDTH / 2) left += red; else right += red;
            }
        }
        return left - right;
    };

    const long difference_positive_y = side_difference(2.5f);
    const long difference_negative_y = side_difference(-2.5f);
    EXPECT_GT(std::abs(difference_positive_y), 0);
    EXPECT_LT(difference_positive_y * difference_negative_y, 0); // the bright side follows the lamp

    rlt::free(device, renderer);
}
