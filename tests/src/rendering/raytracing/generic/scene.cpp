#include <rl_tools/operations/cpu.h>
// compiled twice: pinned to the generic backend, and against the build's active
// backend via the mux — the same backend-agnostic assertions are the cross-backend parity suite
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#ifdef RL_TOOLS_RENDERING_RAYTRACING_SCENE_TEST_ACTIVE_BACKEND
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#endif

#ifndef RL_TOOLS_SCENE_SUITE
#define RL_TOOLS_SCENE_SUITE RENDERING_RAYTRACING_GENERIC_SCENE
#endif

#include "../../../utils/utils.h"
#ifdef RL_TOOLS_TEST_DATA_PATH
#define RL_TOOLS_SCENE_TEST_DATA_PATH RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)
#else
#define RL_TOOLS_SCENE_TEST_DATA_PATH "tests/data"
#endif

#include <gtest/gtest.h>

#include <vector>
#include <chrono>
#include <filesystem>
#include <string>

namespace rlt = rl_tools;

#ifdef RL_TOOLS_RENDERING_RAYTRACING_SCENE_TEST_ACTIVE_BACKEND
using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using rlt::rendering::raytracing::OverlayIndex;

namespace {
    struct SPEC_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = 16, CAM_HEIGHT = 16, NUM_CAMERAS = 1, NUM_PROBES = 4;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using SPEC = rlt::rendering::raytracing::Specification<SPEC_CONFIG>;
    using Renderer = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;

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
    void write_cameras(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const rlt::rendering::raytracing::Camera<T>* staging, TI count){
        rlt::copy_to_renderer(device, renderer, staging, rlt::data(rlt::cameras(device, renderer)), count);
    }

    template <typename RENDERER_SPEC, typename ELEMENT, typename TENSOR>
    void read_output(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const TENSOR& tensor, ELEMENT* out, size_t count){
        rlt::copy_from_renderer(device, renderer, rlt::data(tensor), out, count);
    }

    template <typename RENDERER_SPEC>
    void set_test_camera(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const T position[3], const T look_at[3]){
        const T up[3] = {0, 0, 1};
        constexpr T aspect = (T)RENDERER_SPEC::CAM_WIDTH / (T)RENDERER_SPEC::CAM_HEIGHT;
        const auto camera = rlt::make_camera_data(position, look_at, up, RENDERER_SPEC::COS_FOVY, aspect);
        write_cameras(device, renderer, &camera, 1);
    }

    template <typename RENDERER_SPEC>
    void render_pixels(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const T position[3], const T look_at[3], std::vector<uint32_t>& pixels){
        set_test_camera(device, renderer, position, look_at);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        pixels.resize(RENDERER_SPEC::CAM_PIXELS);
        read_output(device, renderer, rlt::frame_buffer(device, renderer), pixels.data(), RENDERER_SPEC::CAM_PIXELS);
    }

    void render_frame(DEVICE& device, Renderer& renderer, std::vector<uint32_t>& pixels){
        const T position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        render_pixels(device, renderer, position, look_at, pixels);
    }

    // 1x1: the single pixel's ray is exactly the optical axis (dir_00 + 0.5 du + 0.5 dv)
    struct DEPTH_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = 1, CAM_HEIGHT = 1, NUM_CAMERAS = 1, NUM_PROBES = 4;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr bool OUTPUT_RGB = false;
        static constexpr bool OUTPUT_DEPTH = true;
    };
    using DEPTH_SPEC = rlt::rendering::raytracing::Specification<DEPTH_CONFIG>;
    struct PBR_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 1, NUM_PROBES = 4;
        using SHADING = rlt::rendering::raytracing::VeryHigh;
    };
    using PBR_SPEC = rlt::rendering::raytracing::Specification<PBR_CONFIG>;

    template <typename RENDERER_SPEC>
    float render_center_depth(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer){
        const T position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        set_test_camera(device, renderer, position, look_at);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        std::vector<float> depth(RENDERER_SPEC::CAM_PIXELS);
        read_output(device, renderer, rlt::depth_buffer(device, renderer), depth.data(), RENDERER_SPEC::CAM_PIXELS);
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

    rlt::rendering::raytracing::Renderer<DEPTH_SPEC, BACKEND> renderer;
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

    rlt::rendering::raytracing::Renderer<PBR_SPEC, BACKEND> renderer_pbr;
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

    rlt::rendering::raytracing::Renderer<PBR_SPEC, BACKEND> renderer;
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

struct SPLIT_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 128, CAM_HEIGHT = 128, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::VeryHigh;
};
using SPLIT_SPEC = rlt::rendering::raytracing::Specification<SPLIT_CONFIG>;

struct SEGMENTATION_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_RGB = false;
    static constexpr bool OUTPUT_SEGMENTATION = true;
};
using SEGMENTATION_SPEC = rlt::rendering::raytracing::Specification<SEGMENTATION_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, SPLIT_VS_WELDED){
    DEVICE device;
    rlt::init(device);

    const char* scene_path = RL_TOOLS_SCENE_TEST_DATA_PATH "/ProcTHOR-Train-1.glb";
    if(FILE* file = std::fopen(scene_path, "rb")){
        std::fclose(file);
    }
    else{
        GTEST_SKIP() << "scene file not found (run from the repo root): " << scene_path;
    }

    rlt::rendering::raytracing::Renderer<SPLIT_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    const T camera_position[3] = {-2.849f, -8.234f, 1.548f};
    const T look_at[3] = {-2.0f, -8.0f, 1.3f};

    rlt::rendering::raytracing::Scene welded;
    ASSERT_TRUE((rlt::load<typename SPLIT_SPEC::SHADING, SPLIT_SPEC::HAS_RGB>(device, welded, scene_path)));
    rlt::init(device, renderer, welded);
    std::vector<uint32_t> frame_welded;
    render_pixels(device, renderer, camera_position, look_at, frame_welded);

    rlt::rendering::raytracing::ObjectAssembly assembly;
    ASSERT_TRUE((rlt::load<typename SPLIT_SPEC::SHADING, SPLIT_SPEC::HAS_RGB>(device, assembly, scene_path)));
    EXPECT_EQ(assembly.parts.size(), (size_t)192);
    rlt::rendering::raytracing::Scene split;
    const auto placement = rlt::add(device, split, assembly);
    EXPECT_EQ(placement.first_instance, (size_t)0);
    EXPECT_EQ(placement.num_instances, (size_t)192);
    rlt::init(device, renderer, split);
    std::vector<uint32_t> frame_split;
    render_pixels(device, renderer, camera_position, look_at, frame_split);

    // same geometry through different frames (root-local + instance transform vs baked world):
    // float paths differ, so tight tolerance instead of equality
    double absolute_difference_sum = 0;
    size_t mismatched = 0;
    for(size_t pixel_i = 0; pixel_i < frame_welded.size(); pixel_i++){
        for(int channel = 0; channel < 3; channel++){
            const int a = (int)((frame_welded[pixel_i] >> (8 * channel)) & 0xFF);
            const int b = (int)((frame_split[pixel_i] >> (8 * channel)) & 0xFF);
            const int diff = a > b ? a - b : b - a;
            absolute_difference_sum += diff;
            if(diff > 2){ mismatched++; break; }
        }
    }
    const double mean_absolute_difference = absolute_difference_sum / (double)(frame_welded.size() * 3);
    EXPECT_LE(mean_absolute_difference, 1.0);
    EXPECT_LE(mismatched, frame_welded.size() / 50);

    rlt::free(device, renderer);
}

namespace {

    template <typename RENDERER_SPEC>
    std::vector<uint32_t> render_segmentation_pixels(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const T position[3], const T look_at[3]){
        set_test_camera(device, renderer, position, look_at);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        std::vector<uint32_t> segmentation(RENDERER_SPEC::CAM_PIXELS);
        read_output(device, renderer, rlt::segmentation_buffer(device, renderer), segmentation.data(), RENDERER_SPEC::CAM_PIXELS);
        return segmentation;
    }
}

TEST(RL_TOOLS_SCENE_SUITE, SEGMENTATION_ANALYTIC){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Object cube;
    cube.meshes.push_back(make_cube(0, 1));

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, cube); // instance 0 at the origin
    float transform[12];
    const float position_offset[3] = {0, 3, 0};
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position_offset, orientation_wxyz, transform);
    rlt::add(device, scene, cube, transform); // instance 1 to the left

    rlt::rendering::raytracing::Renderer<SEGMENTATION_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene);

    const T camera_position[3] = {-5, 0, 0};
    const T look_at[3] = {0, 0, 0};
    const auto segmentation = render_segmentation_pixels(device, renderer, camera_position, look_at);

    const size_t center = (SEGMENTATION_SPEC::CAM_HEIGHT / 2) * SEGMENTATION_SPEC::CAM_WIDTH + SEGMENTATION_SPEC::CAM_WIDTH / 2;
    EXPECT_EQ(segmentation[center], 0u); // instance 0 covers the view center
    EXPECT_EQ(segmentation[0], 0xFFFFFFFFu); // background misses
    size_t first_count = 0, second_count = 0;
    for(uint32_t id : segmentation){
        first_count += id == 0u;
        second_count += id == 1u;
    }
    EXPECT_GT(first_count, (size_t)0);
    EXPECT_GT(second_count, (size_t)0); // the offset instance is visible with its own id

    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, ASSEMBLY_COMPOSE){
    DEVICE device;
    rlt::init(device);

    // two-part assembly: body at the assembly origin, a smaller part offset to the left
    rlt::rendering::raytracing::ObjectAssembly assembly;
    {
        rlt::rendering::raytracing::Object body;
        body.meshes.push_back(make_cube(0, 1));
        body.name = "body";
        assembly.objects.push_back(body);
        rlt::rendering::raytracing::ObjectAssembly::Part part{0, {1,0,0,0, 0,1,0,0, 0,0,1,0}};
        assembly.parts.push_back(part);
    }
    {
        rlt::rendering::raytracing::Object attachment;
        attachment.meshes.push_back(make_cube(0, 0.5f));
        attachment.name = "attachment";
        assembly.objects.push_back(attachment);
        rlt::rendering::raytracing::ObjectAssembly::Part part{1, {1,0,0,0, 0,1,0,2.0f, 0,0,1,0}};
        assembly.parts.push_back(part);
    }

    rlt::rendering::raytracing::Scene scene;
    const auto placement_a = rlt::add(device, scene, assembly);
    float pose[12];
    const float position_b[3] = {2, -4, 0};
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position_b, orientation_wxyz, pose);
    const auto placement_b = rlt::add(device, scene, assembly, pose);

    EXPECT_EQ(placement_a.first_instance, (size_t)0);
    EXPECT_EQ(placement_a.num_instances, (size_t)2);
    EXPECT_EQ(placement_b.first_instance, (size_t)2);
    EXPECT_EQ(placement_b.num_instances, (size_t)2);

    rlt::rendering::raytracing::Renderer<SEGMENTATION_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene);

    const T camera_position[3] = {-8, 0, 0};
    const T look_at[3] = {0, 0, 0};
    const auto segmentation = render_segmentation_pixels(device, renderer, camera_position, look_at);

    size_t id_counts[4] = {};
    for(uint32_t id : segmentation){
        if(id < 4) id_counts[id]++;
    }
    for(int instance_i = 0; instance_i < 4; instance_i++){
        EXPECT_GT(id_counts[instance_i], (size_t)0) << "instance " << instance_i << " not visible";
    }

    rlt::free(device, renderer);
}

TEST(RL_TOOLS_SCENE_SUITE, ASSEMBLY_STATIC_ARTICULATION){
    DEVICE device;
    rlt::init(device);

    // one flat "propeller" part offset from the body: articulating it at add-time (composing an
    // extra local rotation) is exactly the operation a per-step update will perform in Phase 3
    rlt::rendering::raytracing::ObjectAssembly assembly;
    rlt::rendering::raytracing::Object propeller;
    {
        rlt::rendering::raytracing::Mesh blade = make_cube(0, 1);
        for(size_t vertex_i = 1; vertex_i < blade.vertices.size(); vertex_i += 3){
            blade.vertices[vertex_i] *= 0.1f; // thin in y: a blade along x
        }
        propeller.meshes.push_back(blade);
    }
    assembly.objects.push_back(propeller);
    assembly.parts.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}});

    rlt::rendering::raytracing::Renderer<DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);

    auto blade_center_depth = [&](float spin_radians){
        rlt::rendering::raytracing::ObjectAssembly articulated = assembly;
        const float half = spin_radians / 2.0f;
        float spin[12];
        const float origin[3] = {0, 0, 0};
        const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
        rlt::make_transform(origin, quaternion_wxyz, spin);
        float composed[12];
        rlt::rendering::raytracing::detail::compose_transforms(articulated.parts[0].transform, spin, composed);
        std::memcpy(articulated.parts[0].transform, composed, sizeof(composed));

        rlt::rendering::raytracing::Scene scene;
        rlt::add(device, scene, articulated);
        rlt::init(device, renderer, scene);
        return render_center_depth(device, renderer);
    };

    // blade along x: the axial ray from x=-5 hits the near face at x=-1 -> depth 4;
    // spun 90 degrees the blade lies along y and the near face sits at x=-0.1 -> depth 4.9
    EXPECT_NEAR(blade_center_depth(0.0f), 4.0f, 1e-4f);
    EXPECT_NEAR(blade_center_depth(3.14159265f / 2.0f), 4.9f, 1e-3f);

    rlt::free(device, renderer);
}

namespace {
    template <typename RENDERER_SPEC>
    void set_same_pose_cameras(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer, const T position[3], const T look_at[3]){
        const T up[3] = {0, 0, 1};
        constexpr T aspect = (T)RENDERER_SPEC::CAM_WIDTH / (T)RENDERER_SPEC::CAM_HEIGHT;
        std::vector<rlt::rendering::raytracing::Camera<T>> staging(RENDERER_SPEC::NUM_CAMERAS);
        for(TI camera_i = 0; camera_i < RENDERER_SPEC::NUM_CAMERAS; camera_i++){
            staging[camera_i] = rlt::make_camera_data(position, look_at, up, RENDERER_SPEC::COS_FOVY, aspect);
        }
        write_cameras(device, renderer, staging.data(), RENDERER_SPEC::NUM_CAMERAS);
    }

    rlt::rendering::raytracing::Scene make_far_anchor_scene(DEVICE& device){
        // out-of-view geometry: establishes bounds (camera_radius, max_dist) without appearing
        rlt::rendering::raytracing::Scene scene;
        rlt::add(device, scene, make_cube(100, 30));
        return scene;
    }
}

struct OVERLAY_SEG_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 2, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_RGB = false;
    static constexpr bool OUTPUT_SEGMENTATION = true;
    static constexpr TI NUM_OVERLAYS = 2, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 2;
};
using OVERLAY_SEG_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_SEG_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_ATTACHMENT_SCOPES){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, make_cube(0, 1)); // shared world: global instance id 0
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));
    pool.assemblies[cube_asset.index].objects[0].name = "overlay-cube";

    rlt::rendering::raytracing::Renderer<OVERLAY_SEG_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);

    EXPECT_TRUE(rlt::can_attach(device, renderer, (TI)0, OverlayIndex{0}));
    EXPECT_FALSE(rlt::can_attach(device, renderer, (TI)0, OverlayIndex{5})); // out of range
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0}); // overlay 0: camera 0 only (per-agent)
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0}); // idempotent: must not consume the second slot
    rlt::attach(device, renderer, (TI)0, OverlayIndex{1}); // overlay 1: both cameras (shared)
    rlt::attach(device, renderer, (TI)1, OverlayIndex{1});
    EXPECT_TRUE(rlt::can_attach(device, renderer, (TI)0, OverlayIndex{1})); // already attached: idempotent re-attach allowed

    float transform[12];
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    const float position_private[3] = {0, 3, 0};
    rlt::make_transform(position_private, orientation_wxyz, transform);
    const auto private_placement = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform); // global id 1 (base of overlay 0)
    const float position_shared[3] = {0, -3, 0};
    rlt::make_transform(position_shared, orientation_wxyz, transform);
    rlt::spawn(device, renderer, OverlayIndex{1}, cube_asset, transform); // global id 9 (base of overlay 1)
    rlt::update(device, renderer);

    const T camera_position[3] = {-8, 0, 0};
    const T look_at[3] = {0, 0, 0};
    set_same_pose_cameras(device, renderer, camera_position, look_at);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    std::vector<uint32_t> segmentation_staging((size_t)decltype(renderer)::SPEC::NUM_CAMERAS * decltype(renderer)::SPEC::CAM_PIXELS);
    read_output(device, renderer, rlt::segmentation_buffer(device, renderer), segmentation_staging.data(), segmentation_staging.size());
    const uint32_t* segmentation = segmentation_staging.data();

    size_t counts[2][10] = {};
    for(TI camera_i = 0; camera_i < 2; camera_i++){
        for(TI pixel_i = 0; pixel_i < OVERLAY_SEG_SPEC::CAM_PIXELS; pixel_i++){
            const uint32_t id = segmentation[camera_i * OVERLAY_SEG_SPEC::CAM_PIXELS + pixel_i];
            if(id < 10) counts[camera_i][id]++;
        }
    }
    EXPECT_GT(counts[0][0], (size_t)0); // shared world visible to both
    EXPECT_GT(counts[1][0], (size_t)0);
    EXPECT_GT(counts[0][1], (size_t)0); // private overlay: camera 0 only
    EXPECT_EQ(counts[1][1], (size_t)0);
    EXPECT_GT(counts[0][9], (size_t)0); // shared overlay: both
    EXPECT_GT(counts[1][9], (size_t)0);

    const auto* scene_object = rlt::segmentation_object(device, scene, pool, renderer, 0u);
    ASSERT_NE(scene_object, nullptr);
    EXPECT_EQ(scene_object, &scene.objects[0]);
    const uint32_t private_id = (uint32_t)(scene.instances.size() + private_placement.first_slot);
    const auto* overlay_object = rlt::segmentation_object(device, scene, pool, renderer, private_id);
    ASSERT_NE(overlay_object, nullptr);
    EXPECT_EQ(overlay_object->name, "overlay-cube");
    EXPECT_EQ(rlt::segmentation_object(device, scene, pool, renderer, 0xFFFFFFFFu), nullptr);

    rlt::detach(device, renderer, (TI)0, OverlayIndex{0});
    rlt::update(device, renderer);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    read_output(device, renderer, rlt::segmentation_buffer(device, renderer), segmentation_staging.data(), segmentation_staging.size());
    size_t detached_count = 0;
    for(TI pixel_i = 0; pixel_i < OVERLAY_SEG_SPEC::CAM_PIXELS; pixel_i++){
        detached_count += segmentation[pixel_i] == 1u;
    }
    EXPECT_EQ(detached_count, (size_t)0); // detached overlay no longer visible to camera 0

    rlt::free(device, renderer);
}

struct SEMANTIC_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_RGB = false;
    static constexpr bool OUTPUT_SEGMENTATION = true;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
    static constexpr bool SEMANTIC_SEGMENTATION = true;
};
using SEMANTIC_SPEC = rlt::rendering::raytracing::Specification<SEMANTIC_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, SEMANTIC_SEGMENTATION){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, make_cube(0, 1)); // instance id 0
    scene.objects[0].segmentation_class = 7;
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));
    pool.assemblies[cube_asset.index].objects[0].segmentation_class = 3;

    rlt::rendering::raytracing::Renderer<SEMANTIC_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    float transform[12];
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    const float position[3] = {0, 3, 0};
    rlt::make_transform(position, orientation_wxyz, transform);
    rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform); // instance id 1
    rlt::update(device, renderer);

    const T camera_position[3] = {-8, 0, 0};
    const T look_at[3] = {0, 0, 0};
    set_same_pose_cameras(device, renderer, camera_position, look_at);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    std::vector<uint32_t> segmentation_staging((size_t)decltype(renderer)::SPEC::NUM_CAMERAS * decltype(renderer)::SPEC::CAM_PIXELS);
    read_output(device, renderer, rlt::segmentation_buffer(device, renderer), segmentation_staging.data(), segmentation_staging.size());
    const uint32_t* segmentation = segmentation_staging.data();

    size_t scene_class_count = 0, overlay_class_count = 0, raw_instance_id_count = 0;
    for(TI pixel_i = 0; pixel_i < SEMANTIC_SPEC::CAM_PIXELS; pixel_i++){
        scene_class_count += segmentation[pixel_i] == 7u;
        overlay_class_count += segmentation[pixel_i] == 3u;
        raw_instance_id_count += segmentation[pixel_i] == 0u || segmentation[pixel_i] == 1u;
    }
    EXPECT_GT(scene_class_count, (size_t)0);   // scene cube reports its class, not id 0
    EXPECT_GT(overlay_class_count, (size_t)0); // overlay cube reports its class, not id 1
    EXPECT_EQ(raw_instance_id_count, (size_t)0);

    rlt::free(device, renderer);
}

struct OVERLAY_DEPTH_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 1, CAM_HEIGHT = 1, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_RGB = false;
    static constexpr bool OUTPUT_DEPTH = true;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
};
using OVERLAY_DEPTH_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_DEPTH_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_SPIN_DYNAMIC){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::ObjectAssembly blade_assembly;
    {
        rlt::rendering::raytracing::Object blade;
        rlt::rendering::raytracing::Mesh mesh = make_cube(0, 1);
        for(size_t vertex_i = 1; vertex_i < mesh.vertices.size(); vertex_i += 3){
            mesh.vertices[vertex_i] *= 0.1f; // thin in y: a blade along x
        }
        blade.meshes.push_back(mesh);
        blade_assembly.objects.push_back(blade);
        blade_assembly.parts.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}});
    }

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto blade_asset = rlt::add(device, pool, blade_assembly);

    rlt::rendering::raytracing::Renderer<OVERLAY_DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto blade = rlt::spawn(device, renderer, OverlayIndex{0}, blade_asset, identity);

    auto step_depth = [&](float spin_radians){
        const float half = spin_radians / 2.0f;
        float spin[12];
        const float origin[3] = {0, 0, 0};
        const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
        rlt::make_transform(origin, quaternion_wxyz, spin);
        rlt::set_transform(device, renderer, OverlayIndex{0}, blade, (TI)0, spin);
        rlt::update(device, renderer);
        return render_center_depth(device, renderer);
    };

    // blade along x: near face at x=-1 from camera x=-5 -> depth 4; spun 90 degrees -> x=-0.1 -> 4.9
    EXPECT_NEAR(step_depth(0.0f), 4.0f, 1e-4f);
    EXPECT_NEAR(step_depth(3.14159265f / 2.0f), 4.9f, 1e-3f);
    EXPECT_NEAR(step_depth(0.0f), 4.0f, 1e-4f); // back again: deterministic round trip
    const float depth_first = step_depth(0.7f);
    const float depth_second = step_depth(0.7f);
    EXPECT_EQ(depth_first, depth_second); // identical inputs -> bit-identical output

    // rigid move: one call re-derives every part as pose ∘ part-local, replacing the spin state
    float shifted[12];
    const float shifted_position[3] = {0.5f, 0, 0};
    const float identity_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(shifted_position, identity_wxyz, shifted);
    rlt::set_transform(device, renderer, OverlayIndex{0}, blade, shifted);
    rlt::update(device, renderer);
    EXPECT_NEAR(render_center_depth(device, renderer), 4.5f, 1e-4f); // near face x=-1 -> x=-0.5

    rlt::free(device, renderer);
}

// producer path: writes land in the transforms tensor directly (no host verb, no dirty flag),
// update() must pick them up regardless
TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_TRANSFORMS_TENSOR){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<OVERLAY_DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto cube = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, identity);
    rlt::update(device, renderer);
    EXPECT_NEAR(render_center_depth(device, renderer), 4.0f, 1e-4f); // near face at x=-1, camera at x=-5

    float shifted[12];
    const float shifted_position[3] = {0.5f, 0, 0};
    const float identity_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(shifted_position, identity_wxyz, shifted);
    float* entry = rlt::data(rlt::transforms(device, renderer)) + cube.first_slot * 12;
    rlt::copy_to_renderer(device, renderer, shifted, entry, 12);
    rlt::update(device, renderer);
    EXPECT_NEAR(render_center_depth(device, renderer), 4.5f, 1e-4f);

    rlt::free(device, renderer);
}

struct DYNAMIC_MB_DEPTH_CONFIG: OVERLAY_DEPTH_CONFIG{
    static constexpr bool ENABLE_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = 2;
    static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
};
using DYNAMIC_MB_DEPTH_SPEC = rlt::rendering::raytracing::Specification<DYNAMIC_MB_DEPTH_CONFIG>;

namespace {
    void spin_transform(float radians, float out[12]){
        const float half = radians / 2.0f;
        const float origin[3] = {0, 0, 0};
        const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
        rlt::make_transform(origin, quaternion_wxyz, out);
    }

    template <typename RENDERER_SPEC>
    void mirror_cameras_open(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer){
        std::vector<rlt::rendering::raytracing::Camera<T>> staging(RENDERER_SPEC::NUM_CAMERAS);
        rlt::copy_from_renderer(device, renderer, rlt::data(rlt::cameras(device, renderer)), staging.data(), RENDERER_SPEC::NUM_CAMERAS);
        rlt::copy_to_renderer(device, renderer, staging.data(), rlt::data(rlt::cameras_open(device, renderer)), RENDERER_SPEC::NUM_CAMERAS);
    }

    // static camera under motion blur (open == close): all blur comes from the geometry
    template <typename RENDERER_SPEC>
    float render_center_depth_static_camera(DEVICE& device, rlt::rendering::raytracing::Renderer<RENDERER_SPEC, BACKEND>& renderer){
        const T position[3] = {-5, 0, 0};
        const T look_at[3] = {0, 0, 0};
        set_test_camera(device, renderer, position, look_at);
        mirror_cameras_open(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        float depth = 0;
        read_output(device, renderer, rlt::depth_buffer(device, renderer), &depth, 1);
        return depth;
    }
}

// dynamic motion blur averages per-sample geometry: with a static camera and a 2-sample blur
// from spin 0 (open) to spin 90 degrees (close), the passes render the slerp midpoints 22.5
// and 67.5 degrees exactly — the blurred depth must equal the mean of those two constant-pose
// renders (constant poses via the single-pose verb, which replicates across samples)
TEST(RL_TOOLS_SCENE_SUITE, DYNAMIC_MOTION_BLUR_DEPTH){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::ObjectAssembly blade_assembly;
    {
        rlt::rendering::raytracing::Object blade;
        rlt::rendering::raytracing::Mesh mesh = make_cube(0, 1);
        for(size_t vertex_i = 1; vertex_i < mesh.vertices.size(); vertex_i += 3){
            mesh.vertices[vertex_i] *= 0.1f;
        }
        blade.meshes.push_back(mesh);
        blade_assembly.objects.push_back(blade);
        blade_assembly.parts.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}});
    }

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto blade_asset = rlt::add(device, pool, blade_assembly);

    rlt::rendering::raytracing::Renderer<DYNAMIC_MB_DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto blade = rlt::spawn(device, renderer, OverlayIndex{0}, blade_asset, identity);

    float spin[12];
    spin_transform(3.14159265f / 8.0f, spin);
    rlt::set_transform(device, renderer, OverlayIndex{0}, blade, spin);
    rlt::update(device, renderer);
    const float depth_midpoint_first = render_center_depth_static_camera(device, renderer);

    spin_transform(3.0f * 3.14159265f / 8.0f, spin);
    rlt::set_transform(device, renderer, OverlayIndex{0}, blade, spin);
    rlt::update(device, renderer);
    const float depth_midpoint_second = render_center_depth_static_camera(device, renderer);
    EXPECT_GT(std::abs(depth_midpoint_first - depth_midpoint_second), 1e-3f);

    float open[12], close[12];
    spin_transform(0.0f, open);
    spin_transform(3.14159265f / 2.0f, close);
    rlt::set_transform_pair(device, renderer, OverlayIndex{0}, blade, open, close);
    rlt::update(device, renderer);
    const float depth_blurred = render_center_depth_static_camera(device, renderer);
    EXPECT_NEAR(depth_blurred, (depth_midpoint_first + depth_midpoint_second) / 2.0f, 1e-4f);

    const float depth_blurred_again = render_center_depth_static_camera(device, renderer);
    EXPECT_EQ(depth_blurred, depth_blurred_again); // identical inputs -> bit-identical output

    rlt::free(device, renderer);
}

// producer path for the per-sample tensor: direct writes into transforms_motion (no host verb,
// no dirty flag) must be consumed by the sample passes
TEST(RL_TOOLS_SCENE_SUITE, DYNAMIC_MOTION_BLUR_TRANSFORMS_TENSOR){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<DYNAMIC_MB_DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto cube = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, identity);
    rlt::update(device, renderer);
    EXPECT_NEAR(render_center_depth_static_camera(device, renderer), 4.0f, 1e-4f);

    float shifted[12];
    const float identity_wxyz[4] = {1, 0, 0, 0};
    const float near_position[3] = {0.5f, 0, 0};
    rlt::make_transform(near_position, identity_wxyz, shifted);
    constexpr size_t SLAB = (size_t)DYNAMIC_MB_DEPTH_SPEC::NUM_OVERLAYS * DYNAMIC_MB_DEPTH_SPEC::MAX_OVERLAY_INSTANCES * 12;
    float* base = rlt::data(rlt::transforms_motion(device, renderer));
    rlt::copy_to_renderer(device, renderer, shifted, base + 0 * SLAB + cube.first_slot * 12, 12);
    rlt::copy_to_renderer(device, renderer, identity, base + 1 * SLAB + cube.first_slot * 12, 12);
    rlt::update(device, renderer);
    // sample 0 near face at x=-0.5 (depth 4.5), sample 1 at x=-1 (depth 4.0)
    EXPECT_NEAR(render_center_depth_static_camera(device, renderer), 4.25f, 1e-4f);

    rlt::free(device, renderer);
}

// a dynamic-motion-blur spec driven only by the single-pose verbs must render pixel-identical
// to a plain overlay renderer: constant entries replicate across samples, the camera is static
TEST(RL_TOOLS_SCENE_SUITE, DYNAMIC_MOTION_BLUR_STATIC_EQUIVALENCE){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<DYNAMIC_MB_DEPTH_SPEC, BACKEND> renderer_dynamic;
    rlt::rendering::raytracing::Renderer<OVERLAY_DEPTH_SPEC, BACKEND> renderer_plain;
    rlt::malloc(device, renderer_dynamic);
    rlt::malloc(device, renderer_plain);
    rlt::generate_probe_directions(device, renderer_dynamic);
    rlt::generate_probe_directions(device, renderer_plain);
    rlt::init(device, renderer_dynamic, scene, pool);
    rlt::init(device, renderer_plain, scene, pool);

    float spun[12];
    spin_transform(0.7f, spun);
    float depths[2];
    {
        rlt::attach(device, renderer_dynamic, (TI)0, OverlayIndex{0});
        const auto cube = rlt::spawn(device, renderer_dynamic, OverlayIndex{0}, cube_asset, spun);
        rlt::update(device, renderer_dynamic);
        depths[0] = render_center_depth_static_camera(device, renderer_dynamic);
    }
    {
        rlt::attach(device, renderer_plain, (TI)0, OverlayIndex{0});
        const auto cube = rlt::spawn(device, renderer_plain, OverlayIndex{0}, cube_asset, spun);
        rlt::update(device, renderer_plain);
        depths[1] = render_center_depth(device, renderer_plain);
    }
    EXPECT_EQ(depths[0], depths[1]);

    rlt::free(device, renderer_dynamic);
    rlt::free(device, renderer_plain);
}

struct DYNAMIC_MB_SEG_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 16, CAM_HEIGHT = 16, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_SEGMENTATION = true;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
    static constexpr bool ENABLE_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = 2;
    static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
};
using DYNAMIC_MB_SEG_SPEC = rlt::rendering::raytracing::Specification<DYNAMIC_MB_SEG_CONFIG>;

// segmentation stays single-sample at the shutter-close state: an overlay translating through
// the shutter labels only its close pose, never the open one
TEST(RL_TOOLS_SCENE_SUITE, DYNAMIC_MOTION_BLUR_SEGMENTATION_CLOSE){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    const uint32_t num_scene_instances = (uint32_t)scene.instances.size();
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<DYNAMIC_MB_SEG_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto cube = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, identity);
    float open[12];
    const float identity_wxyz[4] = {1, 0, 0, 0};
    const float open_position[3] = {0, 3, 0};
    rlt::make_transform(open_position, identity_wxyz, open);
    rlt::set_transform_pair(device, renderer, OverlayIndex{0}, cube, open, identity);
    rlt::update(device, renderer);

    const T position[3] = {-8, 0, 0};
    const T look_at[3] = {0, 0, 0};
    set_test_camera(device, renderer, position, look_at);
    mirror_cameras_open(device, renderer);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);

    std::vector<uint32_t> segmentation(DYNAMIC_MB_SEG_SPEC::CAM_PIXELS);
    read_output(device, renderer, rlt::segmentation_buffer(device, renderer), segmentation.data(), segmentation.size());
    const uint32_t cube_id = num_scene_instances + (uint32_t)cube.first_slot;
    constexpr TI CENTER = (DYNAMIC_MB_SEG_SPEC::CAM_HEIGHT / 2) * DYNAMIC_MB_SEG_SPEC::CAM_WIDTH + DYNAMIC_MB_SEG_SPEC::CAM_WIDTH / 2;
    EXPECT_EQ(segmentation[CENTER], cube_id);
    for(TI y = 0; y < DYNAMIC_MB_SEG_SPEC::CAM_HEIGHT; y++){
        for(TI x = 0; x < DYNAMIC_MB_SEG_SPEC::CAM_WIDTH; x++){
            if(segmentation[y * DYNAMIC_MB_SEG_SPEC::CAM_WIDTH + x] == cube_id){
                // labels only at the close pose (image center), never at the open pose
                EXPECT_LE(std::abs((int)x - (int)DYNAMIC_MB_SEG_SPEC::CAM_WIDTH / 2), 3);
                EXPECT_LE(std::abs((int)y - (int)DYNAMIC_MB_SEG_SPEC::CAM_HEIGHT / 2), 3);
            }
        }
    }

    rlt::free(device, renderer);
}

// the scene-accumulation bug class: reusing a scene across loads silently re-uploads all
// previously loaded geometry into every subsequent renderer — load refuses a non-empty scene,
// composition must be the explicit add
TEST(RL_TOOLS_SCENE_SUITE, LOAD_ASSERTS_EMPTY){
    DEVICE device;
    rlt::init(device);
    const std::string scene_file = std::string(RL_TOOLS_SCENE_TEST_DATA_PATH) + "/ProcTHOR-Train-1.glb";
    if(!std::filesystem::exists(scene_file)){
        GTEST_SKIP() << "scene file not available";
    }
    rlt::rendering::raytracing::Scene scene;
    ASSERT_TRUE(rlt::load(device, scene, scene_file));
    EXPECT_EQ(scene.objects.size(), (size_t)1);
    EXPECT_DEATH((void)rlt::load(device, scene, scene_file), "");
    ASSERT_TRUE(rlt::add(device, scene, scene_file)); // composition is explicit
    EXPECT_EQ(scene.objects.size(), (size_t)2);
}

TEST(RL_TOOLS_SCENE_SUITE, CAMERAS_TENSOR){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    rlt::add(device, scene, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<DEPTH_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene);

    // two successive writes through the cameras() accessor: each launch must consume the
    // latest tensor contents (no camera state cached at init or across renders)
    EXPECT_NEAR(render_center_depth(device, renderer), 4.0f, 1e-4f); // camera at x=-5, near face at x=-1
    const T position[3] = {-7, 0, 0};
    const T look_at[3] = {0, 0, 0};
    set_test_camera(device, renderer, position, look_at);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    {
        float center_depth = 0;
        read_output(device, renderer, rlt::depth_buffer(device, renderer), &center_depth, 1);
        EXPECT_NEAR(center_depth, 6.0f, 1e-4f);
    }

    rlt::free(device, renderer);
}

struct OVERLAY_RGB_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
};
using OVERLAY_RGB_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_RGB_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_SPAWN_DESPAWN){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    rlt::rendering::raytracing::Mesh overlay_cube = make_cube(0, 0.4f);
    overlay_cube.color[0] = 1.0f; overlay_cube.color[1] = 0.2f; overlay_cube.color[2] = 0.1f; // distinct from the anchor
    const auto cube_asset = rlt::add(device, pool, overlay_cube);

    rlt::rendering::raytracing::Renderer<OVERLAY_RGB_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0});

    rlt::rendering::raytracing::OverlayPlacement placements[8];
    for(int spawn_i = 0; spawn_i < 8; spawn_i++){
        float transform[12];
        const float position[3] = {0, (float)(spawn_i - 4), 0};
        const float orientation_wxyz[4] = {1, 0, 0, 0};
        rlt::make_transform(position, orientation_wxyz, transform);
        placements[spawn_i] = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform);
        EXPECT_EQ(placements[spawn_i].first_slot, (size_t)spawn_i); // deterministic first-fit
    }
    EXPECT_FALSE(rlt::can_spawn(device, renderer, OverlayIndex{0}, cube_asset)); // capacity exhausted
    rlt::update(device, renderer);
    const T camera_position[3] = {-8, 0, 0};
    const T look_at[3] = {0, 0, 0};
    std::vector<uint32_t> full;
    render_pixels(device, renderer, camera_position, look_at, full);
    rlt::despawn(device, renderer, OverlayIndex{0}, placements[3]);
    EXPECT_TRUE(rlt::can_spawn(device, renderer, OverlayIndex{0}, cube_asset)); // hole reopened
    EXPECT_FALSE(rlt::can_spawn(device, renderer, OverlayIndex{0}, rlt::rendering::raytracing::AssetHandle{99})); // bad handle
    rlt::update(device, renderer);
    std::vector<uint32_t> with_hole;
    render_pixels(device, renderer, camera_position, look_at, with_hole);
    EXPECT_NE(full, with_hole); // the despawned cube disappeared

    float transform[12];
    const float position[3] = {0, -1, 0}; // same pose the despawned cube had
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position, orientation_wxyz, transform);
    const auto respawned = rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform);
    EXPECT_EQ(respawned.first_slot, (size_t)3); // hole reused deterministically
    rlt::update(device, renderer);
    std::vector<uint32_t> refilled;
    render_pixels(device, renderer, camera_position, look_at, refilled);
    EXPECT_EQ(full, refilled);

    rlt::free(device, renderer);
}

struct OVERLAY_PBR_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 32, CAM_HEIGHT = 32, NUM_CAMERAS = 2, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::VeryHigh;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
};
using OVERLAY_PBR_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_PBR_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_SECONDARY_RAYS){
    DEVICE device;
    rlt::init(device);

    rlt::rendering::raytracing::Scene scene;
    {
        rlt::rendering::raytracing::Mesh floor;
        floor.color[0] = 1; floor.color[1] = 1; floor.color[2] = 1;
        const float floor_vertices[4 * 3] = {-10, -10, 0,  10, -10, 0,  10, 10, 0,  -10, 10, 0};
        const int floor_indices[2 * 3] = {0, 1, 2,  0, 2, 3};
        floor.vertices.assign(floor_vertices, floor_vertices + 12);
        floor.indices.assign(floor_indices, floor_indices + 6);
        rlt::add(device, scene, floor);
        scene.lights.push_back({1, {0, 0, 6}, {0, 0, -1}, {30, 30, 30}, 1, 0, 1, 0, 0}); // point light above
    }
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 0.8f));

    rlt::rendering::raytracing::Renderer<OVERLAY_PBR_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0}); // camera 0 sees the overlay cube, camera 1 does not

    float transform[12];
    const float position[3] = {2, 0, 2}; // between the light and the floor, off-axis so the floor stays visible
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position, orientation_wxyz, transform);
    rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform);
    rlt::update(device, renderer);

    const T camera_position[3] = {-6, 0, 4};
    const T look_at[3] = {2, 0, 0}; // looking at the shadowed floor region
    set_same_pose_cameras(device, renderer, camera_position, look_at);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    std::vector<uint32_t> pixel_staging((size_t)decltype(renderer)::SPEC::NUM_CAMERAS * decltype(renderer)::SPEC::CAM_PIXELS);
    read_output(device, renderer, rlt::frame_buffer(device, renderer), pixel_staging.data(), pixel_staging.size());
    const uint32_t* pixels = pixel_staging.data();

    long brightness[2] = {0, 0};
    for(TI camera_i = 0; camera_i < 2; camera_i++){
        for(TI pixel_i = 0; pixel_i < OVERLAY_PBR_SPEC::CAM_PIXELS; pixel_i++){
            brightness[camera_i] += (long)(pixels[camera_i * OVERLAY_PBR_SPEC::CAM_PIXELS + pixel_i] & 0xFF);
        }
    }
    EXPECT_LT(brightness[0], brightness[1]); // overlay cube shades camera 0's view (occlusion + shadow)

    rlt::free(device, renderer);
}

struct OVERLAY_PROBE_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 4, CAM_HEIGHT = 4, NUM_CAMERAS = 2, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr TI NUM_OVERLAYS = 1, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
};
using OVERLAY_PROBE_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_PROBE_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_PROBES){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 1));

    rlt::rendering::raytracing::Renderer<OVERLAY_PROBE_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);
    rlt::attach(device, renderer, (TI)0, OverlayIndex{0}); // only camera 0 senses the overlay

    float transform[12];
    const float position[3] = {-3, 0, 0}; // forward probe (looking -x, away from the anchor) hits the near face at x=-2
    const float orientation_wxyz[4] = {1, 0, 0, 0};
    rlt::make_transform(position, orientation_wxyz, transform);
    rlt::spawn(device, renderer, OverlayIndex{0}, cube_asset, transform);
    rlt::update(device, renderer);

    const T camera_position[3] = {0, 0, 0};
    const T look_at[3] = {-1, 0, 0};
    set_same_pose_cameras(device, renderer, camera_position, look_at);
    rlt::render(device, renderer);
    rlt::probe(device, renderer);
    rlt::synchronize(device, renderer);
    std::vector<rlt::rendering::raytracing::CollisionResult> probe_staging((size_t)OVERLAY_PROBE_SPEC::NUM_CAMERAS * OVERLAY_PROBE_SPEC::NUM_PROBES);
    read_output(device, renderer, rlt::collision_results(device, renderer), probe_staging.data(), probe_staging.size());
    const auto* probes = probe_staging.data();
    ASSERT_NE(rlt::data(renderer.collision_results), nullptr);
    EXPECT_EQ(probes[0].hit, 1); // camera 0, forward probe: overlay cube at distance 2
    EXPECT_NEAR(probes[0].distance, 2.0f, 1e-4f);
    EXPECT_EQ(probes[OVERLAY_PROBE_SPEC::NUM_PROBES].hit, 0); // camera 1, forward probe: nothing

    rlt::free(device, renderer);
}

struct OVERLAY_MANY_CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = 4, CAM_HEIGHT = 4, NUM_CAMERAS = 1, NUM_PROBES = 4;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr TI NUM_OVERLAYS = 64, MAX_OVERLAY_INSTANCES = 8, MAX_OVERLAYS_PER_CAMERA = 1;
};
using OVERLAY_MANY_SPEC = rlt::rendering::raytracing::Specification<OVERLAY_MANY_CONFIG>;

TEST(RL_TOOLS_SCENE_SUITE, OVERLAY_UPDATE_COST_SMOKE){
    DEVICE device;
    rlt::init(device);

    auto scene = make_far_anchor_scene(device);
    rlt::rendering::raytracing::AssetPool pool;
    const auto cube_asset = rlt::add(device, pool, make_cube(0, 0.3f));

    rlt::rendering::raytracing::Renderer<OVERLAY_MANY_SPEC, BACKEND> renderer;
    rlt::malloc(device, renderer);
    rlt::generate_probe_directions(device, renderer);
    rlt::init(device, renderer, scene, pool);

    rlt::rendering::raytracing::OverlayPlacement placements[64][8];
    for(TI overlay = 0; overlay < 64; overlay++){
        for(int spawn_i = 0; spawn_i < 8; spawn_i++){
            float transform[12];
            const float position[3] = {(float)overlay, (float)spawn_i, 0};
            const float orientation_wxyz[4] = {1, 0, 0, 0};
            rlt::make_transform(position, orientation_wxyz, transform);
            placements[overlay][spawn_i] = rlt::spawn(device, renderer, OverlayIndex{overlay}, cube_asset, transform);
        }
    }
    const auto start = std::chrono::steady_clock::now();
    constexpr int STEPS = 100;
    for(int step_i = 0; step_i < STEPS; step_i++){
        for(TI overlay = 0; overlay < 64; overlay++){
            float transform[12];
            const float position[3] = {(float)overlay, (float)step_i * 0.01f, 0};
            const float orientation_wxyz[4] = {1, 0, 0, 0};
            rlt::make_transform(position, orientation_wxyz, transform);
            rlt::set_transform(device, renderer, OverlayIndex{overlay}, placements[overlay][0], (TI)0, transform);
        }
        rlt::update(device, renderer);
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    RL_TOOLS_RENDERING_RAYTRACING_LOG("overlay update smoke: " << milliseconds / STEPS << " ms per step (64 overlays x 8 slots)");

    const T smoke_camera_position[3] = {-8, 0, 0};
    const T smoke_look_at[3] = {0, 0, 0};
    set_same_pose_cameras(device, renderer, smoke_camera_position, smoke_look_at);
    const auto async_start = std::chrono::steady_clock::now();
    for(int step = 0; step < STEPS; step++){
        for(TI overlay = 0; overlay < 64; overlay++){
            float transform[12];
            const float position[3] = {0, 0, 0.01f * (float)(step + 1)};
            const float orientation_wxyz[4] = {1, 0, 0, 0};
            rlt::make_transform(position, orientation_wxyz, transform);
            rlt::set_transform(device, renderer, OverlayIndex{overlay}, placements[overlay][0], (TI)0, transform);
        }
        rlt::update_launch(device, renderer);
        rlt::render_launch(device, renderer);
        rlt::render_sync(device, renderer);
    }
    const auto async_end = std::chrono::steady_clock::now();
    const double async_ms_per_step = std::chrono::duration<double, std::milli>(async_end - async_start).count() / (double)STEPS;
    RL_TOOLS_RENDERING_RAYTRACING_LOG("overlay update+render (async launch) smoke: " << async_ms_per_step << " ms per step (64 overlays x 8 slots)");    EXPECT_LT(milliseconds / STEPS, 50.0); // loose regression tripwire, not a benchmark

    rlt::free(device, renderer);
}
