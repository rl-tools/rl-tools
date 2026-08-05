// Compile-only guard for AGENTS rule 13: this TU must build with -nostdinc++, proving that the
// generic raytracing kernels (and everything they include) pull in no C++ standard library headers.
// It is never executed.
#include <rl_tools/rl_tools.h>
#include <rl_tools/devices/wasm32.h>
#include <rl_tools/math/operations_generic.h>
#include <rl_tools/math/operations_wasm32.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_generic.h>

namespace rlt = rl_tools;
namespace generic = rl_tools::rendering::raytracing::backends::generic;

namespace {
    struct Shading{
        static constexpr bool LOAD_TEXTURES = true;
        static constexpr bool NORMAL_SHADING = true;
        static constexpr bool METALLIC_REFLECTIONS = true;
        static constexpr bool SRGB_OUTPUT = true;
        static constexpr bool CHECKER_BACKGROUND = true;
        static constexpr bool PBR_SHADING = true;
        static constexpr bool PUNCTUAL_LIGHT_SHADOWS = true;
    };
    struct Spec{
        using T = float;
        using TI = unsigned int;
        using SHADING = Shading;
        static constexpr TI CAM_WIDTH = 16;
        static constexpr TI CAM_HEIGHT = 16;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = 4;
        static constexpr TI GRID_COLS = 1;
        static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
        static constexpr bool ENABLE_MOTION_BLUR = false;
        static constexpr TI MOTION_BLUR_SAMPLES = 1;
        static constexpr bool ENABLE_ANTI_ALIASING = false;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = 1;
    };
    struct ShadingBasic: Shading{
        static constexpr bool PBR_SHADING = false;
    };
    struct SpecBasic: Spec{
        using SHADING = ShadingBasic;
    };
}

extern "C" void rl_tools_rendering_raytracing_generic_freestanding_check(){
    using DEVICE = rlt::devices::DefaultWASM32;
    using T = typename Spec::T;
    using TI = typename Spec::TI;
    DEVICE device;

    static T vertices[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    static const int indices[3] = {0, 1, 2};
    static generic::MeshView<T, TI> mesh;
    mesh.indices = indices;
    mesh.vertices = vertices;

    static generic::SceneView<T, TI> scene;
    static TI triangle_mesh[1] = {0};
    static TI triangle_local[1] = {0};
    scene.meshes = &mesh;
    scene.num_meshes = 1;
    scene.triangle_mesh = triangle_mesh;
    scene.triangle_local = triangle_local;
    scene.num_triangles = 1;
    static generic::BVHNode<T, TI> nodes[2];
    static TI primitives[1] = {0};
    static TI temp_primitives[1];
    static T triangle_bounds_min[3] = {0, 0, 0};
    static T triangle_bounds_max[3] = {1, 1, 0};
    static T centroids[3] = {0.5f, 0.5f, 0};
    static generic::ObjectView<T, TI> object;
    object.nodes = nodes;
    object.primitives = primitives;
    object.num_nodes = generic::build_bvh_nodes(nodes, primitives, temp_primitives, triangle_bounds_min, triangle_bounds_max, centroids, (TI)1);
    scene.objects = &object;
    scene.num_objects = 1;
    static generic::InstanceView<T, TI> instance;
    scene.instances = &instance;
    scene.num_instances = 1;
    static generic::BVHNode<T, TI> tlas_node;
    tlas_node = nodes[0];
    tlas_node.left_or_first = 0;
    tlas_node.count = 1;
    static TI tlas_primitive = 0;
    scene.tlas_nodes = &tlas_node;
    scene.tlas_primitives = &tlas_primitive;
    scene.num_tlas_nodes = 1;

    static rlt::rendering::raytracing::Camera<T> camera;
    static T probe_directions[3 * Spec::NUM_PROBES];
    static unsigned int frame_buffer[Spec::NUM_CAMERAS * Spec::CAM_PIXELS];
    static float depth_buffer[Spec::NUM_CAMERAS * Spec::CAM_PIXELS];
    static rlt::rendering::raytracing::CollisionResult collision_results[Spec::NUM_CAMERAS * Spec::NUM_PROBES];
    scene.cameras_close = &camera;
    scene.cameras_open = &camera;
    scene.probe_directions = probe_directions;
    scene.frame_buffer = frame_buffer;
    scene.depth_buffer = depth_buffer;
    scene.collision_results = collision_results;
    scene.max_depth = 10;
    scene.max_dist = 10;

    generic::render_frame<DEVICE, Spec, generic::OutputRGB>(device, scene);
    generic::render_frame<DEVICE, SpecBasic, generic::OutputRGB>(device, scene);
    generic::render_frame<DEVICE, Spec, generic::OutputDepth>(device, scene);
    generic::render_collision<DEVICE, Spec>(device, scene);
}
