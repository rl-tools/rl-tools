// Compile-only guard for AGENTS rule 13: this TU must build with -nostdinc++, proving that the
// generic raytracing kernels (and everything they include) pull in no C++ standard library headers.
// It is never executed.
#include <rl_tools/rl_tools.h>
#include <rl_tools/devices/wasm32.h>
#include <rl_tools/math/operations_generic.h>
#include <rl_tools/math/operations_wasm32.h>
#include <rl_tools/utils/assert/operations_wasm32.h>
#include <rl_tools/rendering/raytracing/backends/generic/operations_generic.h>

namespace rlt = rl_tools;
namespace out = rl_tools::rendering::raytracing::outputs;
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
    using Outputs = out::List<out::RGB<>, out::RGB<out::format::Float32>, out::Depth<>, out::Depth<out::format::RGBA8>, out::Normals<>, out::Normals<out::format::RGBA8>, out::Flow<>, out::FlowStatus<>, out::Flow<out::direction::Forward>, out::FlowStatus<out::direction::Forward>, out::Probes>;
    struct Spec: out::Requirements<Outputs>{
        using OUTPUTS = Outputs;
        using PLAN = out::Requirements<Outputs>;
        static constexpr bool FUSE_AUXILIARY = true;
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
        static constexpr bool ENABLE_OVERLAYS = false;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = 0;
    };
    struct ShadingBasic: Shading{
        static constexpr bool PBR_SHADING = false;
    };
    struct SpecBasic: Spec{
        using SHADING = ShadingBasic;
    };
    // pins the T = double instantiation of the freestanding kernels with overlays: the flow-delta
    // table stays float regardless of T, so the flow path must compile with mixed element types
    struct SpecDouble: Spec{
        using T = double;
        static constexpr bool ENABLE_OVERLAYS = true;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
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
    static out::Buffers<Spec, Spec::OUTPUTS> buffers;
    static out::Buffers<SpecBasic, SpecBasic::OUTPUTS> buffers_basic;
    static out::Buffers<SpecDouble, SpecDouble::OUTPUTS> buffers_double;
    scene.cameras_close = &camera;
    scene.cameras_open = &camera;
    scene.cameras_previous = &camera;
    scene.probe_directions = probe_directions;
    scene.max_dist = 10;
    static unsigned int camera_table[out::CAMERA_TABLE_STRIDE];
    rlt::fill_camera_table<Spec>(device, camera_table);
    scene.camera_table = camera_table;

    generic::render_frame<DEVICE, Spec>(device, scene, buffers);
    generic::render_frame<DEVICE, SpecBasic>(device, scene, buffers_basic);
    generic::render_auxiliary_frame<DEVICE, Spec>(device, scene, buffers);
    generic::render_auxiliary_frame<DEVICE, Spec, true>(device, scene, buffers);
    generic::render_collision<DEVICE, Spec>(device, scene, buffers);

    static generic::SceneView<double, TI> scene_double;
    scene_double.camera_table = camera_table;
    generic::render_auxiliary_frame<DEVICE, SpecDouble>(device, scene_double, buffers_double);
    generic::render_auxiliary_frame<DEVICE, SpecDouble, true>(device, scene_double, buffers_double);
}
