#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_TYPED_OUTPUTS_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include "render_copy.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
    using namespace rl_tools;
    namespace rrt = rendering::raytracing;
    namespace outputs = rrt::outputs;
    using DEVICE = devices::DefaultCPU;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_TYPED_OUTPUTS_ACTIVE_BACKEND)
    using BACKEND = rrt::backends::Default;
#else
    using BACKEND = rrt::backends::Generic;
#endif
    struct DepthParameters { static constexpr float SCALE = 6; };
    struct FlowParameters { static constexpr float SCALE = 7; };
    using DepthColor = outputs::Depth<outputs::format::RGBA8, DepthParameters>;
    using FlowColor = outputs::Flow<outputs::direction::Backward, outputs::format::RGBA8, FlowParameters>;
    using ForwardColor = outputs::Flow<outputs::direction::Forward, outputs::format::RGBA8, FlowParameters>;
    using All = outputs::List<
        outputs::Probes, outputs::Normals<outputs::format::RGBA8>, outputs::RGB<outputs::format::Float32>, DepthColor,
        outputs::RGB<>, outputs::Depth<>, outputs::Normals<>, outputs::Segmentation<>, outputs::Segmentation<outputs::format::RGBA8>,
        outputs::Segmentation<outputs::format::UInt32, outputs::labels::Semantic>, outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>,
        outputs::Flow<>, FlowColor, outputs::FlowStatus<>, outputs::FlowStatus<outputs::direction::Backward, outputs::format::RGBA8>,
        outputs::Flow<outputs::direction::Forward>, ForwardColor, outputs::FlowStatus<outputs::direction::Forward>, outputs::FlowStatus<outputs::direction::Forward, outputs::format::RGBA8>>;

    template <typename LIST, bool AA = false, bool MB = false, bool OVERLAYS = false>
    struct Config: rrt::config::Default<float, DEVICE::index_t> {
        using OUTPUTS = LIST;
        using SHADING = rrt::Low;
        static constexpr DEVICE::index_t CAM_WIDTH = 16, CAM_HEIGHT = 12, NUM_CAMERAS = 1, NUM_PROBES = 2;
        static constexpr bool ENABLE_ANTI_ALIASING = AA, ENABLE_MOTION_BLUR = MB, ENABLE_DYNAMIC_MOTION_BLUR = MB && OVERLAYS;
        static constexpr DEVICE::index_t NUM_OVERLAYS = OVERLAYS ? 1 : 0, MAX_OVERLAY_INSTANCES = OVERLAYS ? 1 : 0, MAX_OVERLAYS_PER_CAMERA = OVERLAYS ? 1 : 0;
        static constexpr DEVICE::index_t ANTI_ALIASING_GRID_SIZE = 2, MOTION_BLUR_SAMPLES = 2;
    };
    using ForwardOnly = rrt::Specification<Config<outputs::List<outputs::Flow<outputs::direction::Forward>>>>;
    static_assert(ForwardOnly::HAS_FLOW && ForwardOnly::HAS_FLOW_FORWARD && !ForwardOnly::HAS_FLOW_BACKWARD);
    static_assert(!ForwardOnly::HAS_FLOW_FORWARD_STATUS && !ForwardOnly::HAS_AUXILIARY && !ForwardOnly::HAS_PROBES);
    static_assert(outputs::Layout<ForwardOnly>::SIZE_BYTES == 16 * 12 * 2 * sizeof(float));
    using FloatOnly = rrt::Specification<Config<outputs::List<outputs::RGB<outputs::format::Float32>>>>;
    static_assert(FloatOnly::HAS_RGB && FloatOnly::HAS_RGB32F && !FloatOnly::HAS_RGBA8);
    static_assert(outputs::Layout<FloatOnly>::SIZE_BYTES == 16 * 12 * 3 * sizeof(float));

    rendering::Bundle<float> scene(DEVICE& device) {
        rendering::Bundle<float> bundle;
        rrt::Mesh triangle;
        triangle.vertices = {3, -100, -100, 3, 100, -100, 3, 0, 100};
        triangle.indices = {0, 1, 2};
        triangle.color[0] = 1;
        triangle.color[1] = 0;
        triangle.color[2] = 0;
        add(device, bundle.scene, triangle);
        bundle.scene.objects[0].segmentation_class = 3;
        bundle.metadata.max_ray_length = 20;
        return bundle;
    }
    template <typename SPEC>
    void set_inputs(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, unsigned int state) {
        const float position[3] = {0, 0, 0}, target[3] = {3, 0, 0}, up[3] = {0, 0, 1};
        const auto camera = make_camera_data(position, target, up, 80.0f, 4.0f / 3.0f);
        golden::copy_in(device, renderer.device, &camera, cameras(device, renderer));
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            golden::copy_in(device, renderer.device, &camera, cameras_open(device, renderer));
        }
        if constexpr (SPEC::HAS_FLOW) {
            auto previous = camera;
            if(state == 0) {
                previous.pos[1] -= .03f;
            }
            else if(state == 1) {
                const float reverse[3] = {-3, 0, 0};
                previous = make_camera_data(position, reverse, up, 80.0f, 4.0f / 3.0f);
            }
            else {
                previous.pos[1] -= 50.0f;
            }
            golden::copy_in(device, renderer.device, &previous, cameras_previous(device, renderer));
        }
    }
    template <typename SPEC, typename OUTPUT>
    auto read(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, OUTPUT descriptor) {
        std::vector<typename outputs::Traits<OUTPUT>::T> values;
        golden::copy_out(renderer.device, device, output(device, renderer, descriptor), values);
        return values;
    }
    void expect_color_near(uint32_t actual, uint32_t expected) {
        for(unsigned int channel = 0; channel < 4; channel++) {
            EXPECT_LE(std::abs(int((actual >> (8 * channel)) & 255) - int((expected >> (8 * channel)) & 255)), 1);
        }
    }
    template <typename SPEC, typename DIRECTION, typename COLOR>
    void check_flow(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, DIRECTION, COLOR descriptor) {
        const auto flow = read(device, renderer, outputs::Flow<DIRECTION>{});
        const auto status = read(device, renderer, outputs::FlowStatus<DIRECTION>{});
        const auto color = read(device, renderer, descriptor);
        const auto status_color = read(device, renderer, outputs::FlowStatus<DIRECTION, outputs::format::RGBA8>{});
        for(size_t index = 0; index < status.size(); index++) {
            const auto expected = status[index] == 0 ? flow_color(device, flow[index * 2], flow[index * 2 + 1], FlowParameters::SCALE) : flow_status_color(device, status[index]);
            expect_color_near(color[index], expected);
            EXPECT_EQ(status_color[index], flow_status_color(device, status[index]));
        }
    }
    template <typename SPEC>
    void check_joint(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, unsigned int state) {
        const auto depth = read(device, renderer, outputs::Depth<>{});
        const auto depth_colors = read(device, renderer, DepthColor{});
        const auto normals = read(device, renderer, outputs::Normals<>{});
        const auto normal_colors = read(device, renderer, outputs::Normals<outputs::format::RGBA8>{});
        const auto ids = read(device, renderer, outputs::Segmentation<>{});
        const auto id_color = read(device, renderer, outputs::Segmentation<outputs::format::RGBA8>{});
        const auto classes = read(device, renderer, outputs::Segmentation<outputs::format::UInt32, outputs::labels::Semantic>{});
        const auto class_color = read(device, renderer, outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>{});
        const auto rgb = read(device, renderer, outputs::RGB<>{});
        const auto rgb_float = read(device, renderer, outputs::RGB<outputs::format::Float32>{});
        for(size_t index = 0; index < depth.size(); index++) {
            EXPECT_NEAR(depth[index], 3.0f, 1e-4f);
            const uint32_t shade = uint32_t(std::clamp(depth[index] / DepthParameters::SCALE, 0.0f, 1.0f) * 255.0f);
            expect_color_near(depth_colors[index], 0xff000000u | shade | (shade << 8) | (shade << 16));
            EXPECT_EQ(normal_colors[index], normal_color(device, normals.data() + index * 3));
            EXPECT_EQ(ids[index], 0u);
            EXPECT_EQ(classes[index], 3u);
            EXPECT_EQ(id_color[index], segmentation_color(device, ids[index]));
            EXPECT_EQ(class_color[index], segmentation_color(device, classes[index]));
            EXPECT_EQ(rgb[index], 0xff0000ffu);
            EXPECT_FLOAT_EQ(rgb_float[index * 3], 1.0f);
            EXPECT_FLOAT_EQ(rgb_float[index * 3 + 1], 0.0f);
            EXPECT_FLOAT_EQ(rgb_float[index * 3 + 2], 0.0f);
        }
        check_flow(device, renderer, outputs::direction::Backward{}, FlowColor{});
        check_flow(device, renderer, outputs::direction::Forward{}, ForwardColor{});
        if(state == 1) {
            const auto backward = read(device, renderer, outputs::FlowStatus<>{});
            const auto forward = read(device, renderer, outputs::FlowStatus<outputs::direction::Forward>{});
            EXPECT_TRUE(std::all_of(backward.begin(), backward.end(), [](uint32_t value) { return value == 2; }));
            EXPECT_TRUE(std::all_of(forward.begin(), forward.end(), [](uint32_t value) { return value == 1; }));
        }
        if(state == 2) {
            const auto backward = read(device, renderer, outputs::FlowStatus<>{});
            EXPECT_TRUE(std::all_of(backward.begin(), backward.end(), [](uint32_t value) { return value == 3; }));
        }
        const auto probes = read(device, renderer, outputs::Probes{});
        EXPECT_EQ(probes[0].hit, 1);
        EXPECT_NEAR(probes[0].distance, 3.0f, 1e-5f);
    }
    template <bool AA, bool MB>
    void joint_case() {
        DEVICE device;
        init(device);
        device.logger.verbosity = devices::logging::Verbosity::warning;
        auto bundle = scene(device);
        using SPEC = rrt::Specification<Config<All, AA, MB>>;
        rrt::Renderer<SPEC, BACKEND> renderer;
        malloc(device, renderer);
        for(unsigned int state = 0; state < 3; state++) {
            init(device, renderer, bundle);
            set_inputs(device, renderer, state);
            render(device, renderer);
            generate_probe_directions(device, renderer);
            probe(device, renderer);
            check_joint(device, renderer, state);
        }
        free(device, renderer);
    }
    template <auto... OFFSETS>
    using Of3 = outputs::cameras::Every<3, OFFSETS...>;
    using Mixed = outputs::List<
        outputs::Cameras<Of3<0, 1>, outputs::RGB<>>, outputs::Cameras<Of3<1>, outputs::RGB<outputs::format::Float32>>,
        outputs::Cameras<Of3<0, 2>, outputs::Depth<>>, outputs::Cameras<Of3<2>, DepthColor>,
        outputs::Cameras<Of3<0>, outputs::Normals<>>, outputs::Normals<outputs::format::RGBA8>,
        outputs::Cameras<Of3<1>, outputs::Segmentation<>>, outputs::Cameras<Of3<0, 1>, outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>>,
        outputs::Cameras<Of3<0>, outputs::Flow<>>, outputs::Cameras<Of3<1>, outputs::FlowStatus<>>, outputs::Cameras<Of3<2>, FlowColor>,
        outputs::Cameras<Of3<2>, outputs::Flow<outputs::direction::Forward>>, outputs::FlowStatus<outputs::direction::Forward>,
        outputs::Cameras<Of3<1>, outputs::Probes>>;
    template <typename LIST, bool AA, bool MB, bool OVERLAYS = false>
    struct TripleConfig: Config<LIST, AA, MB, OVERLAYS> {
        static constexpr DEVICE::index_t NUM_CAMERAS = 3;
    };
    template <typename SPEC>
    void set_triple_inputs(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer) {
        const float up[3] = {0, 0, 1};
        const float positions[3][3] = {{0, 0, 0}, {0, .4f, .2f}, {.5f, -.3f, 0}}, targets[3][3] = {{3, 0, 0}, {3, -.3f, 0}, {3, .2f, .1f}};
        rrt::Camera<float> close[3], previous[3];
        for(unsigned int camera_i = 0; camera_i < 3; camera_i++) {
            close[camera_i] = make_camera_data(positions[camera_i], targets[camera_i], up, 80.0f, 4.0f / 3.0f);
            previous[camera_i] = close[camera_i];
            previous[camera_i].pos[1] -= .03f * (float)(camera_i + 1);
        }
        golden::copy_in(device, renderer.device, close, cameras(device, renderer));
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            golden::copy_in(device, renderer.device, previous, cameras_open(device, renderer));
        }
        golden::copy_in(device, renderer.device, previous, cameras_previous(device, renderer));
    }
    // every row of the selected output equals the selected camera's slice of the all-camera render
    template <typename ENTRY, typename FULL_SPEC, typename MIXED_SPEC>
    void expect_selected_rows(DEVICE& device, rrt::Renderer<FULL_SPEC, BACKEND>& full, rrt::Renderer<MIXED_SPEC, BACKEND>& mixed) {
        using OUTPUT = typename outputs::Entry<ENTRY>::OUTPUT;
        using SELECTION = typename outputs::Entry<ENTRY>::SELECTION;
        const auto all = read(device, full, OUTPUT{});
        const auto selected = read(device, mixed, OUTPUT{});
        const size_t per_camera = all.size() / FULL_SPEC::NUM_CAMERAS;
        ASSERT_EQ(selected.size(), per_camera * outputs::cameras::count<SELECTION>(FULL_SPEC::NUM_CAMERAS));
        size_t row = 0;
        for(size_t camera = 0; camera < FULL_SPEC::NUM_CAMERAS; camera++) {
            if(!SELECTION::selected(camera)) {
                continue;
            }
            EXPECT_EQ(std::memcmp(selected.data() + row * per_camera, all.data() + camera * per_camera, per_camera * sizeof(selected[0])), 0) << "camera " << camera;
            row++;
        }
    }
    template <typename FULL_SPEC, typename MIXED_SPEC, typename... ENTRIES>
    void expect_selected_rows(DEVICE& device, rrt::Renderer<FULL_SPEC, BACKEND>& full, rrt::Renderer<MIXED_SPEC, BACKEND>& mixed, outputs::List<ENTRIES...>) {
        (expect_selected_rows<ENTRIES>(device, full, mixed), ...);
    }
    template <bool AA, bool MB, bool OVERLAYS = false>
    void camera_selection_case() {
        DEVICE device;
        init(device);
        device.logger.verbosity = devices::logging::Verbosity::warning;
        auto bundle = scene(device);
        using FULL_SPEC = rrt::Specification<TripleConfig<All, AA, MB, OVERLAYS>>;
        using MIXED_SPEC = rrt::Specification<TripleConfig<Mixed, AA, MB, OVERLAYS>>;
        static_assert(outputs::TensorSpecification<MIXED_SPEC, outputs::Depth<>>::SIZE == 2 * FULL_SPEC::CAM_PIXELS);
        static_assert(outputs::TensorSpecification<MIXED_SPEC, outputs::RGB<>>::SIZE == 2 * FULL_SPEC::CAM_PIXELS);
        rrt::Renderer<FULL_SPEC, BACKEND> full;
        rrt::Renderer<MIXED_SPEC, BACKEND> mixed;
        malloc(device, full);
        malloc(device, mixed);
        rrt::AssetPool pool;
        if constexpr (OVERLAYS) {
            rrt::Object object;
            object.segmentation_class = 5;
            rrt::Mesh triangle;
            triangle.vertices = {0, -.8f, -.8f, 0, .8f, -.8f, 0, 0, .8f};
            triangle.indices = {0, 1, 2};
            triangle.color[0] = 0;
            triangle.color[1] = 1;
            triangle.color[2] = 0;
            object.meshes.push_back(triangle);
            const auto asset = add(device, pool, object);
            const auto prepare = [&](auto& renderer) {
                init(device, renderer, bundle, pool);
                const float open[12] = {1, 0, 0, 1.5f, 0, 1, 0, -.8f, 0, 0, 1, 0};
                const float close[12] = {1, 0, 0, 1.5f, 0, 1, 0, .8f, 0, 0, 1, 0};
                const auto placement = spawn(device, renderer, rrt::OverlayIndex{0}, asset, open);
                for(DEVICE::index_t camera = 0; camera < FULL_SPEC::NUM_CAMERAS; camera++) {
                    attach(device, renderer, camera, rrt::OverlayIndex{0});
                }
                set_transform_pair(device, renderer, rrt::OverlayIndex{0}, placement, open, close);
                update(device, renderer);
            };
            prepare(full);
            prepare(mixed);
        }
        else {
            init(device, full, bundle);
            init(device, mixed, bundle);
        }
        set_triple_inputs(device, full);
        set_triple_inputs(device, mixed);
        render(device, full);
        render(device, mixed);
        generate_probe_directions(device, full);
        generate_probe_directions(device, mixed);
        probe(device, full);
        probe(device, mixed);
        expect_selected_rows(device, full, mixed, Mixed{});
        if constexpr (OVERLAYS) {
            const auto colors = read(device, full, outputs::RGB<>{});
            EXPECT_TRUE(std::any_of(colors.begin(), colors.end(), [](uint32_t color) {
                return (color & 255u) > 0 && ((color >> 8) & 255u) > 0;
            }));
        }
        free(device, mixed);
        free(device, full);
    }
    template <typename OUTPUT, typename JOINT_SPEC>
    void single_case(DEVICE& device, rendering::Bundle<float>& bundle, rrt::Renderer<JOINT_SPEC, BACKEND>& joint, OUTPUT descriptor) {
        using SPEC = rrt::Specification<Config<outputs::List<OUTPUT>, JOINT_SPEC::ENABLE_ANTI_ALIASING, JOINT_SPEC::ENABLE_MOTION_BLUR, JOINT_SPEC::ENABLE_OVERLAYS>>;
        rrt::Renderer<SPEC, BACKEND> renderer;
        malloc(device, renderer);
        init(device, renderer, bundle);
        set_inputs(device, renderer, 0);
        if constexpr (SPEC::HAS_PROBES) {
            generate_probe_directions(device, renderer);
            probe(device, renderer);
        }
        else {
            render(device, renderer);
        }
        const auto actual = read(device, renderer, descriptor);
        const auto expected = read(device, joint, descriptor);
        ASSERT_EQ(actual.size(), expected.size());
        for(size_t index = 0; index < actual.size(); index++) {
            if constexpr (std::is_same_v<OUTPUT, outputs::Probes>) {
                EXPECT_EQ(actual[index].hit, expected[index].hit);
                EXPECT_FLOAT_EQ(actual[index].distance, expected[index].distance);
            }
            else if constexpr (std::is_same_v<typename outputs::Traits<OUTPUT>::T, float>) {
                EXPECT_NEAR(actual[index], expected[index], 1e-5f);
            }
            else {
                EXPECT_EQ(actual[index], expected[index]);
            }
        }
        free(device, renderer);
    }
}

TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, JOINT_REPRESENTATIONS) {
    joint_case<false, false>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, ANTI_ALIASING) {
    joint_case<true, false>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, MOTION_BLUR) {
    joint_case<false, true>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, CAMERA_SELECTIONS) {
    camera_selection_case<false, false>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, CAMERA_SELECTIONS_SEPARATE_PASS) {
    camera_selection_case<true, true>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, CAMERA_SELECTIONS_DYNAMIC_MOTION_BLUR) {
    camera_selection_case<false, true, true>();
}
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, SELECTED_OUTPUTS_MATCH_JOINT) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto bundle = scene(device);
    using SPEC = rrt::Specification<Config<All>>;
    rrt::Renderer<SPEC, BACKEND> joint;
    malloc(device, joint);
    init(device, joint, bundle);
    set_inputs(device, joint, 0);
    render(device, joint);
    generate_probe_directions(device, joint);
    probe(device, joint);
    single_case(device, bundle, joint, outputs::RGB<outputs::format::Float32>{});
    single_case(device, bundle, joint, DepthColor{});
    single_case(device, bundle, joint, outputs::Normals<outputs::format::RGBA8>{});
    single_case(device, bundle, joint, outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>{});
    single_case(device, bundle, joint, outputs::Flow<outputs::direction::Forward>{});
    single_case(device, bundle, joint, outputs::FlowStatus<outputs::direction::Forward, outputs::format::RGBA8>{});
    single_case(device, bundle, joint, outputs::FlowStatus<>{});
    single_case(device, bundle, joint, outputs::Probes{});
    free(device, joint);
}

TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, DYNAMIC_MOTION_BLUR_FLOAT_RGB_ONLY) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto bundle = scene(device);
    using SPEC = rrt::Specification<Config<All, false, true, true>>;
    static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR);
    rrt::Renderer<SPEC, BACKEND> joint;
    malloc(device, joint);
    init(device, joint, bundle);
    set_inputs(device, joint, 0);
    render(device, joint);
    const auto color = read(device, joint, outputs::RGB<outputs::format::Float32>{});
    for(size_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++) {
        EXPECT_FLOAT_EQ(color[pixel * 3], 1.0f);
        EXPECT_FLOAT_EQ(color[pixel * 3 + 1], 0.0f);
        EXPECT_FLOAT_EQ(color[pixel * 3 + 2], 0.0f);
    }
    single_case(device, bundle, joint, outputs::RGB<outputs::format::Float32>{});
    free(device, joint);
}

TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, ANGLED_NORMAL_COLOR) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto bundle = scene(device);
    const auto original_vertices = bundle.scene.objects[0].meshes[0].vertices;
    using SPEC = rrt::Specification<Config<outputs::List<outputs::Normals<>, outputs::Normals<outputs::format::RGBA8>>>>;
    rrt::Renderer<SPEC, BACKEND> renderer;
    malloc(device, renderer);
    const float components[] = {-.9960784316062927f, .06274503469467163f, .07843130826950073f, .6431371569633484f, .5333332419395447f};
    for(float component : components) {
        const float initial_slope = component < 0 ? std::sqrt(1.f / (component * component) - 1.f) : component / std::sqrt(1.f - component * component);
        for(int neighbor = -16; neighbor <= 16; neighbor++) {
            float slope = initial_slope;
            for(int step = 0; step < std::abs(neighbor); step++) {
                slope = std::nextafter(slope, float(neighbor));
            }
            auto& vertices = bundle.scene.objects[0].meshes[0].vertices;
            vertices = original_vertices;
            for(size_t vertex = 0; vertex < vertices.size(); vertex += 3) {
                vertices[vertex] += slope * vertices[vertex + 1];
            }
            init(device, renderer, bundle);
            set_inputs(device, renderer, 0);
            render(device, renderer);
            const auto normals = read(device, renderer, outputs::Normals<>{});
            const auto colors = read(device, renderer, outputs::Normals<outputs::format::RGBA8>{});
            for(size_t pixel = 0; pixel < colors.size(); pixel++) {
                EXPECT_LT(normals[pixel * 3], 0.f);
                EXPECT_GT(normals[pixel * 3 + 1], 0.f);
                EXPECT_EQ(colors[pixel], normal_color(device, normals.data() + pixel * 3)) << component << ":" << neighbor;
            }
        }
    }
    free(device, renderer);
}

TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, DEPTH_MISS_REPRESENTATIONS) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto bundle = scene(device);
    using SPEC = rrt::Specification<Config<outputs::List<DepthColor, outputs::Depth<>>>>;
    rrt::Renderer<SPEC, BACKEND> renderer;
    malloc(device, renderer);
    init(device, renderer, bundle);
    const float position[3] = {0, 0, 0}, target[3] = {-3, 0, 0}, up[3] = {0, 0, 1};
    const auto camera = make_camera_data(position, target, up, 80.f, 4.f / 3.f);
    golden::copy_in(device, renderer.device, &camera, cameras(device, renderer));
    render(device, renderer);
    const auto depth = read(device, renderer, outputs::Depth<>{});
    const auto color = read(device, renderer, DepthColor{});
    for(size_t pixel = 0; pixel < depth.size(); pixel++) {
        uint32_t bits;
        std::memcpy(&bits, &depth[pixel], sizeof(bits));
        EXPECT_EQ(bits, 0x7f800000u);
        EXPECT_EQ(color[pixel], 0xffffffffu);
    }
    free(device, renderer);
}

#if defined(RL_TOOLS_RENDERING_RAYTRACING_TYPED_OUTPUTS_ACTIVE_BACKEND) && defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU)
TEST(RENDERING_RAYTRACING_TYPED_OUTPUTS, INPUT_COPIES_LEAVE_OUTPUTS_ON_DEVICE) {
    DEVICE device;
    init(device);
    device.logger.verbosity = devices::logging::Verbosity::warning;
    auto bundle = scene(device);
    rrt::AssetPool pool;
    auto object = bundle.scene.objects[0];
    object.meshes[0].color[0] = 0;
    object.meshes[0].color[1] = 1;
    const auto asset = add(device, pool, object);
    using SPEC = rrt::Specification<Config<outputs::List<outputs::RGB<>, outputs::Depth<>, outputs::Probes>, false, false, true>>;
    rrt::Renderer<SPEC, BACKEND> renderer;
    malloc(device, renderer);
    init(device, renderer, bundle, pool);
    set_inputs(device, renderer, 0);
    generate_probe_directions(device, renderer);
    const float pose[12] = {1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0};
    spawn(device, renderer, rrt::OverlayIndex{0}, asset, pose);
    attach(device, renderer, DEVICE::index_t(0), rrt::OverlayIndex{0});
    update(device, renderer);

    auto& depth = output(device, renderer, outputs::Depth<>{});
    auto& rgb = output(device, renderer, outputs::RGB<>{});
    auto& probes = output(device, renderer, outputs::Probes{});
    std::fill_n(data(depth), SPEC::CAM_PIXELS, -99.f);
    std::fill_n(data(rgb), SPEC::CAM_PIXELS, 0u);
    const auto poison_probes = [&] {
        for(DEVICE::index_t index = 0; index < SPEC::NUM_PROBES; index++) {
            data(probes)[index].distance = -99.f;
            data(probes)[index].hit = 42;
        }
    };
    const auto expect_images_unread = [&] {
        EXPECT_TRUE(std::all_of(data(depth), data(depth) + SPEC::CAM_PIXELS, [](float value) { return value == -99.f; }));
        EXPECT_TRUE(std::all_of(data(rgb), data(rgb) + SPEC::CAM_PIXELS, [](uint32_t value) { return value == 0u; }));
    };
    poison_probes();
    render_launch(device, renderer);
    std::vector<rrt::Camera<float>> camera_values;
    golden::copy_out(renderer.device, device, cameras(device, renderer), camera_values);
    expect_images_unread();
    camera_values[0].pos[0] = .25f;
    golden::copy_in(device, renderer.device, camera_values.data(), cameras(device, renderer));
    expect_images_unread();

    std::vector<float> poses;
    golden::copy_out(renderer.device, device, transforms(device, renderer), poses);
    poses[3] = -.5f;
    golden::copy_in(device, renderer.device, poses.data(), transforms(device, renderer));
    update(device, renderer);
    expect_images_unread();
    probe_launch(device, renderer);
    const auto probe_values = read(device, renderer, outputs::Probes{});
    EXPECT_EQ(probe_values[0].hit, 1);
    EXPECT_NEAR(probe_values[0].distance, 2.25f, 1e-5f);
    expect_images_unread();

    poison_probes();
    probe_launch(device, renderer);
    render_launch(device, renderer);
    expect_images_unread();
    EXPECT_EQ(data(probes)[0].hit, 42);
    auto camera_depth = view(device, depth, DEVICE::index_t(0));
    auto row = view(device, camera_depth, DEVICE::index_t(1));
    std::vector<float> row_values;
    golden::copy_out(renderer.device, device, row, row_values);
    for(float value : row_values) {
        EXPECT_NEAR(value, 2.25f, 1e-5f);
    }
    EXPECT_EQ(data(rgb)[0], 0xff00ff00u);
    EXPECT_EQ(data(probes)[0].hit, 42);

    camera_values[0].pos[0] = .75f;
    golden::copy_in(device, renderer.device, camera_values.data(), cameras(device, renderer));
    render_launch(device, renderer);
    render_sync(device, renderer);
    for(DEVICE::index_t pixel = 0; pixel < SPEC::CAM_PIXELS; pixel++) {
        EXPECT_NEAR(data(depth)[pixel], 1.75f, 1e-5f);
    }
    EXPECT_EQ(data(probes)[0].hit, 42);
    probe_sync(device, renderer);
    EXPECT_EQ(data(probes)[0].hit, 1);
    EXPECT_NEAR(data(probes)[0].distance, 2.25f, 1e-5f);
    free(device, renderer);
}
#endif
