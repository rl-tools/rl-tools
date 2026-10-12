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
            const auto expected = status[index] == 0 ? rrt::detail::flow_to_rgba(flow[index * 2], flow[index * 2 + 1], FlowParameters::SCALE) : rrt::detail::flow_status_to_rgba(status[index]);
            expect_color_near(color[index], expected);
            EXPECT_EQ(status_color[index], rrt::detail::flow_status_to_rgba(status[index]));
        }
    }
    template <typename SPEC>
    void check_joint(DEVICE& device, rrt::Renderer<SPEC, BACKEND>& renderer, unsigned int state) {
        const auto depth = read(device, renderer, outputs::Depth<>{});
        const auto depth_color = read(device, renderer, DepthColor{});
        const auto normals = read(device, renderer, outputs::Normals<>{});
        const auto normal_color = read(device, renderer, outputs::Normals<outputs::format::RGBA8>{});
        const auto ids = read(device, renderer, outputs::Segmentation<>{});
        const auto id_color = read(device, renderer, outputs::Segmentation<outputs::format::RGBA8>{});
        const auto classes = read(device, renderer, outputs::Segmentation<outputs::format::UInt32, outputs::labels::Semantic>{});
        const auto class_color = read(device, renderer, outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>{});
        const auto rgb = read(device, renderer, outputs::RGB<>{});
        const auto rgb_float = read(device, renderer, outputs::RGB<outputs::format::Float32>{});
        for(size_t index = 0; index < depth.size(); index++) {
            EXPECT_NEAR(depth[index], 3.0f, 1e-4f);
            const uint32_t shade = uint32_t(std::clamp(depth[index] / DepthParameters::SCALE, 0.0f, 1.0f) * 255.0f);
            expect_color_near(depth_color[index], 0xff000000u | shade | (shade << 8) | (shade << 16));
            EXPECT_EQ(normal_color[index], rrt::detail::normal_to_rgba(normals.data() + index * 3));
            EXPECT_EQ(ids[index], 0u);
            EXPECT_EQ(classes[index], 3u);
            EXPECT_EQ(id_color[index], rrt::detail::segmentation_id_to_rgba(ids[index]));
            EXPECT_EQ(class_color[index], rrt::detail::segmentation_id_to_rgba(classes[index]));
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
                EXPECT_EQ(colors[pixel], rrt::detail::normal_to_rgba(normals.data() + pixel * 3)) << component << ":" << neighbor;
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
