#include <rl_tools/operations/cpu.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_PRODUCER_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#include "render_copy.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef RL_TOOLS_OVERLAY_PRODUCER_SUITE
#define RL_TOOLS_OVERLAY_PRODUCER_SUITE RENDERING_RAYTRACING_OVERLAY_PRODUCER_GENERIC
#endif

// Overlay pose ownership contract, per backend: the transforms/transforms_pair tensors are the
// single source of truth consumed unconditionally at update(), so a producer-style tensor write
// (rlt::copy with no dirty flag — the GPU-resident training path) must render identically to the
// equivalent set_transform* verb. The dirty flags only arbitrate which side owns a row between
// updates: a verb leaves its row dirty until the next update's flush, so producer writes are
// valid exactly after an update has consumed the flags — the sequence below encodes that rule.

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_PRODUCER_ACTIVE_BACKEND)
using BACKEND = rlt::rendering::raytracing::backends::Default;
#else
using BACKEND = rlt::rendering::raytracing::backends::Generic;
#endif

namespace producer_contract {
    struct PoseConfig: rlt::rendering::raytracing::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = 64;
        static constexpr TI CAM_HEIGHT = 64;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = 1;
        static constexpr T FOV = 80;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr bool OUTPUT_RGB = true;
        static constexpr bool OUTPUT_DEPTH = true;
        static constexpr TI NUM_OVERLAYS = 1;
        static constexpr TI MAX_OVERLAY_INSTANCES = 2;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
    };
    struct PairConfig: PoseConfig{
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 2;
        static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
    };

    using Transform = std::array<float, 12>;

    constexpr Transform pose(float x, float y, float z){
        return {{1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z}};
    }

    static constexpr Transform POSE_A = pose(4.0f, -0.8f, 0.0f);
    static constexpr Transform POSE_B = pose(4.0f, 0.9f, 0.4f);
    static constexpr Transform POSE_C = pose(4.0f, 0.2f, -0.5f);

    static constexpr std::size_t MIN_MOVE_CHANGED_PIXELS = 32;
    // the pair path slerps on-device on OptiX but on the host for the verb staging: same code,
    // but FMA contraction may differ by ulps, so equivalence there is near-exact, not bitwise
    static constexpr std::size_t MAX_EQUIVALENCE_PIXELS = 8;
    static constexpr float DEPTH_EQUIVALENCE_EPSILON = 1e-3f;

    inline rlt::rendering::raytracing::Mesh make_quad(float half_extent, float red, float green, float blue, float x = 0.0f){
        rlt::rendering::raytracing::Mesh mesh;
        mesh.vertices = {
            x, -half_extent, -half_extent,
            x, +half_extent, -half_extent,
            x, +half_extent, +half_extent,
            x, -half_extent, +half_extent,
        };
        mesh.indices = {0, 2, 1, 0, 3, 2};
        mesh.color[0] = red;
        mesh.color[1] = green;
        mesh.color[2] = blue;
        return mesh;
    }

    struct Frame {
        std::vector<std::uint32_t> rgb;
        std::vector<float> depth;
    };

    struct Difference {
        std::size_t rgb = 0;
        std::size_t depth = 0;
    };

    inline Difference difference(const Frame& first, const Frame& second, float depth_epsilon = 0.0f){
        Difference counts;
        for(std::size_t pixel = 0; pixel < first.rgb.size(); pixel++){
            counts.rgb += first.rgb[pixel] != second.rgb[pixel];
            counts.depth += std::fabs(first.depth[pixel] - second.depth[pixel]) > depth_epsilon;
        }
        return counts;
    }

    template <typename SPEC, typename RENDERER>
    struct Harness {
        DEVICE device;
        rlt::rendering::raytracing::Scene scene;
        rlt::rendering::raytracing::AssetPool pool;
        rlt::rendering::raytracing::AssetHandle asset;
        RENDERER renderer;
        rlt::rendering::raytracing::OverlayPlacement placement;

        Harness(){
            rlt::init(device);
            rlt::rendering::raytracing::Object background;
            background.name = "background";
            background.meshes.push_back(make_quad(10.0f, 0.2f, 0.2f, 0.2f));
            const auto background_pose = pose(8.0f, 0.0f, 0.0f);
            rlt::add(device, scene, background, background_pose.data());
            rlt::rendering::raytracing::Object dynamic;
            dynamic.name = "dynamic";
            dynamic.meshes.push_back(make_quad(0.6f, 0.8f, 0.1f, 0.1f));
            asset = rlt::add(device, pool, dynamic);

            rlt::malloc(device, renderer);
            rlt::generate_probe_directions(device, renderer);
            rlt::init(device, renderer, scene, pool);
            set_camera();
            rlt::attach(device, renderer, (TI)0, rlt::rendering::raytracing::OverlayIndex{0});
            placement = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, asset, POSE_A.data());
        }
        ~Harness(){
            rlt::free(device, renderer);
        }

        void set_camera(){
            const T position[3] = {0, 0, 0};
            const T look_at[3] = {1, 0, 0};
            const T up[3] = {0, 0, 1};
            constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
            const auto camera = rlt::make_camera_data(position, look_at, up, (T)SPEC::CONFIG::FOV, aspect);
            std::array<rlt::rendering::raytracing::Camera<T>, SPEC::NUM_CAMERAS> cameras;
            cameras.fill(camera);
            golden::copy_in(device, renderer.device, cameras.data(), rlt::cameras(device, renderer));
            if constexpr (SPEC::HAS_CAMERA_PAIR){
                golden::copy_in(device, renderer.device, cameras.data(), rlt::cameras_open(device, renderer));
            }
        }

        Frame capture(){
            rlt::update(device, renderer);
            rlt::render(device, renderer);
            rlt::synchronize(device, renderer);
            Frame frame;
            golden::copy_out(renderer.device, device, rlt::frame_buffer(device, renderer), frame.rgb);
            golden::copy_out(renderer.device, device, rlt::depth_buffer(device, renderer), frame.depth);
            return frame;
        }

        std::size_t slot_row() const {
            return (std::size_t)0 * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot;
        }
    };
}

TEST(RL_TOOLS_OVERLAY_PRODUCER_SUITE, POSE_TENSOR_WRITE_EQUALS_VERB){
    using namespace producer_contract;
    using SPEC = rlt::rendering::raytracing::Specification<PoseConfig>;
    using RENDERER = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;
    Harness<SPEC, RENDERER> harness;

    const Frame baseline = harness.capture();

    rlt::set_transform(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, POSE_B.data());
    const Frame verb = harness.capture();
    const auto move = difference(baseline, verb);
    ASSERT_GT(move.rgb, MIN_MOVE_CHANGED_PIXELS);
    ASSERT_GT(move.depth, MIN_MOVE_CHANGED_PIXELS);

    rlt::set_transform(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, POSE_A.data());
    const Frame reset = harness.capture();
    const auto reset_difference = difference(baseline, reset);
    EXPECT_EQ(reset_difference.rgb, 0u) << "verb-driven render is not deterministic";
    EXPECT_EQ(reset_difference.depth, 0u) << "verb-driven render is not deterministic";

    // producer path: read-modify-write of the transforms tensor with no dirty flag
    std::vector<float> slab;
    golden::copy_out(harness.renderer.device, harness.device, rlt::transforms(harness.device, harness.renderer), slab);
    std::memcpy(slab.data() + harness.slot_row() * 12, POSE_B.data(), 12 * sizeof(float));
    golden::copy_in(harness.device, harness.renderer.device, slab.data(), rlt::transforms(harness.device, harness.renderer));
    const Frame producer = harness.capture();

    const auto equivalence = difference(verb, producer);
    EXPECT_EQ(equivalence.rgb, 0u) << "producer tensor write did not take effect like the set_transform verb";
    EXPECT_EQ(equivalence.depth, 0u) << "producer tensor write did not take effect like the set_transform verb";
    const auto visibility = difference(baseline, producer);
    EXPECT_GT(visibility.rgb, MIN_MOVE_CHANGED_PIXELS) << "producer frame did not move away from the baseline (stale transforms)";
}

TEST(RL_TOOLS_OVERLAY_PRODUCER_SUITE, PAIR_TENSOR_WRITE_EQUALS_VERB){
    using namespace producer_contract;
    using SPEC = rlt::rendering::raytracing::Specification<PairConfig>;
    using RENDERER = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;
    Harness<SPEC, RENDERER> harness;

    const Frame baseline = harness.capture();

    rlt::set_transform_pair(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, POSE_B.data(), POSE_C.data());
    const Frame verb = harness.capture();
    const auto move = difference(baseline, verb);
    ASSERT_GT(move.rgb, MIN_MOVE_CHANGED_PIXELS);
    ASSERT_GT(move.depth, MIN_MOVE_CHANGED_PIXELS);

    rlt::set_transform_pair(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, POSE_A.data(), POSE_A.data());
    const Frame reset = harness.capture();
    const auto reset_difference = difference(baseline, reset);
    EXPECT_EQ(reset_difference.rgb, 0u) << "verb-driven render is not deterministic";
    EXPECT_EQ(reset_difference.depth, 0u) << "verb-driven render is not deterministic";

    // producer path: write the transforms_pair tensor with no dirty flag and expand on the
    // backend's expansion path (device kernel on OptiX, host loop elsewhere)
    constexpr std::size_t SLOTS = (std::size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
    std::vector<float> pairs(2 * SLOTS * 12);
    for(std::size_t slot = 0; slot < 2 * SLOTS; slot++){
        std::memcpy(pairs.data() + slot * 12, rlt::rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
    }
    std::memcpy(pairs.data() + harness.slot_row() * 12, POSE_B.data(), 12 * sizeof(float));
    std::memcpy(pairs.data() + (SLOTS + harness.slot_row()) * 12, POSE_C.data(), 12 * sizeof(float));
    golden::copy_in(harness.device, harness.renderer.device, pairs.data(), rlt::transforms_pair(harness.device, harness.renderer));
    rlt::expand_motion_transforms(harness.device, harness.renderer);
    const Frame producer = harness.capture();

    const auto equivalence = difference(verb, producer, DEPTH_EQUIVALENCE_EPSILON);
    EXPECT_LE(equivalence.rgb, MAX_EQUIVALENCE_PIXELS) << "producer pair write + expansion did not take effect like the set_transform_pair verb";
    EXPECT_LE(equivalence.depth, MAX_EQUIVALENCE_PIXELS) << "producer pair write + expansion did not take effect like the set_transform_pair verb";
    const auto visibility = difference(baseline, producer);
    EXPECT_GT(visibility.rgb, MIN_MOVE_CHANGED_PIXELS) << "producer frame did not move away from the baseline (stale transforms)";
}
