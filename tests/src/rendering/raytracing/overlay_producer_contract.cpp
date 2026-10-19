#include <rl_tools/operations/cpu.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_PRODUCER_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif
#include <rl_tools/rendering/datasets/operations_cpu.h>

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

// Overlay pose ownership contract, per backend: the transforms/transforms_motion tensors are the
// single source of truth consumed unconditionally at update(), so a producer write (set_pose on a
// pose sink over the resident tensors, here staged through a host copy of them, with no pending
// row — the GPU-resident training path) must render identically to the host set_pose verb.
// Pending rows only arbitrate which side owns a row between updates: a verb leaves its rows
// pending until the next update publishes them, so producer writes are valid exactly after an
// update has consumed them — the sequence below encodes that rule.

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;

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
        using OUTPUTS = rro::List<rro::RGB<>, rro::Depth<>, rro::Probes>;
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
            const bool first_miss = rlt::is_miss(first.depth[pixel]);
            const bool second_miss = rlt::is_miss(second.depth[pixel]);
            if(first_miss || second_miss){
                counts.depth += first_miss != second_miss;
            }
            else{
                counts.depth += std::fabs(first.depth[pixel] - second.depth[pixel]) > depth_epsilon;
            }
        }
        return counts;
    }

    template <typename SPEC, typename RENDERER>
    struct Harness {
        DEVICE device;
        rlt::rendering::Bundle<float> bundle;
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
            rlt::add(device, bundle.scene, background, background_pose.data());
            rlt::rendering::datasets::compute_bounds(device, bundle);
            rlt::rendering::raytracing::Object dynamic;
            dynamic.name = "dynamic";
            dynamic.meshes.push_back(make_quad(0.6f, 0.8f, 0.1f, 0.1f));
            asset = rlt::add(device, pool, dynamic);

            rlt::malloc(device, renderer);
            rlt::generate_probe_directions(device, renderer);
            rlt::init(device, renderer, bundle, pool);
            set_camera();
            rlt::attach(device, renderer, (TI)0, rlt::rendering::raytracing::OverlayIndex{0});
            placement = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, asset, rlt::make_pose(POSE_A.data()));
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
            golden::copy_out(renderer.device, device, rlt::output(device, renderer, rro::RGB<>{}), frame.rgb);
            golden::copy_out(renderer.device, device, rlt::output(device, renderer, rro::Depth<>{}), frame.depth);
            return frame;
        }

        // a device producer's write, through a host copy of the resident tensors
        template <typename... POSES>
        void produce(const POSES&... poses){
            std::vector<float> slab, motion;
            golden::copy_out(renderer.device, device, renderer.transforms, slab);
            rlt::rendering::raytracing::PoseSink<SPEC> sink;
            sink.transforms = slab.data();
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                golden::copy_out(renderer.device, device, renderer.transforms_motion, motion);
                sink.transforms_motion = motion.data();
            }
            sink.parts = renderer.asset_parts.data();
            rlt::set_pose(device, sink, rlt::rendering::raytracing::OverlayIndex{0}, placement, poses...);
            golden::copy_in(device, renderer.device, slab.data(), renderer.transforms);
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                golden::copy_in(device, renderer.device, motion.data(), renderer.transforms_motion);
            }
        }
    };
}

TEST(RL_TOOLS_OVERLAY_PRODUCER_SUITE, POSE_TENSOR_WRITE_EQUALS_VERB){
    using namespace producer_contract;
    using SPEC = rlt::rendering::raytracing::Specification<PoseConfig>;
    using RENDERER = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;
    Harness<SPEC, RENDERER> harness;

    const Frame baseline = harness.capture();

    rlt::set_pose(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, rlt::make_pose(POSE_B.data()));
    const Frame verb = harness.capture();
    const auto move = difference(baseline, verb);
    ASSERT_GT(move.rgb, MIN_MOVE_CHANGED_PIXELS);
    ASSERT_GT(move.depth, MIN_MOVE_CHANGED_PIXELS);

    rlt::set_pose(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, rlt::make_pose(POSE_A.data()));
    const Frame reset = harness.capture();
    const auto reset_difference = difference(baseline, reset);
    EXPECT_EQ(reset_difference.rgb, 0u) << "verb-driven render is not deterministic";
    EXPECT_EQ(reset_difference.depth, 0u) << "verb-driven render is not deterministic";

    harness.produce(rlt::make_pose(POSE_B.data()));
    const Frame producer = harness.capture();

    const auto equivalence = difference(verb, producer);
    EXPECT_EQ(equivalence.rgb, 0u) << "producer write did not take effect like the set_pose verb";
    EXPECT_EQ(equivalence.depth, 0u) << "producer write did not take effect like the set_pose verb";
    const auto visibility = difference(baseline, producer);
    EXPECT_GT(visibility.rgb, MIN_MOVE_CHANGED_PIXELS) << "producer frame did not move away from the baseline (stale transforms)";
}

TEST(RL_TOOLS_OVERLAY_PRODUCER_SUITE, MOTION_TENSOR_WRITE_EQUALS_VERB){
    using namespace producer_contract;
    using SPEC = rlt::rendering::raytracing::Specification<PairConfig>;
    using RENDERER = rlt::rendering::raytracing::Renderer<SPEC, BACKEND>;
    Harness<SPEC, RENDERER> harness;

    const Frame baseline = harness.capture();

    rlt::set_pose(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, rlt::make_pose(POSE_B.data()), rlt::make_pose(POSE_C.data()));
    const Frame verb = harness.capture();
    const auto move = difference(baseline, verb);
    ASSERT_GT(move.rgb, MIN_MOVE_CHANGED_PIXELS);
    ASSERT_GT(move.depth, MIN_MOVE_CHANGED_PIXELS);

    rlt::set_pose(harness.device, harness.renderer, rlt::rendering::raytracing::OverlayIndex{0}, harness.placement, rlt::make_pose(POSE_A.data()), rlt::make_pose(POSE_A.data()));
    const Frame reset = harness.capture();
    const auto reset_difference = difference(baseline, reset);
    EXPECT_EQ(reset_difference.rgb, 0u) << "verb-driven render is not deterministic";
    EXPECT_EQ(reset_difference.depth, 0u) << "verb-driven render is not deterministic";

    harness.produce(rlt::make_pose(POSE_B.data()), rlt::make_pose(POSE_C.data()));
    const Frame producer = harness.capture();

    const auto equivalence = difference(verb, producer);
    EXPECT_EQ(equivalence.rgb, 0u) << "producer motion write did not take effect like the set_pose verb";
    EXPECT_EQ(equivalence.depth, 0u) << "producer motion write did not take effect like the set_pose verb";
    const auto visibility = difference(baseline, producer);
    EXPECT_GT(visibility.rgb, MIN_MOVE_CHANGED_PIXELS) << "producer frame did not move away from the baseline (stale transforms)";
}

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX) && defined(RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_PRODUCER_ACTIVE_BACKEND)
namespace producer_contract {
    struct StructureConfig: PairConfig {
        static constexpr TI NUM_OVERLAYS = 2;
    };
}

TEST(RL_TOOLS_OVERLAY_PRODUCER_SUITE, MOTION_STRUCTURE_WRITE_WITHOUT_UPDATE){
    using namespace rl_tools;
    using namespace producer_contract;
    using rendering::raytracing::OverlayIndex;
    using SPEC = rendering::raytracing::Specification<StructureConfig>;
    DEVICE device;
    init(device);
    rendering::Bundle<float> bundle;
    add(device, bundle.scene, make_quad(10, 0.2f, 0.2f, 0.2f, 8));
    rendering::datasets::compute_bounds(device, bundle);
    rendering::raytracing::AssetPool pool;
    const auto red = add(device, pool, make_quad(0.6f, 1, 0, 0, 3));
    const auto green = add(device, pool, make_quad(1, 0, 1, 0, 2));
    rendering::raytracing::Renderer<SPEC, BACKEND> renderer;
    malloc(device, renderer);
    init(device, renderer, bundle, pool);
    generate_probe_directions(device, renderer);
    spawn(device, renderer, OverlayIndex{0}, red);
    spawn(device, renderer, OverlayIndex{1}, green);
    attach(device, renderer, TI{0}, OverlayIndex{0});
    const T position[3] = {0, 0, 0}, target[3] = {1, 0, 0}, up[3] = {0, 0, 1};
    const auto camera = make_camera_data(position, target, up, T{80}, T{1});
    golden::copy_in(device, renderer.device, &camera, cameras(device, renderer));
    golden::copy_in(device, renderer.device, &camera, cameras_open(device, renderer));
    update(device, renderer);
    auto capture = [&](){
        render_launch(device, renderer);
        probe_launch(device, renderer);
        synchronize(device, renderer);
        Frame frame;
        golden::copy_out(renderer.device, device, output(device, renderer, rro::RGB<>{}), frame.rgb);
        golden::copy_out(renderer.device, device, output(device, renderer, rro::Depth<>{}), frame.depth);
        return frame;
    };
    const auto before = capture();
    using Structure = rendering::raytracing::backends::optix::OverlaySlotStructure;
    Structure first{}, second{};
    ASSERT_EQ(cudaMemcpy(&first, overlay_structure(device, renderer), sizeof(first), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&second, overlay_structure(device, renderer) + SPEC::MAX_OVERLAY_INSTANCES, sizeof(second), cudaMemcpyDeviceToHost), cudaSuccess);
    first.object = second.object;
    ASSERT_EQ(cudaMemcpyAsync(overlay_structure(device, renderer), &first, sizeof(first), cudaMemcpyHostToDevice), cudaSuccess);
    const auto replaced = capture();
    constexpr TI CENTER = SPEC::CAM_HEIGHT / 2 * SPEC::CAM_WIDTH + SPEC::CAM_WIDTH / 2;
    EXPECT_NE(before.rgb[CENTER], replaced.rgb[CENTER]);
    EXPECT_LT(replaced.depth[CENTER], before.depth[CENTER] - 0.9f);
    first.active = 0;
    ASSERT_EQ(cudaMemcpyAsync(overlay_structure(device, renderer), &first, sizeof(first), cudaMemcpyHostToDevice), cudaSuccess);
    const auto inactive = capture();
    EXPECT_GT(inactive.depth[CENTER], 7.0f);
    first.active = 1;
    ASSERT_EQ(cudaMemcpyAsync(overlay_structure(device, renderer), &first, sizeof(first), cudaMemcpyHostToDevice), cudaSuccess);
    const auto restored = capture();
    EXPECT_EQ(replaced.rgb, restored.rgb);
    EXPECT_EQ(replaced.depth, restored.depth);
    free(device, renderer);
}
#endif
