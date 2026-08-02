// Renders the golden cases (tests/src/rendering/raytracing/golden_cases.h) and compares against
// the OptiX golden renderings in tests/data/rendering_raytracing_golden (see AGENTS.md
// "Raytracing Golden Renderings"). Two targets are compiled from this file:
// - test_rendering_raytracing_golden_comparison_cpu: pinned to the generic (CPU) backend (included
//   directly, not via the mux) so the CPU raytracer is tested in every build configuration.
// - test_rendering_raytracing_golden_comparison_<metal|optix>: uses the mux, i.e. the backend the
//   build is configured for (target name matches, e.g. _metal on macOS, _optix on CUDA machines).
#include <rl_tools/operations/cpu.h>
#if defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#else
#include <rl_tools/rendering/raytracing/backends/generic/operations_cpu.h>
#endif

#if !defined(RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_CPU
#define RL_TOOLS_GOLDEN_BACKEND_NAME "cpu"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_METAL
#define RL_TOOLS_GOLDEN_BACKEND_NAME "metal"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_OPTIX
#define RL_TOOLS_GOLDEN_BACKEND_NAME "optix"
#else
#define RL_TOOLS_GOLDEN_SUITE RENDERING_RAYTRACING_GOLDEN_GENERIC
#define RL_TOOLS_GOLDEN_BACKEND_NAME "generic"
#endif

#include "golden_cases.h"
#include "golden_io.h"
#include "../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// falls back to the CWD-relative convention (run from the repo root) like other targets
#ifdef RL_TOOLS_TEST_DATA_PATH
#define RL_TOOLS_GOLDEN_TEST_DATA_PATH RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)
#else
#define RL_TOOLS_GOLDEN_TEST_DATA_PATH "tests/data"
#endif

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using CASES = golden::Cases<T, TI>;

// Tolerances for CPU-vs-OptiX-golden comparison. The goldens come from a different machine and RT
// implementation; per AGENTS.md, silhouette pixels may land on different surfaces, so all metrics
// are tolerance-based (measured generic-vs-Metal baseline: RGB MAD ~0.03/255, probes exact).
static constexpr double RGB_MAD_THRESHOLD = 1.0;          // mean abs diff in 8-bit levels
static constexpr double DEPTH_MAD_THRESHOLD = 1e-3;       // mean abs diff normalized by max_depth
static constexpr double DEPTH_OUTLIER_REL = 1e-3;         // per-pixel relative outlier threshold
static constexpr double DEPTH_OUTLIER_FRACTION = 0.005;   // max fraction of outlier pixels
static constexpr int PROBE_HIT_MISMATCH_MAX = 6;        // out of NUM_CAMERAS * NUM_PROBES = 768
static constexpr double PROBE_DISTANCE_REL = 1e-3;

static const std::string SCENE_PATH = RL_TOOLS_GOLDEN_TEST_DATA_PATH "/ProcTHOR-Train-1.glb";
static const std::string GOLDEN_DIR = RL_TOOLS_GOLDEN_TEST_DATA_PATH "/rendering_raytracing_golden";
// Frames rendered by the backend under test are written here (gitignored) for visual comparison
// against the goldens.
static const std::string BACKEND_OUTPUT_DIR = GOLDEN_DIR + "/backend/" + RL_TOOLS_GOLDEN_BACKEND_NAME;

namespace {
    bool goldens_available(){
        // per-pose layout: <golden_dir>/<pose_id>/<case>.png
        return std::filesystem::exists(GOLDEN_DIR + "/" + CASES::POSES[0].id);
    }

    struct Rendered{
        std::vector<uint32_t> frame_buffer;   // camera-major
        std::vector<float> depth_buffer;      // camera-major
        std::vector<rlt::rendering::raytracing::CollisionResult> probes;
        float max_depth = 0;
        float camera_radius = 0;
    };

    template <typename SPEC>
    bool render_case(DEVICE& device, Rendered& out){
        using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;
        Renderer renderer;
        rlt::malloc(device, renderer);
        if(!rlt::load_model(device, renderer, SCENE_PATH)){
            rlt::free(device, renderer);
            return false;
        }
        rlt::upload_geometry(device, renderer);

        // camera setup mirrors tests/src/rendering/raytracing/generate_golden.cpp — keep in sync
        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const golden::Pose<T>& pose = CASES::POSES[camera_i];
            rlt::set(device, renderer.cameras, rlt::make_camera_data(pose.position, pose.look_at, pose.up, SPEC::COS_FOVY, aspect), camera_i);
        }
        if constexpr(SPEC::ENABLE_MOTION_BLUR){
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
                const golden::Pose<T>& pose = CASES::POSES[camera_i];
                T position[3], look_at[3];
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    position[dim_i] = pose.position[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                    look_at[dim_i] = pose.look_at[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                }
                rlt::set(device, renderer.cameras_open, rlt::make_camera_data(position, look_at, pose.up, SPEC::COS_FOVY, aspect), camera_i);
            }
            rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
        }
        else{
            rlt::set_cameras(device, renderer, renderer.cameras);
        }
        rlt::generate_probe_directions(device, renderer);
        rlt::build_pipeline(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);

        constexpr size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
        out.frame_buffer.assign(rlt::data(renderer.frame_buffer), rlt::data(renderer.frame_buffer) + pixel_count);
        if constexpr(SPEC::HAS_DEPTH){
            rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
            out.depth_buffer.assign(rlt::data(renderer.depth_buffer), rlt::data(renderer.depth_buffer) + pixel_count);
        }
        const auto* probe_results = rlt::read_collision_results_raw(device, renderer);
        if(probe_results != nullptr){
            out.probes.assign(probe_results, probe_results + (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
        }
        out.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        out.camera_radius = renderer.camera_radius;

        rlt::free(device, renderer);
        return true;
    }

    struct RGBStats{
        double mad = 0;
        int channel_max[3] = {0, 0, 0};
    };

    RGBStats compare_rgb(const uint32_t* ours, const uint32_t* golden, size_t count){
        RGBStats stats;
        const auto* ours_bytes = (const unsigned char*)ours;
        const auto* golden_bytes = (const unsigned char*)golden;
        double total = 0;
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            for(int channel = 0; channel < 3; channel++){
                const int diff = (int)ours_bytes[pixel_i * 4 + channel] - (int)golden_bytes[pixel_i * 4 + channel];
                const int abs_diff = diff < 0 ? -diff : diff;
                if(abs_diff > stats.channel_max[channel]) stats.channel_max[channel] = abs_diff;
                total += abs_diff;
            }
        }
        stats.mad = total / ((double)count * 3);
        return stats;
    }

    template <typename SPEC>
    void write_backend_frames(const char* name, const Rendered& rendered){
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const std::string directory = BACKEND_OUTPUT_DIR + "/" + CASES::POSES[camera_i].id;
            std::filesystem::create_directories(directory);
            golden::write_camera_png(directory + "/" + name + ".png", rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            if(!rendered.depth_buffer.empty()){
                golden::write_camera_depth_png(directory + "/" + name + "_depth.png", rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.max_depth);
            }
        }
    }

    template <typename SPEC>
    void expect_rgb_matches_golden(const char* name, const Rendered& rendered){
        double worst_mad = 0;
        const char* worst_id = CASES::POSES[0].id;
        int max_channel[3] = {0, 0, 0};
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<uint32_t> golden_pixels;
            ASSERT_TRUE(golden::load_camera_png(GOLDEN_DIR + "/" + id + "/" + name + ".png", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_pixels)) << "failed to load golden: " << id << "/" << name;
            const RGBStats stats = compare_rgb(rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, golden_pixels.data(), SPEC::CAM_PIXELS);
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                golden::write_camera_diff_png(directory + "/" + name + "_diff.png", rendered.frame_buffer.data() + camera_i * SPEC::CAM_PIXELS, golden_pixels.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            EXPECT_LE(stats.mad, RGB_MAD_THRESHOLD)
                << name << " pose " << id << ": RGB mean abs diff " << stats.mad
                << " (max R=" << stats.channel_max[0] << " G=" << stats.channel_max[1] << " B=" << stats.channel_max[2] << ")";
            if(stats.mad > worst_mad){
                worst_mad = stats.mad;
                worst_id = id;
            }
            for(int channel = 0; channel < 3; channel++){
                if(stats.channel_max[channel] > max_channel[channel]) max_channel[channel] = stats.channel_max[channel];
            }
        }
        std::printf("[golden] %s: worst pose %s rgb mad=%.4f max=(%d,%d,%d) over %d poses\n", name, worst_id, worst_mad, max_channel[0], max_channel[1], max_channel[2], (int)SPEC::NUM_CAMERAS);
    }

    template <typename SPEC>
    void run_rgb_case(const char* name){
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE(render_case<SPEC>(device, rendered)) << "failed to load scene: " << SCENE_PATH;
        write_backend_frames<SPEC>(name, rendered);
        expect_rgb_matches_golden<SPEC>(name, rendered);
    }

    template <typename SPEC>
    void run_rgbd_case(const char* name){
        DEVICE device;
        rlt::init(device);
        Rendered rendered;
        ASSERT_TRUE(render_case<SPEC>(device, rendered)) << "failed to load scene: " << SCENE_PATH;
        write_backend_frames<SPEC>(name, rendered);
        expect_rgb_matches_golden<SPEC>(name, rendered);

        double worst_mad = 0;
        double worst_outliers = 0;
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
            const char* id = CASES::POSES[camera_i].id;
            std::vector<float> golden_depth;
            ASSERT_TRUE(golden::load_camera_depth_bin(GOLDEN_DIR + "/" + id + "/" + name + "_depth.bin", SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, golden_depth)) << "failed to load golden depth: " << id << "/" << name;
            const float* ours = rendered.depth_buffer.data() + camera_i * SPEC::CAM_PIXELS;
            {
                const std::string directory = BACKEND_OUTPUT_DIR + "/" + id;
                std::filesystem::create_directories(directory);
                golden::write_camera_depth_diff_png(directory + "/" + name + "_depth_diff.png", ours, golden_depth.data(), SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
            }
            double total_abs_diff = 0;
            size_t outliers = 0;
            for(size_t pixel_i = 0; pixel_i < golden_depth.size(); pixel_i++){
                const double diff = (double)ours[pixel_i] - (double)golden_depth[pixel_i];
                const double abs_diff = diff < 0 ? -diff : diff;
                total_abs_diff += abs_diff;
                const double magnitude = std::max((double)golden_depth[pixel_i], 1e-6);
                if(abs_diff / magnitude > DEPTH_OUTLIER_REL) outliers++;
            }
            const double depth_mad = total_abs_diff / (double)golden_depth.size() / (double)rendered.max_depth;
            const double outlier_fraction = (double)outliers / (double)golden_depth.size();
            EXPECT_LE(depth_mad, DEPTH_MAD_THRESHOLD) << name << " pose " << id << ": normalized depth mean abs diff " << depth_mad;
            EXPECT_LE(outlier_fraction, DEPTH_OUTLIER_FRACTION) << name << " pose " << id << ": depth outlier fraction " << outlier_fraction;
            worst_mad = std::max(worst_mad, depth_mad);
            worst_outliers = std::max(worst_outliers, outlier_fraction);
        }
        std::printf("[golden] %s: worst depth mad=%.2e worst outliers=%.4f%%\n", name, worst_mad, worst_outliers * 100);
    }
}

#define RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE() if(!goldens_available()){ GTEST_SKIP() << "golden renderings not found at " << GOLDEN_DIR << " (per-pose layout; generate with test_rendering_raytracing_generate_golden on a CUDA machine)"; }

TEST(RL_TOOLS_GOLDEN_SUITE, LOW_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::LOW_RGB>("low_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, MEDIUM_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::MEDIUM_RGB>("medium_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB>("high_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, VERY_HIGH_RGB){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::VERY_HIGH_RGB>("very_high_rgb");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_AA2){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_AA2>("high_rgb_aa2");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGB_MB4){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgb_case<CASES::HIGH_RGB_MB4>("high_rgb_mb4");
}
TEST(RL_TOOLS_GOLDEN_SUITE, LOW_RGBD){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgbd_case<CASES::LOW_RGBD>("low_rgbd");
}
TEST(RL_TOOLS_GOLDEN_SUITE, HIGH_RGBD){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    run_rgbd_case<CASES::HIGH_RGBD>("high_rgbd");
}

TEST(RL_TOOLS_GOLDEN_SUITE, PROBES){
    RL_TOOLS_GOLDEN_SKIP_IF_UNAVAILABLE();
    using SPEC = CASES::LOW_RGB; // probes.bin is written during the low_rgb golden case
    DEVICE device;
    rlt::init(device);
    Rendered rendered;
    ASSERT_TRUE(render_case<SPEC>(device, rendered)) << "failed to load scene: " << SCENE_PATH;
    ASSERT_FALSE(rendered.probes.empty());

    int hit_mismatches = 0;
    double max_rel_distance_diff = 0;
    for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++){
        const char* id = CASES::POSES[camera_i].id;
        std::vector<rlt::rendering::raytracing::CollisionResult> golden_probes;
        ASSERT_TRUE(golden::load_camera_probes(GOLDEN_DIR + "/" + id + "/probes.bin", SPEC::NUM_PROBES, golden_probes)) << "failed to load golden probes: " << id;
        const auto* ours = rendered.probes.data() + camera_i * SPEC::NUM_PROBES;
        for(size_t probe_i = 0; probe_i < golden_probes.size(); probe_i++){
            if((ours[probe_i].hit != 0) != (golden_probes[probe_i].hit != 0)){
                hit_mismatches++;
                continue;
            }
            if(golden_probes[probe_i].hit){
                const double diff = (double)ours[probe_i].distance - (double)golden_probes[probe_i].distance;
                const double abs_diff = diff < 0 ? -diff : diff;
                const double rel = abs_diff / std::max((double)golden_probes[probe_i].distance, 1e-6);
                if(rel > max_rel_distance_diff) max_rel_distance_diff = rel;
            }
        }
    }
    EXPECT_LE(hit_mismatches, PROBE_HIT_MISMATCH_MAX) << "probe hit flag mismatches: " << hit_mismatches << "/" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
    EXPECT_LE(max_rel_distance_diff, PROBE_DISTANCE_REL) << "max relative probe distance diff: " << max_rel_distance_diff;
    std::printf("[golden] probes: hit mismatches=%d/%d max rel dist diff=%.2e\n", hit_mismatches, (int)(SPEC::NUM_CAMERAS * SPEC::NUM_PROBES), max_rel_distance_diff);
}
