#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_RENDER_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_RENDER_H

#include "golden_cases.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace golden {
    template <typename T>
    struct Rendered {
        std::vector<uint32_t> frame_buffer;
        std::vector<T> depth_buffer;
        std::vector<rl_tools::rendering::raytracing::CollisionResult> probes;
        T max_depth = 0;
        T camera_radius = 0;
    };

    template <typename SPEC, typename BACKEND, typename DEVICE, typename CASES>
    bool render_case(DEVICE& device, const std::string& scene_path, Rendered<typename CASES::T>& out) {
        using T = typename CASES::T;
        using TI = typename CASES::TI;
        using Renderer = rl_tools::rendering::raytracing::Renderer<SPEC, BACKEND>;

        out = {};
        Renderer renderer;
        rl_tools::malloc(device, renderer);
        rl_tools::rendering::raytracing::Scene scene;
        if(!rl_tools::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, scene, scene_path)) {
            rl_tools::free(device, renderer);
            return false;
        }
        rl_tools::init(device, renderer, scene);

        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        std::vector<rl_tools::rendering::raytracing::Camera<T>> camera_staging(SPEC::NUM_CAMERAS);
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const Pose<T>& pose = CASES::POSES[camera_i];
            camera_staging[camera_i] = rl_tools::make_camera_data(pose.position, pose.look_at, pose.up, SPEC::COS_FOVY, aspect);
        }
        rl_tools::copy_to_renderer(device, renderer, camera_staging.data(), rl_tools::data(rl_tools::cameras(device, renderer)), camera_staging.size());
        if constexpr(SPEC::ENABLE_MOTION_BLUR) {
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                const Pose<T>& pose = CASES::POSES[camera_i];
                T position[3], look_at[3];
                for(TI dim_i = 0; dim_i < 3; dim_i++) {
                    position[dim_i] = pose.position[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                    look_at[dim_i] = pose.look_at[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                }
                camera_staging[camera_i] = rl_tools::make_camera_data(position, look_at, pose.up, SPEC::COS_FOVY, aspect);
            }
            rl_tools::copy_to_renderer(device, renderer, camera_staging.data(), rl_tools::data(rl_tools::cameras_open(device, renderer)), camera_staging.size());
        }
        rl_tools::generate_probe_directions(device, renderer);
        rl_tools::render(device, renderer);
        rl_tools::probe(device, renderer);
        rl_tools::synchronize(device, renderer);

        constexpr size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        out.frame_buffer.resize(pixel_count);
        rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::frame_buffer(device, renderer)), out.frame_buffer.data(), pixel_count);
        if constexpr(SPEC::HAS_DEPTH) {
            out.depth_buffer.resize(pixel_count);
            rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::depth_buffer(device, renderer)), out.depth_buffer.data(), pixel_count);
        }
        if(rl_tools::data(renderer.collision_results) != nullptr) {
            out.probes.resize((size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
            rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::collision_results(device, renderer)), out.probes.data(), out.probes.size());
        }
        out.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * T{2} : static_cast<T>(1e30);
        out.camera_radius = renderer.camera_radius;

        rl_tools::free(device, renderer);
        return true;
    }
}

#endif
