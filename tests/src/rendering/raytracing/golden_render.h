#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_RENDER_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_RENDER_H

#include "golden_cases.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace golden {
    template <typename T>
    struct Rendered {
        std::vector<uint32_t> frame_buffer;
        std::vector<T> depth_buffer;
        std::vector<T> normals; // 3 per pixel, raw float output (encoding happens at write/compare)
        std::vector<rl_tools::rendering::raytracing::CollisionResult> probes;
        T max_depth = 0;
        T camera_radius = 0;
    };

    // T_CAMERA_MOTION distinguishes the combined case (moving camera + moving overlay) from the
    // object-only isolation case (static camera, shutter-open == shutter-close)
    template <typename SPEC, typename BACKEND, typename DEVICE, typename CASES, bool T_CAMERA_MOTION = true>
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
        rl_tools::rendering::raytracing::AssetPool pool;
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            rl_tools::rendering::raytracing::Mesh mesh;
            mesh.color[0] = 0.8f;
            mesh.color[1] = 0.35f;
            mesh.color[2] = 0.15f;
            for(TI vertex_i = 0; vertex_i < (TI)rl_tools::rendering::raytracing::constants::NUM_VERTICES; vertex_i++) {
                for(TI dim_i = 0; dim_i < 3; dim_i++) {
                    mesh.vertices.push_back(rl_tools::rendering::raytracing::constants::default_vertices[vertex_i][dim_i] * CASES::OVERLAY_HALF_EXTENT);
                }
            }
            for(TI triangle_i = 0; triangle_i < (TI)rl_tools::rendering::raytracing::constants::NUM_INDICES; triangle_i++) {
                for(TI corner_i = 0; corner_i < 3; corner_i++) {
                    mesh.indices.push_back(rl_tools::rendering::raytracing::constants::default_indices[triangle_i][corner_i]);
                }
            }
            rl_tools::add(device, pool, mesh);
            rl_tools::init(device, renderer, scene, pool);
        }
        else {
            rl_tools::init(device, renderer, scene);
        }

        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        std::vector<rl_tools::rendering::raytracing::Camera<T>> camera_staging(SPEC::NUM_CAMERAS);
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const Pose<T>& pose = CASES::POSES[camera_i];
            camera_staging[camera_i] = rl_tools::make_camera_data(pose.position, pose.look_at, pose.up, SPEC::COS_FOVY, aspect);
        }
        rl_tools::copy_to_renderer(device, renderer, camera_staging.data(), rl_tools::data(rl_tools::cameras(device, renderer)), camera_staging.size());
        if constexpr(SPEC::ENABLE_MOTION_BLUR) {
            if constexpr(T_CAMERA_MOTION) {
                for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                    const Pose<T>& pose = CASES::POSES[camera_i];
                    T position[3], look_at[3];
                    for(TI dim_i = 0; dim_i < 3; dim_i++) {
                        position[dim_i] = pose.position[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                        look_at[dim_i] = pose.look_at[dim_i] - CASES::MOTION_BLUR_DELTA[dim_i];
                    }
                    camera_staging[camera_i] = rl_tools::make_camera_data(position, look_at, pose.up, SPEC::COS_FOVY, aspect);
                }
            }
            rl_tools::copy_to_renderer(device, renderer, camera_staging.data(), rl_tools::data(rl_tools::cameras_open(device, renderer)), camera_staging.size());
        }
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                rl_tools::attach(device, renderer, camera_i, rl_tools::rendering::raytracing::OverlayIndex{0});
            }
            const auto spin_pose = [](const typename CASES::T position[3], typename CASES::T radians, float out[12]){
                const float half = (float)radians / 2.0f;
                const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
                const float position_float[3] = {(float)position[0], (float)position[1], (float)position[2]};
                rl_tools::make_transform(position_float, quaternion_wxyz, out);
            };
            float close_transform[12];
            spin_pose(CASES::OVERLAY_POSITION_CLOSE, CASES::OVERLAY_SPIN_CLOSE, close_transform);
            const auto placement = rl_tools::spawn(device, renderer, rl_tools::rendering::raytracing::OverlayIndex{0}, rl_tools::rendering::raytracing::AssetHandle{0}, close_transform);
            if constexpr(SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                float open_transform[12];
                spin_pose(CASES::OVERLAY_POSITION_OPEN, CASES::OVERLAY_SPIN_OPEN, open_transform);
                rl_tools::set_transform_pair(device, renderer, rl_tools::rendering::raytracing::OverlayIndex{0}, placement, open_transform, close_transform);
            }
            rl_tools::update(device, renderer);
        }
        rl_tools::generate_probe_directions(device, renderer);
        rl_tools::render(device, renderer);
        rl_tools::probe(device, renderer);
        rl_tools::synchronize(device, renderer);

        constexpr size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        if constexpr(SPEC::HAS_RGB) {
            out.frame_buffer.resize(pixel_count);
            rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::frame_buffer(device, renderer)), out.frame_buffer.data(), pixel_count);
        }
        if constexpr(SPEC::HAS_DEPTH) {
            out.depth_buffer.resize(pixel_count);
            rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::depth_buffer(device, renderer)), out.depth_buffer.data(), pixel_count);
        }
        if constexpr(SPEC::HAS_NORMALS) {
            out.normals.resize(pixel_count * 3);
            rl_tools::copy_from_renderer(device, renderer, rl_tools::data(rl_tools::normals_buffer(device, renderer)), out.normals.data(), out.normals.size());
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
