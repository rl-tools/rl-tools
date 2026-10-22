#ifndef TESTS_HYPERLUX_GOLDEN_RENDER_H
#define TESTS_HYPERLUX_GOLDEN_RENDER_H

#include "golden_cases.h"
#include "render_copy.h"

#include <rl_tools/hyperlux/datasets/glb/operations_cpu.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace golden {
    namespace outputs = rl_tools::hyperlux::outputs;
    template <typename T>
    struct Rendered {
        std::vector<uint32_t> frame_buffer;
        std::vector<T> depth_buffer;
        std::vector<T> normals; // 3 per pixel, raw float output (encoding happens at write/compare)
        std::vector<T> flow;    // 2 per pixel, raw float backward flow in pixels (current -> previous)
        std::vector<uint32_t> flow_status;
        std::vector<T> flow_forward; // 2 per pixel, previous -> current, indexed in the previous camera
        std::vector<uint32_t> flow_forward_status;
        std::vector<uint32_t> segmentation;
        std::vector<rl_tools::hyperlux::CollisionResult> probes;
        std::map<uint32_t, std::string> semantic_classes;
        size_t num_instances = 0;
        T max_depth = 0;
    };

    namespace detail {
        template <typename T>
        void normalize(T vector[3]){
            const T norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
            for(int dim_i = 0; dim_i < 3; dim_i++){
                vector[dim_i] /= norm;
            }
        }

        template <typename T>
        void cross(const T a[3], const T b[3], T out[3]){
            out[0] = a[1] * b[2] - a[2] * b[1];
            out[1] = a[2] * b[0] - a[0] * b[2];
            out[2] = a[0] * b[1] - a[1] * b[0];
        }

        template <typename T>
        void rotate_about_z(T vector[3], T radians){
            const T x = vector[0];
            const T y = vector[1];
            vector[0] = x * std::cos(radians) - y * std::sin(radians);
            vector[1] = x * std::sin(radians) + y * std::cos(radians);
        }

        // the previous (shutter-open) instant of a pose under the camera motion of CASES
        template <typename CASES>
        Pose<typename CASES::T> previous_pose(const Pose<typename CASES::T>& pose){
            using T = typename CASES::T;
            Pose<T> previous = pose;
            T forward[3];
            for(int dim_i = 0; dim_i < 3; dim_i++){
                previous.position[dim_i] = pose.position[dim_i] - CASES::CAMERA_TRANSLATION[dim_i];
                forward[dim_i] = pose.look_at[dim_i] - pose.position[dim_i];
            }
            rotate_about_z(forward, -CASES::CAMERA_YAW);
            rotate_about_z(previous.up, -CASES::CAMERA_YAW);
            for(int dim_i = 0; dim_i < 3; dim_i++){
                previous.look_at[dim_i] = previous.position[dim_i] + forward[dim_i];
            }
            return previous;
        }

        template <typename T>
        void spin_transform(const T position[3], T radians, float out[12]){
            const float half = (float)radians / 2.0f;
            const float quaternion_wxyz[4] = {std::cos(half), 0, 0, std::sin(half)};
            const float position_float[3] = {(float)position[0], (float)position[1], (float)position[2]};
            rl_tools::make_transform(position_float, quaternion_wxyz, out);
        }

        // the private overlay of a pose: in the frustum at PRIVATE_AHEAD, offset along the image
        // axes, and dropping along the vertical axis between the instants
        template <typename CASES>
        void private_overlay_transforms(const Pose<typename CASES::T>& pose, float open[12], float close[12]){
            using T = typename CASES::T;
            T forward[3], horizontal[3], vertical[3];
            for(int dim_i = 0; dim_i < 3; dim_i++){
                forward[dim_i] = pose.look_at[dim_i] - pose.position[dim_i];
            }
            normalize(forward);
            cross(forward, pose.up, horizontal);
            normalize(horizontal);
            cross(horizontal, forward, vertical);
            T position_close[3], position_open[3];
            for(int dim_i = 0; dim_i < 3; dim_i++){
                position_close[dim_i] = pose.position[dim_i] + CASES::PRIVATE_AHEAD * forward[dim_i] + CASES::PRIVATE_ASIDE * horizontal[dim_i] - CASES::PRIVATE_BELOW * vertical[dim_i];
                position_open[dim_i] = position_close[dim_i] + CASES::PRIVATE_TRANSLATION * vertical[dim_i];
            }
            spin_transform(position_open, CASES::PRIVATE_SPIN_OPEN, open);
            spin_transform(position_close, CASES::PRIVATE_SPIN_CLOSE, close);
        }

        template <typename TI>
        rl_tools::hyperlux::Mesh cube_mesh(float half_extent, float red, float green, float blue){
            rl_tools::hyperlux::Mesh mesh;
            mesh.color[0] = red;
            mesh.color[1] = green;
            mesh.color[2] = blue;
            for(TI vertex_i = 0; vertex_i < (TI)rl_tools::hyperlux::constants::NUM_VERTICES; vertex_i++){
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    mesh.vertices.push_back(rl_tools::hyperlux::constants::default_vertices[vertex_i][dim_i] * half_extent);
                }
            }
            for(TI triangle_i = 0; triangle_i < (TI)rl_tools::hyperlux::constants::NUM_INDICES; triangle_i++){
                for(TI corner_i = 0; corner_i < 3; corner_i++){
                    mesh.indices.push_back(rl_tools::hyperlux::constants::default_indices[triangle_i][corner_i]);
                }
            }
            return mesh;
        }

        // fixed overlays sit at the close pose for both instants (spawn initialises previous ==
        // current and every motion sample); moving ones go through set_pose: flow keeps the spawn
        // pose as the previous instant, dynamic motion blur takes the shutter pair
        template <typename CASE, typename DEVICE, typename RENDERER>
        void place_overlay(DEVICE& device, RENDERER& renderer, rl_tools::hyperlux::OverlayIndex overlay, rl_tools::hyperlux::AssetHandle asset, const float open[12], const float close[12]){
            using SPEC = typename CASE::SPEC;
            if constexpr(CASE::OVERLAYS == Overlays::MOVING && SPEC::HAS_FLOW_OVERLAYS){
                const auto placement = rl_tools::spawn(device, renderer, overlay, asset, rl_tools::make_pose(open));
                rl_tools::set_pose(device, renderer, overlay, placement, rl_tools::make_pose(close));
            }
            else if constexpr(CASE::OVERLAYS == Overlays::MOVING){
                const auto placement = rl_tools::spawn(device, renderer, overlay, asset, rl_tools::make_pose(close));
                rl_tools::set_pose(device, renderer, overlay, placement, rl_tools::make_pose(open), rl_tools::make_pose(close));
            }
            else{
                rl_tools::spawn(device, renderer, overlay, asset, rl_tools::make_pose(close));
            }
        }
    }

    template <typename CASE, typename BACKEND, typename DEVICE, typename CASES>
    bool render_case(DEVICE& device, const std::string& scene_path, Rendered<typename CASES::T>& out) {
        using SPEC = typename CASE::SPEC;
        using T = typename CASES::T;
        using TI = typename CASES::TI;
        using Renderer = rl_tools::HyperLux<SPEC, BACKEND>;
        using OverlayIndex = rl_tools::hyperlux::OverlayIndex;
        using AssetHandle = rl_tools::hyperlux::AssetHandle;

        out = {};
        Renderer renderer;
        rl_tools::malloc(device, renderer);
        rl_tools::hyperlux::Bundle<T> bundle;
        using rl_tools::load;
        if constexpr(SPEC::HAS_SEGMENTATION) {
            if(!load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, scene_path, rl_tools::Mode<rl_tools::mode::hyperlux::SceneRoots<>>{})) {
                rl_tools::free(device, renderer);
                return false;
            }
        }
        else if(!load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, scene_path)) {
            rl_tools::free(device, renderer);
            return false;
        }
        out.num_instances = bundle.scene.instances.size();
        if constexpr(SPEC::HAS_SEMANTIC_SEGMENTATION){
            using rl_tools::segmentation_taxonomy;
            const auto taxonomy = segmentation_taxonomy(device, bundle.scene);
            if(!taxonomy){
                rl_tools::free(device, renderer);
                return false;
            }
            out.semantic_classes = taxonomy->classes;
        }
        rl_tools::hyperlux::AssetPool pool;
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            rl_tools::add(device, pool, detail::cube_mesh<TI>((float)CASES::SHARED_HALF_EXTENT, 0.8f, 0.35f, 0.15f));
            rl_tools::add(device, pool, detail::cube_mesh<TI>((float)CASES::PRIVATE_HALF_EXTENT, 0.15f, 0.45f, 0.85f));
            rl_tools::init(device, renderer, bundle, pool);
        }
        else {
            rl_tools::init(device, renderer, bundle);
        }

        constexpr T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        std::vector<rl_tools::hyperlux::Camera<T>> camera_staging(SPEC::NUM_CAMERAS);
        for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
            const Pose<T>& pose = CASES::POSES[camera_i];
            camera_staging[camera_i] = rl_tools::make_camera_data(pose.position, pose.look_at, pose.up, CASES::FOV, aspect);
        }
        copy_in(device, renderer.device, camera_staging.data(), rl_tools::cameras(device, renderer));
        // the shutter-open (motion blur) and previous (flow) cameras share the same pose
        if constexpr(SPEC::HAS_CAMERA_PAIR || SPEC::HAS_FLOW) {
            if constexpr(CASE::CAMERA == Camera::MOVING) {
                for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                    const Pose<T> previous = detail::previous_pose<CASES>(CASES::POSES[camera_i]);
                    camera_staging[camera_i] = rl_tools::make_camera_data(previous.position, previous.look_at, previous.up, CASES::FOV, aspect);
                }
            }
            if constexpr(SPEC::HAS_CAMERA_PAIR) {
                copy_in(device, renderer.device, camera_staging.data(), rl_tools::cameras_open(device, renderer));
            }
            if constexpr(SPEC::HAS_FLOW) {
                copy_in(device, renderer.device, camera_staging.data(), rl_tools::cameras_previous(device, renderer));
            }
        }
        if constexpr(SPEC::ENABLE_OVERLAYS) {
            float open[12], close[12];
            detail::spin_transform(CASES::SHARED_POSITION_OPEN, CASES::SHARED_SPIN_OPEN, open);
            detail::spin_transform(CASES::SHARED_POSITION_CLOSE, CASES::SHARED_SPIN_CLOSE, close);
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                rl_tools::attach(device, renderer, camera_i, OverlayIndex{0});
                rl_tools::attach(device, renderer, camera_i, OverlayIndex{(size_t)(1 + camera_i)});
            }
            detail::place_overlay<CASE>(device, renderer, OverlayIndex{0}, AssetHandle{0}, open, close);
            for(TI camera_i = 0; camera_i < SPEC::NUM_CAMERAS; camera_i++) {
                detail::private_overlay_transforms<CASES>(CASES::POSES[camera_i], open, close);
                detail::place_overlay<CASE>(device, renderer, OverlayIndex{(size_t)(1 + camera_i)}, AssetHandle{1}, open, close);
            }
            rl_tools::update(device, renderer);
        }
        if constexpr(SPEC::HAS_PROBES){
            rl_tools::generate_probe_directions(device, renderer);
        }
        rl_tools::render(device, renderer);
        if constexpr(SPEC::HAS_PROBES){
            rl_tools::probe(device, renderer);
        }
        rl_tools::synchronize(device, renderer);

        if constexpr(SPEC::HAS_RGB) {
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::RGB<>{}), out.frame_buffer);
        }
        if constexpr(SPEC::HAS_DEPTH) {
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::Depth<>{}), out.depth_buffer);
        }
        if constexpr(SPEC::HAS_NORMALS) {
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::Normals<>{}), out.normals);
        }
        if constexpr(SPEC::HAS_FLOW) {
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::Flow<>{}), out.flow);
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::FlowStatus<>{}), out.flow_status);
        }
        if constexpr(SPEC::HAS_FLOW_FORWARD) {
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::Flow<rl_tools::hyperlux::outputs::direction::Forward>{}), out.flow_forward);
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::FlowStatus<rl_tools::hyperlux::outputs::direction::Forward>{}), out.flow_forward_status);
        }
        if constexpr(SPEC::HAS_SEGMENTATION) {
            using LABELS = rl_tools::utils::typing::conditional_t<SPEC::HAS_SEMANTIC_SEGMENTATION, outputs::labels::Semantic, outputs::labels::Instance>;
            copy_out(renderer.device, device, rl_tools::output(device, renderer, outputs::Segmentation<outputs::format::UInt32, LABELS>{}), out.segmentation);
        }
        if constexpr(SPEC::HAS_PROBES){
            copy_out(renderer.device, device, rl_tools::output(device, renderer, rl_tools::hyperlux::outputs::Probes{}), out.probes);
        }
        out.max_depth = renderer.max_ray_length > 0 ? renderer.max_ray_length : static_cast<T>(1e30);

        rl_tools::free(device, renderer);
        return true;
    }
}

#endif
