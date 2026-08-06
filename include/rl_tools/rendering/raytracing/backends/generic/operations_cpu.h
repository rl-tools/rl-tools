#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_GENERIC_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_GENERIC_OPERATIONS_CPU_H

// Host side of the generic (pure software) raytracing backend: implements the common Renderer
// interface on top of the freestanding kernels in operations_generic.h. Rendering is synchronous
// and single-threaded; "device" pointers are host pointers.
#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "operations_generic.h"

#include <vector>
#include <cstring>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::backends::generic{
        template <typename SPEC>
        struct State{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            std::vector<MeshView<T, TI>> meshes; // views into the Scene passed to init (which must outlive rendering)
            std::vector<TI> triangle_mesh;
            std::vector<TI> triangle_local;
            std::vector<ObjectView<T, TI>> objects;
            std::vector<InstanceView<T, TI>> instances;
            std::vector<BVHNode<T, TI>> nodes;   // all BLAS nodes, one slice per object
            std::vector<TI> primitives;          // all BLAS leaf permutations (global triangle ids), one slice per object
            std::vector<BVHNode<T, TI>> tlas_nodes;
            std::vector<TI> tlas_primitives;
            TI num_scene_instances = 0;
            std::vector<OverlayView<T, TI>> overlay_views;
            std::vector<BVHNode<T, TI>> overlay_tlas_nodes;    // one 2*CAP slice per overlay
            std::vector<TI> overlay_tlas_primitives;           // one CAP slice per overlay (global instance ids)
            std::vector<TI> overlay_attachments;
            std::vector<T> overlay_bounds_min;                 // scratch indexed by global instance id
            std::vector<T> overlay_bounds_max;
            std::vector<T> overlay_centroids;
            std::vector<TI> overlay_temp_primitives;
            std::vector<SceneLight> lights;
            std::vector<T> probe_directions;
            std::vector<Camera<T>> cameras;
            std::vector<Camera<T>> cameras_open;
            std::vector<unsigned int> frame_buffer;
            std::vector<float> depth_buffer;
            std::vector<unsigned int> segmentation_buffer;
            std::vector<CollisionResult> collision_results;
            SceneView<T, TI> scene;
        };

        template <typename SPEC>
        State<SPEC>& state(rendering::raytracing::Renderer<SPEC>& renderer){
            return *(State<SPEC>*)renderer.backend.context;
        }

        template <typename SPEC>
        void instance_world_bounds(State<SPEC>& backend_state, const InstanceView<typename SPEC::T, typename SPEC::TI>& instance_view, typename SPEC::T bounds_min[3], typename SPEC::T bounds_max[3]){
            using T = typename SPEC::T;
            for(int axis = 0; axis < 3; axis++){
                bounds_min[axis] = (T)1e30;
                bounds_max[axis] = (T)-1e30;
            }
            const auto& object_view = backend_state.objects[instance_view.object];
            if(object_view.num_nodes == 0) return;
            const auto& root = object_view.nodes[0];
            if(instance_view.identity){
                for(int axis = 0; axis < 3; axis++){
                    bounds_min[axis] = root.bounds_min[axis];
                    bounds_max[axis] = root.bounds_max[axis];
                }
            }
            else{
                for(int corner = 0; corner < 8; corner++){
                    const float local[3] = {
                        (corner & 1) ? (float)root.bounds_max[0] : (float)root.bounds_min[0],
                        (corner & 2) ? (float)root.bounds_max[1] : (float)root.bounds_min[1],
                        (corner & 4) ? (float)root.bounds_max[2] : (float)root.bounds_min[2]
                    };
                    float world[3];
                    rendering::raytracing::detail::transform_point(instance_view.object_to_world, local, world);
                    for(int axis = 0; axis < 3; axis++){
                        bounds_min[axis] = minimum(bounds_min[axis], (T)world[axis]);
                        bounds_max[axis] = maximum(bounds_max[axis], (T)world[axis]);
                    }
                }
            }
        }

        template <typename TI>
        TextureView<TI> texture_view(const rendering::raytracing::Texture& texture){
            TextureView<TI> view;
            if(texture.present()){
                view.pixels = texture.pixels.data();
                view.width = (TI)texture.width;
                view.height = (TI)texture.height;
            }
            return view;
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The generic raytracing backend requires T = float");

        malloc(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            malloc(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            malloc(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            malloc(device, renderer.depth_buffer);
        }
        malloc(device, renderer.collision_results);

        auto* backend_state = new generic::State<SPEC>{};
        backend_state->cameras.resize(SPEC::NUM_CAMERAS);
        backend_state->scene.cameras_close = backend_state->cameras.data();
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            backend_state->cameras_open.resize(SPEC::NUM_CAMERAS);
            backend_state->scene.cameras_open = backend_state->cameras_open.data();
        }
        else {
            backend_state->scene.cameras_open = backend_state->cameras.data();
        }
        if constexpr (SPEC::HAS_RGB) {
            backend_state->frame_buffer.resize((size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            backend_state->scene.frame_buffer = backend_state->frame_buffer.data();
            renderer.backend.frame_buffer_handle = backend_state->frame_buffer.data();
        }
        if constexpr (SPEC::HAS_DEPTH) {
            backend_state->depth_buffer.resize((size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            backend_state->scene.depth_buffer = backend_state->depth_buffer.data();
            renderer.backend.depth_buffer_handle = backend_state->depth_buffer.data();
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            malloc(device, renderer.segmentation_buffer);
            backend_state->segmentation_buffer.resize((size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            backend_state->scene.segmentation_buffer = backend_state->segmentation_buffer.data();
            renderer.backend.segmentation_buffer_handle = backend_state->segmentation_buffer.data();
        }
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            backend_state->overlay_views.resize(SPEC::NUM_OVERLAYS);
            backend_state->overlay_tlas_nodes.resize((size_t)SPEC::NUM_OVERLAYS * 2 * SPEC::MAX_OVERLAY_INSTANCES);
            backend_state->overlay_tlas_primitives.resize((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES);
            backend_state->overlay_temp_primitives.resize(SPEC::MAX_OVERLAY_INSTANCES);
            backend_state->overlay_attachments.resize((size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA, ~(TI)0);
            for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                auto& view = backend_state->overlay_views[overlay];
                view.tlas_nodes = backend_state->overlay_tlas_nodes.data() + (size_t)overlay * 2 * SPEC::MAX_OVERLAY_INSTANCES;
                view.tlas_primitives = backend_state->overlay_tlas_primitives.data() + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
                view.num_tlas_nodes = 0;
            }
            backend_state->scene.overlays = backend_state->overlay_views.data();
            backend_state->scene.num_overlays = SPEC::NUM_OVERLAYS;
            backend_state->scene.attachments = backend_state->overlay_attachments.data();
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        backend_state->collision_results.resize((size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
        backend_state->scene.collision_results = backend_state->collision_results.data();
        renderer.backend.collision_results_buffer = backend_state->collision_results.data();
        renderer.backend.collision_ray_gen = backend_state; // non-null marker: collision rays available
#endif
        renderer.backend.context = backend_state;
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer);

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool){
        namespace generic = rendering::raytracing::backends::generic;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        auto& backend_state = generic::state(renderer);

        rendering::raytracing::detail::compute_scene_bounds(renderer, scene);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << scene.objects.size() << " object(s), " << scene.instances.size() << " instance(s), " << pool.assemblies.size() << " pool asset(s) ...");

        // one combined object list: scene objects first, then pool objects (same global arrays,
        // same BLAS build; pool objects only ever appear through overlay instances)
        std::vector<const rendering::raytracing::Object*> all_objects;
        for(const auto& object : scene.objects){
            all_objects.push_back(&object);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            rendering::raytracing::detail::register_pool_assets(device, renderer, pool, all_objects);
        }

        backend_state.meshes.clear();
        backend_state.triangle_mesh.clear();
        backend_state.triangle_local.clear();
        std::vector<TI> object_first_triangle;
        std::vector<TI> object_triangle_count;
        for(const auto* object_pointer : all_objects){
            const auto& object = *object_pointer;
            object_first_triangle.push_back((TI)backend_state.triangle_mesh.size());
            for(const auto& md : object.meshes){
                const TI mesh_i = (TI)backend_state.meshes.size();
                generic::MeshView<T, TI> view;
                view.indices = md.indices.data();
                view.vertices = md.vertices.data();
                view.tex_coords = md.tex_coords.empty() ? nullptr : md.tex_coords.data();
                view.normals = md.normals.empty() ? nullptr : md.normals.data();
                view.texture = generic::texture_view<TI>(md.texture);
                view.normal_map = generic::texture_view<TI>(md.normal_map);
                view.metallic_roughness_map = generic::texture_view<TI>(md.metallic_roughness_map);
                view.emissive_map = generic::texture_view<TI>(md.emissive_map);
                view.occlusion_map = generic::texture_view<TI>(md.occlusion_map);
                for(int component = 0; component < 3; component++){
                    view.color[component] = md.color[component];
                    view.emissive[component] = md.emissive[component];
                }
                view.metallic = md.metallic;
                view.roughness = md.roughness;
                view.opacity = md.opacity;
                view.alpha_cutoff = md.alpha_cutoff;
                view.alpha_mode = md.alpha_mode;
                backend_state.meshes.push_back(view);

                const TI mesh_triangles = (TI)(md.indices.size() / 3);
                for(TI triangle = 0; triangle < mesh_triangles; triangle++){
                    backend_state.triangle_mesh.push_back(mesh_i);
                    backend_state.triangle_local.push_back(triangle);
                }
            }
            object_triangle_count.push_back((TI)backend_state.triangle_mesh.size() - object_first_triangle.back());
        }
        backend_state.scene.meshes = backend_state.meshes.data();
        backend_state.scene.num_meshes = (TI)backend_state.meshes.size();
        backend_state.scene.triangle_mesh = backend_state.triangle_mesh.data();
        backend_state.scene.triangle_local = backend_state.triangle_local.data();
        backend_state.scene.num_triangles = (TI)backend_state.triangle_mesh.size();

        const size_t num_triangles = backend_state.triangle_mesh.size();
        std::vector<T> triangle_bounds_min(num_triangles > 0 ? 3 * num_triangles : 1);
        std::vector<T> triangle_bounds_max(num_triangles > 0 ? 3 * num_triangles : 1);
        std::vector<T> centroids(num_triangles > 0 ? 3 * num_triangles : 1);
        for(TI triangle = 0; triangle < (TI)num_triangles; triangle++){
            T bounds_min[3] = {(T)1e30, (T)1e30, (T)1e30};
            T bounds_max[3] = {(T)-1e30, (T)-1e30, (T)-1e30};
            generic::expand_triangle_bounds(backend_state.scene, triangle, bounds_min, bounds_max);
            for(int axis = 0; axis < 3; axis++){
                triangle_bounds_min[3 * triangle + axis] = bounds_min[axis];
                triangle_bounds_max[3 * triangle + axis] = bounds_max[axis];
                centroids[3 * triangle + axis] = (bounds_min[axis] + bounds_max[axis]) * (T)0.5;
            }
        }

        backend_state.objects.clear();
        backend_state.nodes.resize(num_triangles > 0 ? 2 * num_triangles : 1);
        backend_state.primitives.resize(num_triangles > 0 ? num_triangles : 1);
        std::vector<TI> temp_primitives(backend_state.primitives.size());
        for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
            const TI first = object_first_triangle[object_i];
            const TI count = object_triangle_count[object_i];
            generic::ObjectView<T, TI> object_view;
            object_view.nodes = backend_state.nodes.data() + 2 * (size_t)first;
            object_view.primitives = backend_state.primitives.data() + first;
            for(TI i = 0; i < count; i++){
                backend_state.primitives[first + i] = first + i;
            }
            object_view.num_nodes = generic::build_bvh_nodes(backend_state.nodes.data() + 2 * (size_t)first, backend_state.primitives.data() + first, temp_primitives.data(), triangle_bounds_min.data(), triangle_bounds_max.data(), centroids.data(), count);
            backend_state.objects.push_back(object_view);
        }
        backend_state.scene.objects = backend_state.objects.data();
        backend_state.scene.num_objects = (TI)backend_state.objects.size();

        backend_state.instances.clear();
        for(const auto& instance : scene.instances){
            generic::InstanceView<T, TI> instance_view;
            instance_view.object = (TI)instance.object;
            instance_view.identity = instance.identity;
            for(int element = 0; element < 12; element++){
                instance_view.object_to_world[element] = (T)instance.transform[element];
            }
            if(instance.identity){
                for(int element = 0; element < 12; element++){
                    instance_view.world_to_object[element] = instance_view.object_to_world[element];
                }
            }
            else{
                float world_to_object[12];
                rendering::raytracing::detail::invert_transform(instance.transform, world_to_object);
                for(int element = 0; element < 12; element++){
                    instance_view.world_to_object[element] = (T)world_to_object[element];
                }
            }
            backend_state.instances.push_back(instance_view);
        }
        backend_state.num_scene_instances = (TI)backend_state.instances.size();
        if constexpr (SPEC::ENABLE_OVERLAYS){
            generic::InstanceView<T, TI> inactive_slot;
            const T identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
            for(int element = 0; element < 12; element++){
                inactive_slot.object_to_world[element] = identity[element];
                inactive_slot.world_to_object[element] = identity[element];
            }
            inactive_slot.object = 0;
            inactive_slot.identity = true;
            backend_state.instances.resize(backend_state.num_scene_instances + (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES, inactive_slot);
            const size_t bounds_size = 3 * backend_state.instances.size();
            backend_state.overlay_bounds_min.resize(bounds_size);
            backend_state.overlay_bounds_max.resize(bounds_size);
            backend_state.overlay_centroids.resize(bounds_size);
            for(auto& overlay_state : renderer.overlays){
                for(auto& slot : overlay_state.slots){
                    slot.active = false;
                }
                overlay_state.dirty = true;
            }
            renderer.attachments_dirty = true;
        }
        backend_state.scene.instances = backend_state.instances.data();
        backend_state.scene.num_instances = (TI)backend_state.instances.size();

        // the main TLAS spans only the shared-world instances; overlay slots live in their own TLASes
        const size_t num_instances = backend_state.num_scene_instances;
        std::vector<T> instance_bounds_min(num_instances > 0 ? 3 * num_instances : 1);
        std::vector<T> instance_bounds_max(num_instances > 0 ? 3 * num_instances : 1);
        std::vector<T> instance_centroids(num_instances > 0 ? 3 * num_instances : 1);
        for(size_t instance_i = 0; instance_i < num_instances; instance_i++){
            T bounds_min[3], bounds_max[3];
            generic::instance_world_bounds(backend_state, backend_state.instances[instance_i], bounds_min, bounds_max);
            for(int axis = 0; axis < 3; axis++){
                instance_bounds_min[3 * instance_i + axis] = bounds_min[axis];
                instance_bounds_max[3 * instance_i + axis] = bounds_max[axis];
                instance_centroids[3 * instance_i + axis] = (bounds_min[axis] + bounds_max[axis]) * (T)0.5;
            }
        }
        backend_state.tlas_nodes.resize(num_instances > 0 ? 2 * num_instances : 1);
        backend_state.tlas_primitives.resize(num_instances > 0 ? num_instances : 1);
        std::vector<TI> tlas_temp(backend_state.tlas_primitives.size());
        for(size_t instance_i = 0; instance_i < num_instances; instance_i++){
            backend_state.tlas_primitives[instance_i] = (TI)instance_i;
        }
        backend_state.scene.num_tlas_nodes = generic::build_bvh_nodes(backend_state.tlas_nodes.data(), backend_state.tlas_primitives.data(), tlas_temp.data(), instance_bounds_min.data(), instance_bounds_max.data(), instance_centroids.data(), (TI)num_instances);
        backend_state.scene.tlas_nodes = backend_state.tlas_nodes.data();
        backend_state.scene.tlas_primitives = backend_state.tlas_primitives.data();
        renderer.backend.world = backend_state.tlas_nodes.data();

        backend_state.lights = rendering::raytracing::detail::effective_scene_lights<SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING>(scene);
        backend_state.scene.lights = backend_state.lights.data();
        backend_state.scene.num_lights = (TI)backend_state.lights.size();

        backend_state.scene.ambient_color[0] = 0.10f;
        backend_state.scene.ambient_color[1] = 0.10f;
        backend_state.scene.ambient_color[2] = 0.10f;
        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            for(int component = 0; component < 3; component++){
                backend_state.scene.miss_color_0[component] = 0.f;
                backend_state.scene.miss_color_1[component] = 0.f;
            }
        } else {
            backend_state.scene.miss_color_0[0] = .8f; backend_state.scene.miss_color_0[1] = 0.f; backend_state.scene.miss_color_0[2] = 0.f;
            backend_state.scene.miss_color_1[0] = .8f; backend_state.scene.miss_color_1[1] = .8f; backend_state.scene.miss_color_1[2] = .8f;
        }
        backend_state.scene.max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        backend_state.scene.max_dist = renderer.camera_radius * 2.0f;

        if constexpr (SPEC::ENABLE_OVERLAYS){
            update(device, renderer); // publish the (empty) overlays and the attachment table
        }
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const rendering::raytracing::Scene& scene){
        static const rendering::raytracing::AssetPool empty_pool{};
        init(device, renderer, scene, empty_pool);
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        auto& backend_state = generic::state(renderer);

        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_state = renderer.overlays[overlay];
            if(!overlay_state.dirty) continue;
            const TI base = backend_state.num_scene_instances + overlay * SPEC::MAX_OVERLAY_INSTANCES;
            TI* primitives = backend_state.overlay_tlas_primitives.data() + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
            TI num_active = 0;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                const auto& host_slot = overlay_state.slots[slot];
                if(!host_slot.active) continue;
                const TI global = base + slot;
                auto& instance_view = backend_state.instances[global];
                instance_view.object = host_slot.object;
                instance_view.identity = rendering::raytracing::detail::transform_is_identity(host_slot.transform);
                for(int element = 0; element < 12; element++){
                    instance_view.object_to_world[element] = (T)host_slot.transform[element];
                }
                if(instance_view.identity){
                    for(int element = 0; element < 12; element++){
                        instance_view.world_to_object[element] = instance_view.object_to_world[element];
                    }
                }
                else{
                    float world_to_object[12];
                    rendering::raytracing::detail::invert_transform(host_slot.transform, world_to_object);
                    for(int element = 0; element < 12; element++){
                        instance_view.world_to_object[element] = (T)world_to_object[element];
                    }
                }
                T bounds_min[3], bounds_max[3];
                generic::instance_world_bounds(backend_state, instance_view, bounds_min, bounds_max);
                for(int axis = 0; axis < 3; axis++){
                    backend_state.overlay_bounds_min[3 * (size_t)global + axis] = bounds_min[axis];
                    backend_state.overlay_bounds_max[3 * (size_t)global + axis] = bounds_max[axis];
                    backend_state.overlay_centroids[3 * (size_t)global + axis] = (bounds_min[axis] + bounds_max[axis]) * (T)0.5;
                }
                primitives[num_active++] = global;
            }
            backend_state.overlay_views[overlay].num_tlas_nodes = generic::build_bvh_nodes(
                backend_state.overlay_tlas_nodes.data() + (size_t)overlay * 2 * SPEC::MAX_OVERLAY_INSTANCES,
                primitives,
                backend_state.overlay_temp_primitives.data(),
                backend_state.overlay_bounds_min.data(),
                backend_state.overlay_bounds_max.data(),
                backend_state.overlay_centroids.data(),
                num_active);
            overlay_state.dirty = false;
        }
        if(renderer.attachments_dirty){
            std::memcpy(backend_state.overlay_attachments.data(), renderer.attachments, backend_state.overlay_attachments.size() * sizeof(TI));
            renderer.attachments_dirty = false;
        }
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::generate_camera_poses(device, renderer, center, radius, up, fov);
        auto& backend_state = generic::state(renderer);
        std::memcpy(backend_state.cameras.data(), data(renderer.cameras), SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>));
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            std::memcpy(backend_state.cameras_open.data(), data(renderer.cameras), SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>));
        }
        renderer.backend.cameras_buffer = backend_state.cameras.data();
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            renderer.backend.cameras_open_buffer = backend_state.cameras_open.data();
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace generic = rendering::raytracing::backends::generic;
        auto& backend_state = generic::state(renderer);
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        std::memcpy(backend_state.cameras.data(), data(cameras), camera_bytes);
        renderer.backend.cameras_buffer = backend_state.cameras.data();
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            std::memcpy(backend_state.cameras_open.data(), data(cameras), camera_bytes);
            renderer.backend.cameras_open_buffer = backend_state.cameras_open.data();
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras(device, renderer, cameras);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_OPEN_SPEC, typename CAMERAS_CLOSE_SPEC>
    void set_motion_blur_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_OPEN_SPEC>& cameras_open, const Tensor<CAMERAS_CLOSE_SPEC>& cameras_close){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "set_motion_blur_cameras requires a motion-blur renderer specification");
        static_assert(utils::typing::is_same_v<typename CAMERAS_OPEN_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(utils::typing::is_same_v<typename CAMERAS_CLOSE_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_OPEN_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<0>(typename CAMERAS_CLOSE_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace generic = rendering::raytracing::backends::generic;
        auto& backend_state = generic::state(renderer);
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        std::memcpy(backend_state.cameras_open.data(), data(cameras_open), camera_bytes);
        std::memcpy(backend_state.cameras.data(), data(cameras_close), camera_bytes);
        renderer.backend.cameras_buffer = backend_state.cameras.data();
        renderer.backend.cameras_open_buffer = backend_state.cameras_open.data();
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace generic = rendering::raytracing::backends::generic;
        auto& backend_state = generic::state(renderer);
        backend_state.probe_directions = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        backend_state.scene.probe_directions = backend_state.probe_directions.data();
        renderer.backend.probe_dirs_buffer = backend_state.probe_directions.data();
#endif
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        generic::render_frame<DEVICE, SPEC, generic::OutputRGB>(device, generic::state(renderer).scene);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_only_launch(device, renderer);
        render_rgb_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        generic::render_frame<DEVICE, SPEC, generic::OutputDepth>(device, generic::state(renderer).scene);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_depth_only_launch(device, renderer);
        render_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_segmentation_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_SEGMENTATION, "render_segmentation_only requires a segmentation-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        generic::render_segmentation_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
    }

    template <typename DEVICE, typename SPEC>
    void render_segmentation_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_SEGMENTATION, "render_segmentation_only requires a segmentation-capable renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void render_segmentation_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_segmentation_only_launch(device, renderer);
        render_segmentation_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        render_rgb_only_launch(device, renderer);
        render_depth_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_depth_only_launch(device, renderer);
        render_rgb_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        if(renderer.backend.collision_ray_gen != nullptr){
            generic::render_collision<DEVICE, SPEC>(device, generic::state(renderer).scene);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_collision_only_launch(device, renderer);
        render_collision_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        if constexpr (SPEC::HAS_RGB) {
            render_rgb_only_launch(device, renderer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            render_depth_only_launch(device, renderer);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            render_segmentation_only_launch(device, renderer);
        }
        render_collision_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void render_rgb_only_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras_async(device, renderer, cameras);
        render_rgb_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename FB_SPEC>
    void read_frame_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<FB_SPEC>& out_pixels){
        static_assert(SPEC::HAS_RGB, "read_frame_buffer requires an RGB-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename FB_SPEC::T, uint32_t>);
        static_assert(get<0>(typename FB_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace generic = rendering::raytracing::backends::generic;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_pixels), generic::state(renderer).frame_buffer.data(), expected * sizeof(uint32_t));
    }

    template <typename DEVICE, typename SPEC, typename DEPTH_SPEC>
    void read_depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<DEPTH_SPEC>& out_depth){
        static_assert(SPEC::HAS_DEPTH, "read_depth_buffer requires a depth-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename DEPTH_SPEC::T, float>);
        static_assert(get<0>(typename DEPTH_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);
        namespace generic = rendering::raytracing::backends::generic;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_depth), generic::state(renderer).depth_buffer.data(), expected * sizeof(float));
    }

    template <typename DEVICE, typename SPEC, typename SEGMENTATION_SPEC>
    void read_segmentation_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<SEGMENTATION_SPEC>& out_segmentation){
        static_assert(SPEC::HAS_SEGMENTATION, "read_segmentation_buffer requires a segmentation-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename SEGMENTATION_SPEC::T, uint32_t>);
        static_assert(get<0>(typename SEGMENTATION_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace generic = rendering::raytracing::backends::generic;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_segmentation), generic::state(renderer).segmentation_buffer.data(), expected * sizeof(uint32_t));
    }

    template <typename DEVICE, typename SPEC>
    void save_segmentation_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_SEGMENTATION, "save_segmentation_image requires a segmentation-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::write_segmentation_grid_png<SPEC>(generic::state(renderer).segmentation_buffer.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::write_grid_png<SPEC>((const uint32_t*)generic::state(renderer).frame_buffer.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::write_depth_grid_png<SPEC>(generic::state(renderer).depth_buffer.data(), renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::write_depth_bin<SPEC>(generic::state(renderer).depth_buffer.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        namespace generic = rendering::raytracing::backends::generic;
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(generic::state(renderer).collision_results.data(), filename);
#endif
    }

    template <typename DEVICE, typename SPEC, typename COLL_SPEC>
    void read_collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<COLL_SPEC>& out){
        static_assert(utils::typing::is_same_v<typename COLL_SPEC::T, rendering::raytracing::CollisionResult>);
        static_assert(get<0>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_PROBES);
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        namespace generic = rendering::raytracing::backends::generic;
        if(renderer.backend.collision_results_buffer != nullptr){
            std::memcpy(data(out), generic::state(renderer).collision_results.data(),
                        SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult));
        }
#endif
    }

    template <typename DEVICE, typename SPEC>
    const rendering::raytracing::CollisionResult* read_collision_results_raw(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        return nullptr;
#else
        namespace generic = rendering::raytracing::backends::generic;
        if(renderer.backend.collision_results_buffer == nullptr){
            return nullptr;
        }
        return generic::state(renderer).collision_results.data();
#endif
    }

    template <typename DEVICE, typename SPEC>
    uint32_t* get_framebuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "get_framebuffer_device_ptr requires an RGB-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        return (uint32_t*)generic::state(renderer).frame_buffer.data();
    }

    template <typename DEVICE, typename SPEC>
    float* get_depthbuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "get_depthbuffer_device_ptr requires a depth-capable renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        return generic::state(renderer).depth_buffer.data();
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        if(renderer.backend.context != nullptr){
            delete (generic::State<SPEC>*)renderer.backend.context;
            renderer.backend.context = nullptr;
        }
        free(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            free(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            free(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            free(device, renderer.depth_buffer);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            free(device, renderer.segmentation_buffer);
        }
        free(device, renderer.collision_results);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
