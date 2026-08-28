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
            std::vector<unsigned int> object_classes;   // per global object
            std::vector<unsigned int> instance_classes; // per global instance id
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
            SceneView<T, TI> scene;
        };
    }

    namespace rendering::raytracing::backends {
        template <typename SPEC>
        struct RendererState<rendering::raytracing::backends::Generic, SPEC>: generic::State<SPEC> {};

        template <typename SPEC>
        struct LibraryState<rendering::raytracing::backends::Generic, SPEC> {};

        template <typename SPEC>
        struct SceneState<rendering::raytracing::backends::Generic, SPEC> {};
    }

    namespace rendering::raytracing::backends::generic{

        template <typename SPEC>
        State<SPEC>& state(rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
            return *renderer.backend;
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
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The generic raytracing backend requires T = float");

        malloc(device, renderer.cameras);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            malloc(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            malloc(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            malloc(device, renderer.depth_buffer);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            malloc(device, renderer.transforms);
        }
        if constexpr (SPEC::HAS_TRANSFORM_PAIR) {
            malloc(device, renderer.transforms_pair);
            std::memset(data(renderer.transforms_pair), 0, decltype(renderer.transforms_pair)::SPEC::SIZE_BYTES);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            malloc(device, renderer.transforms_motion);
            std::memset(data(renderer.transforms_motion), 0, decltype(renderer.transforms_motion)::SPEC::SIZE_BYTES);
            renderer.transforms_motion_staging.assign((size_t)SPEC::MOTION_BLUR_SAMPLES * SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, 0.0f);
            if constexpr (SPEC::HAS_RGB) {
                malloc(device, renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                malloc(device, renderer.depth_accumulator);
            }
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        malloc(device, renderer.collision_results);
#endif

        auto* backend_state = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Generic, SPEC>{};
        renderer.backend = backend_state;
        // the renderer-owned camera tensors are the render input — no staging copy
        backend_state->scene.cameras_close = data(renderer.cameras);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            backend_state->scene.cameras_open = data(renderer.cameras_open);
        }
        else {
            backend_state->scene.cameras_open = data(renderer.cameras);
        }
        // the renderer-owned output tensors are the render targets — no staging copy
        if constexpr (SPEC::HAS_RGB) {
            backend_state->scene.frame_buffer = data(renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            backend_state->scene.depth_buffer = data(renderer.depth_buffer);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            malloc(device, renderer.segmentation_buffer);
            backend_state->scene.segmentation_buffer = data(renderer.segmentation_buffer);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            malloc(device, renderer.normals_buffer);
            backend_state->scene.normals_buffer = data(renderer.normals_buffer);
        }
        if constexpr (SPEC::HAS_FLOW) {
            malloc(device, renderer.flow_buffer);
            backend_state->scene.flow_buffer = data(renderer.flow_buffer);
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                malloc(device, renderer.flow_deltas);
                backend_state->scene.flow_deltas = data(renderer.flow_deltas);
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            static_assert(utils::typing::is_same_v<typename SPEC::OBSERVATION_T, float>, "The generic raytracing backend requires OBSERVATION_T = float");
            malloc(device, renderer.observation);
            backend_state->scene.observation = data(renderer.observation);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            if constexpr (SPEC::HAS_RGB) {
                backend_state->scene.rgb_accumulation = data(renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                backend_state->scene.depth_accumulation = data(renderer.depth_accumulator);
            }
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
        backend_state->scene.collision_results = data(renderer.collision_results);
#endif
        rendering::raytracing::detail::announce_backend(renderer);
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer);

    template <typename DEVICE, typename SPEC, typename METADATA_T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::SceneMetadata<METADATA_T>& metadata){
        renderer.max_ray_length = (typename SPEC::T)metadata.max_ray_length;
        rendering::raytracing::detail::announce_configuration<SPEC>();
        namespace generic = rendering::raytracing::backends::generic;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        auto& backend_state = generic::state(renderer);

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

        backend_state.object_classes.clear();
        for(const auto* object_pointer : all_objects){
            backend_state.object_classes.push_back(object_pointer->segmentation_class);
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
            // flat array position == global instance id (contract: segmentation_object, operations_cpu_common.h)
            backend_state.instances.resize(backend_state.num_scene_instances + (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES, inactive_slot);
            const size_t bounds_size = 3 * backend_state.instances.size();
            backend_state.overlay_bounds_min.resize(bounds_size);
            backend_state.overlay_bounds_max.resize(bounds_size);
            backend_state.overlay_centroids.resize(bounds_size);
            rendering::raytracing::detail::reset_overlay_state(renderer);
        }
        backend_state.instance_classes.assign(backend_state.instances.size(), 0);
        for(TI instance_i = 0; instance_i < backend_state.num_scene_instances; instance_i++){
            backend_state.instance_classes[instance_i] = backend_state.object_classes[backend_state.instances[instance_i].object];
        }
        backend_state.scene.instance_classes = backend_state.instance_classes.data();
        backend_state.scene.instances = backend_state.instances.data();
        backend_state.scene.num_instances = (TI)backend_state.instances.size();
        backend_state.scene.first_overlay_instance = backend_state.num_scene_instances;

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
        backend_state.scene.max_depth = renderer.max_ray_length > 0 ? renderer.max_ray_length : 1e30f;
        backend_state.scene.max_dist = renderer.max_ray_length;

        if constexpr (SPEC::ENABLE_OVERLAYS){
            update(device, renderer); // publish the (empty) overlays and the attachment table
        }
    }


    namespace rendering::raytracing::backends::generic{
        // instance/BVH rebuild for the overlays from an arbitrary transforms slab — the regular
        // update() path passes the transforms tensor, the dynamic-motion-blur loop passes
        // per-sample slabs of transforms_motion
        template <typename DEVICE, typename SPEC>
        void rebuild_overlay_instances(DEVICE& device, rl_tools::rendering::raytracing::Renderer<SPEC, rl_tools::rendering::raytracing::backends::Generic>& renderer, const float* transforms_base){
            namespace generic = rl_tools::rendering::raytracing::backends::generic;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            auto& backend_state = generic::state(renderer);
            for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                auto& overlay_state = renderer.overlays[overlay];
                const TI base = backend_state.num_scene_instances + overlay * SPEC::MAX_OVERLAY_INSTANCES;
                TI* primitives = backend_state.overlay_tlas_primitives.data() + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
                TI num_active = 0;
                for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                    const auto& host_slot = overlay_state.slots[slot];
                    if(!host_slot.active) continue;
                    const TI global = base + slot;
                    float world[12];
                    rl_tools::rendering::raytracing::detail::compose_overlay_slot_transform(renderer, transforms_base, overlay, slot, world);
                    auto& instance_view = backend_state.instances[global];
                    instance_view.object = host_slot.object;
                    instance_view.identity = rl_tools::rendering::raytracing::detail::transform_is_identity(world);
                    for(int element = 0; element < 12; element++){
                        instance_view.object_to_world[element] = (T)world[element];
                    }
                    if(instance_view.identity){
                        for(int element = 0; element < 12; element++){
                            instance_view.world_to_object[element] = instance_view.object_to_world[element];
                        }
                    }
                    else{
                        float world_to_object[12];
                        rl_tools::rendering::raytracing::detail::invert_transform(world, world_to_object);
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
                    backend_state.instance_classes[global] = backend_state.object_classes[host_slot.object];
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
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace generic = rendering::raytracing::backends::generic;
        using TI = typename SPEC::TI;
        auto& backend_state = generic::state(renderer);
        if constexpr (SPEC::HAS_FLOW){
            rendering::raytracing::detail::compose_flow_deltas(renderer, data(renderer.flow_deltas));
        }
        rendering::raytracing::detail::flush_overlay_transforms(renderer);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            rendering::raytracing::detail::flush_overlay_motion_transforms(renderer);
        }

        // rebuilt unconditionally: producers may write the transforms tensor directly, which
        // leaves no host-observable dirty flag
        generic::rebuild_overlay_instances(device, renderer, data(renderer.transforms));
        if(renderer.attachments_dirty){
            std::memcpy(backend_state.overlay_attachments.data(), renderer.attachments, backend_state.overlay_attachments.size() * sizeof(TI));
            renderer.attachments_dirty = false;
        }
    }

    template <typename DEVICE, typename SPEC>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        update(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
        rendering::raytracing::detail::expand_motion_transforms_host(renderer);
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        expand_motion_transforms_launch(device, renderer);
        expand_motion_transforms_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
    }


    template <typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(rendering::raytracing::backends::Device<rendering::raytracing::backends::Generic>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        copy(to_device, to_device, from, to);
    }
    template <typename FROM_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, rendering::raytracing::backends::Device<rendering::raytracing::backends::Generic>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        copy(from_device, from_device, from, to);
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace generic = rendering::raytracing::backends::generic;
        auto& backend_state = generic::state(renderer);
        backend_state.probe_directions = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        backend_state.scene.probe_directions = backend_state.probe_directions.data();
#endif
    }

    // render produces the image outputs the spec declares; the collision-probe pass is the
    // separate probe verb so it can be scheduled independently (e.g. alongside update).
    // Under dynamic motion blur it runs MOTION_BLUR_SAMPLES sequential passes, rebuilding the
    // overlay BVHs from the per-sample transforms_motion slab before each, accumulating linear
    // radiance, then restores the shutter-close state (segmentation, probes, steady state)
    // before resolving.
    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            if constexpr (SPEC::HAS_RGB) {
                std::memset(data(renderer.rgb_accumulator), 0, decltype(renderer.rgb_accumulator)::SPEC::SIZE_BYTES);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                std::memset(data(renderer.depth_accumulator), 0, decltype(renderer.depth_accumulator)::SPEC::SIZE_BYTES);
            }
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                generic::rebuild_overlay_instances(device, renderer, data(renderer.transforms_motion) + sample * SLAB);
                const T shutter_t = ((T)sample + (T)0.5) / (T)SPEC::MOTION_BLUR_SAMPLES;
                if constexpr (SPEC::HAS_RGB) {
                    generic::render_frame_accumulate<DEVICE, SPEC, generic::OutputRGB>(device, generic::state(renderer).scene, shutter_t);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    generic::render_frame_accumulate<DEVICE, SPEC, generic::OutputDepth>(device, generic::state(renderer).scene, shutter_t);
                }
            }
            generic::rebuild_overlay_instances(device, renderer, data(renderer.transforms));
            if constexpr (SPEC::HAS_RGB) {
                generic::resolve_frame<DEVICE, SPEC, generic::OutputRGB>(device, generic::state(renderer).scene);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                generic::resolve_frame<DEVICE, SPEC, generic::OutputDepth>(device, generic::state(renderer).scene);
            }
            if constexpr (SPEC::HAS_SEGMENTATION) {
                generic::render_segmentation_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
            }
            if constexpr (SPEC::HAS_NORMALS) {
                generic::render_normals_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
            }
            if constexpr (SPEC::HAS_FLOW) {
                generic::render_flow_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
            }
            return;
        }
        if constexpr (SPEC::HAS_RGB) {
            generic::render_frame<DEVICE, SPEC, generic::OutputRGB>(device, generic::state(renderer).scene);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            generic::render_frame<DEVICE, SPEC, generic::OutputDepth>(device, generic::state(renderer).scene);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            generic::render_segmentation_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            generic::render_normals_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
        }
        if constexpr (SPEC::HAS_FLOW) {
            generic::render_flow_frame<DEVICE, SPEC>(device, generic::state(renderer).scene);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        generic::render_collision<DEVICE, SPEC>(device, generic::state(renderer).scene);
#endif
    }

    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void probe(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        probe_launch(device, renderer);
        probe_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer){
        namespace generic = rendering::raytracing::backends::generic;
        if(renderer.backend != nullptr){
            delete renderer.backend;
            renderer.backend = nullptr;
        }
        free(device, renderer.cameras);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
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
        if constexpr (SPEC::HAS_NORMALS) {
            free(device, renderer.normals_buffer);
        }
        if constexpr (SPEC::HAS_FLOW) {
            free(device, renderer.flow_buffer);
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                free(device, renderer.flow_deltas);
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            free(device, renderer.observation);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            free(device, renderer.transforms);
        }
        if constexpr (SPEC::HAS_TRANSFORM_PAIR) {
            free(device, renderer.transforms_pair);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            free(device, renderer.transforms_motion);
            if constexpr (SPEC::HAS_RGB) {
                free(device, renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                free(device, renderer.depth_accumulator);
            }
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        free(device, renderer.collision_results);
#endif
    }

    // shared-asset-library fallbacks: this backend has no cross-renderer sharing, so the
    // library is empty and every renderer builds its own copy — the API stays uniform
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Generic>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<rendering::raytracing::backends::Generic, SPEC>{};
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Generic>& library){
        for(auto* assets : library.assets){
            delete assets;
        }
        library.assets.clear();
        library.scenes.clear();
        library.metadata.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Generic>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Generic>& library){
        malloc(device, renderer);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#include "../../operations_cpu_post.h"

#endif
