#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H

#include "renderer.h"
#include "../segmentation_cpu.h"
#include "transforms_generic.h"

#include <vector>
#include <limits>
#include <algorithm>
#include <functional>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>

#ifndef RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        inline constexpr char LOG_PREFIX[] = "#rl_tools::rendering::raytracing: ";
    }

    template <typename SPEC, typename DEVICE>
    void validate_semantic_segmentation(DEVICE& device, const rendering::Scene& scene, const rendering::AssetPool& pool){
        if constexpr(SPEC::SEMANTIC_SEGMENTATION){
            if constexpr(SPEC::ENABLE_OVERLAYS){
                validate_semantic_segmentation(device, scene, pool);
            }
            else{
                validate_semantic_segmentation(device, segmentation_objects(device, scene));
            }
        }
    }

    // backend-generic launch+sync composite, defined in operations_cpu_post.h; declared here so
    // backend-internal callers (init) resolve it before the definition is included
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer);

    namespace rendering::raytracing::detail{
        template <typename DEVICE, typename SPEC, typename BACKEND>
        void announce_backend(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>&){
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "backend=", rendering::raytracing::backends::name<BACKEND>());
        }
    }

    // =========================================================================
    // Default cube geometry
    // =========================================================================
    namespace rendering::raytracing::constants{
        const int NUM_VERTICES = 8;
        const float default_vertices[8][3] = {
            { -1.f,-1.f,-1.f },
            { +1.f,-1.f,-1.f },
            { -1.f,+1.f,-1.f },
            { +1.f,+1.f,-1.f },
            { -1.f,-1.f,+1.f },
            { +1.f,-1.f,+1.f },
            { -1.f,+1.f,+1.f },
            { +1.f,+1.f,+1.f }
        };
        const int NUM_INDICES = 12;
        const int default_indices[12][3] = {
            { 0,1,3 }, { 2,3,0 },
            { 5,7,6 }, { 5,6,4 },
            { 0,4,5 }, { 0,5,1 },
            { 2,3,7 }, { 2,7,6 },
            { 1,5,7 }, { 1,7,3 },
            { 4,0,2 }, { 4,2,6 }
        };
    }

    namespace rendering::raytracing::detail{

    inline constexpr float IDENTITY_TRANSFORM[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    // Host staging of the overlay transform inputs. The verbs write the staging mirrors and mark
    // rows pending; update() publishes exactly the pending rows into the resident tensors in the
    // backend's memory domain and clears the marks. Producers write the resident tensors
    // directly and are never touched by the publish — a row belongs to whoever wrote it last.
    template <typename SPEC, typename BACKEND>
    void malloc_overlay_staging(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        renderer.transforms_staging.assign(SLOTS * 12, 0.0f);
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            renderer.transforms_previous_staging.assign(SLOTS * 12, 0.0f);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            renderer.transforms_motion_staging.assign((size_t)SPEC::MOTION_BLUR_SAMPLES * SLOTS * 12, 0.0f);
        }
    }

    // one slot's current pose; under dynamic blur the same pose in every motion sample (sharp)
    template <typename SPEC, typename BACKEND>
    void stage_transform(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, size_t slot, const float pose[12]){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        std::memcpy(renderer.transforms_staging.data() + slot * 12, pose, 12 * sizeof(float));
        renderer.transforms_pending[slot] = true;
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            for(typename SPEC::TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                std::memcpy(renderer.transforms_motion_staging.data() + (sample * SLOTS + slot) * 12, pose, 12 * sizeof(float));
            }
            renderer.transforms_motion_pending[slot] = true;
        }
    }

    // one slot's motion: MOTION_BLUR_SAMPLES poses sample_stride floats apart, and the
    // shutter-close pose as the steady state
    template <typename SPEC, typename BACKEND>
    void stage_motion(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, size_t slot, const float* samples, size_t sample_stride, const float close[12]){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        std::memcpy(renderer.transforms_staging.data() + slot * 12, close, 12 * sizeof(float));
        renderer.transforms_pending[slot] = true;
        for(typename SPEC::TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
            std::memcpy(renderer.transforms_motion_staging.data() + (sample * SLOTS + slot) * 12, samples + sample * sample_stride, 12 * sizeof(float));
        }
        renderer.transforms_motion_pending[slot] = true;
    }

    template <typename SPEC, typename BACKEND>
    void stage_previous(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, size_t slot, const float pose[12]){
        std::memcpy(renderer.transforms_previous_staging.data() + slot * 12, pose, 12 * sizeof(float));
        renderer.transforms_previous_pending[slot] = true;
    }

    // publish for backends whose resident tensors are host-addressable (after their in-flight
    // wait); OptiX scatters the same pending rows on its stream (optix::publish_rows)
    template <typename SPEC, typename BACKEND>
    void publish_overlay_transforms(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        for(size_t slot = 0; slot < SLOTS; slot++){
            if(renderer.transforms_pending[slot]){
                std::memcpy(data(renderer.transforms) + slot * 12, renderer.transforms_staging.data() + slot * 12, 12 * sizeof(float));
                renderer.transforms_pending[slot] = false;
            }
        }
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            for(size_t slot = 0; slot < SLOTS; slot++){
                if(renderer.transforms_previous_pending[slot]){
                    std::memcpy(data(renderer.transforms_previous) + slot * 12, renderer.transforms_previous_staging.data() + slot * 12, 12 * sizeof(float));
                    renderer.transforms_previous_pending[slot] = false;
                }
            }
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            for(size_t slot = 0; slot < SLOTS; slot++){
                if(renderer.transforms_motion_pending[slot]){
                    for(typename SPEC::TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                        const size_t offset = (sample * SLOTS + slot) * 12;
                        std::memcpy(data(renderer.transforms_motion) + offset, renderer.transforms_motion_staging.data() + offset, 12 * sizeof(float));
                    }
                    renderer.transforms_motion_pending[slot] = false;
                }
            }
        }
    }

    // host half of advance_flow: a row the host wrote but has not yet published becomes the
    // previous instant through its staging mirror; the backend copies the resident rows
    template <typename SPEC, typename BACKEND>
    void advance_flow_staging(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        for(size_t slot = 0; slot < SLOTS; slot++){
            if(renderer.transforms_pending[slot]){
                stage_previous(renderer, slot, renderer.transforms_staging.data() + slot * 12);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void assert_placement(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement){
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "overlay verb: overlay index out of range");
        utils::assert_exit(device, placement.first_slot + placement.num_parts <= SPEC::MAX_OVERLAY_INSTANCES, "overlay verb: placement out of range");
        for(size_t part = 0; part < placement.num_parts; part++){
            utils::assert_exit(device, renderer.overlays[overlay.index].slots[placement.first_slot + part].active, "overlay verb: placement is not spawned");
        }
    }

    // per-slot flow delta table [2][SLOTS][12] from two transforms-layout slabs: slab 0 carries a
    // current-instant point to the previous instant (world_previous ∘ world_current⁻¹, the
    // backward pass), slab 1 the inverse (forward pass); identity for inactive slots. Backends
    // with host-resident tensors call it in update() after the flush; OptiX runs the same
    // composition on-device (overlay_accel_compose_flow_deltas).
    template <typename SPEC, typename BACKEND>
    void compose_flow_deltas(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const float* current_slab, const float* previous_slab, float* deltas){
        using TI = typename SPEC::TI;
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                const size_t index = (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot;
                float* backward = deltas + index * 12;
                float* forward = deltas + (SLOTS + index) * 12;
                if(!renderer.overlays[overlay].slots[slot].active){
                    std::memcpy(backward, IDENTITY_TRANSFORM, 12 * sizeof(float));
                    std::memcpy(forward, IDENTITY_TRANSFORM, 12 * sizeof(float));
                    continue;
                }
                const float* world_current = current_slab + index * 12;
                const float* world_previous = previous_slab + index * 12;
                float world_current_inverse[12], world_previous_inverse[12];
                invert_transform(world_current, world_current_inverse);
                invert_transform(world_previous, world_previous_inverse);
                compose_transforms(world_previous, world_current_inverse, backward);
                compose_transforms(world_current, world_previous_inverse, forward);
            }
        }
    }

    // flattens the asset pool for overlay spawns: appends pool objects to the combined object
    // list and records per-part global object indices + local transforms on the renderer
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void register_pool_assets(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::raytracing::AssetPool& pool, std::vector<const rendering::raytracing::Object*>& all_objects){
        using TI = typename SPEC::TI;
        renderer.assets.clear();
        renderer.asset_part_objects.clear();
        renderer.asset_part_transforms.clear();
        for(const auto& assembly : pool.assemblies){
            typename rendering::raytracing::Renderer<SPEC, BACKEND>::AssetRecord record;
            record.first_part = (TI)renderer.asset_part_objects.size();
            record.num_parts = (TI)assembly.parts.size();
            utils::assert_exit(device, record.num_parts <= SPEC::MAX_OVERLAY_INSTANCES, "asset has more parts than the overlay capacity");
            const TI object_base = (TI)all_objects.size();
            for(const auto& object : assembly.objects){
                all_objects.push_back(&object);
            }
            for(const auto& part : assembly.parts){
                renderer.asset_part_objects.push_back(object_base + (TI)part.object);
                renderer.asset_part_transforms.insert(renderer.asset_part_transforms.end(), part.transform, part.transform + 12);
            }
            renderer.assets.push_back(record);
        }
    }

    // Deterministic first-fit is a contract, not an implementation detail: the chosen slot defines
    // the global instance id (segmentation output), which must be reproducible across runs and
    // identical across backends. Do not replace with a free-list or best-fit strategy.
    template <typename SPEC, typename BACKEND>
    typename SPEC::TI first_fit_slot(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, size_t overlay, typename SPEC::TI num_parts){
        using TI = typename SPEC::TI;
        const auto& state = renderer.overlays[overlay];
        TI run = 0;
        for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
            run = state.slots[slot].active ? 0 : run + 1;
            if(run == num_parts){
                return slot + 1 - num_parts;
            }
        }
        return SPEC::MAX_OVERLAY_INSTANCES;
    }

    template <typename SPEC, typename BACKEND>
    void reset_overlay_state(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        for(auto& overlay_state : renderer.overlays){
            for(auto& slot : overlay_state.slots){
                slot.active = false;
            }
            overlay_state.dirty = true;
        }
        for(size_t slot = 0; slot < SLOTS; slot++){
            renderer.transforms_pending[slot] = false;
            if constexpr (SPEC::HAS_FLOW_OVERLAYS){
                renderer.transforms_previous_pending[slot] = false;
            }
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                renderer.transforms_motion_pending[slot] = false;
            }
        }
        for(TI attachment_i = 0; attachment_i < SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA; attachment_i++){
            renderer.attachments[attachment_i] = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
        }
        renderer.attachments_dirty = true;
    }

    // resolved-configuration echo at debug verbosity: makes a silently-defaulted (e.g.
    // misspelled) fringe config member visible
    template <typename SPEC, typename DEVICE>
    void announce_configuration(DEVICE& device){
        log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "config: ", SPEC::NUM_CAMERAS, " camera(s) ", SPEC::CAM_WIDTH, "x", SPEC::CAM_HEIGHT,
            " outputs[rgb=", SPEC::HAS_RGB, " depth=", SPEC::HAS_DEPTH, " segmentation=", SPEC::HAS_SEGMENTATION, (SPEC::SEMANTIC_SEGMENTATION ? " (semantic)" : ""), " normals=", SPEC::HAS_NORMALS, " flow=", SPEC::HAS_FLOW, " flow_forward=", SPEC::HAS_FLOW_FORWARD, "]",
            " probes=", (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS ? 0 : SPEC::NUM_PROBES),
            " motion_blur_samples=", (SPEC::ENABLE_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 0),
            " anti_aliasing_grid=", (SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 0),
            " overlays=", SPEC::NUM_OVERLAYS, "x", SPEC::MAX_OVERLAY_INSTANCES, " overlays_per_camera=", SPEC::MAX_OVERLAYS_PER_CAMERA);
    }

    // resolves a Scene::Environment into the concrete ambient + miss color pair the shaders
    // consume. DEFAULT reproduces the historical constants exactly (golden-pinned); GRADIENT and
    // EQUIRECT fall back to a solid horizon background until the shaders consume them.
    template <typename SPEC>
    void resolve_environment(const rendering::Scene::Environment& environment, typename SPEC::T ambient_color[3], typename SPEC::T miss_color_0[3], typename SPEC::T miss_color_1[3]){
        using T = typename SPEC::T;
        using Environment = rendering::Scene::Environment;
        if(environment.mode == Environment::Mode::DEFAULT){
            ambient_color[0] = 0.10f; ambient_color[1] = 0.10f; ambient_color[2] = 0.10f;
            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING){
                for(int component = 0; component < 3; component++){
                    miss_color_0[component] = 0.f;
                    miss_color_1[component] = 0.f;
                }
            }
            else{
                miss_color_0[0] = .8f; miss_color_0[1] = 0.f; miss_color_0[2] = 0.f;
                miss_color_1[0] = .8f; miss_color_1[1] = .8f; miss_color_1[2] = .8f;
            }
        }
        else{
            for(int component = 0; component < 3; component++){
                ambient_color[component] = (T)environment.ambient[component];
                miss_color_0[component] = (T)environment.horizon[component];
                miss_color_1[component] = (T)environment.horizon[component];
            }
        }
    }

    // Lights as uploaded to the device: only the PBR tiers consume punctual lights. Object lights
    // are authored in the object's local frame and follow the instance placing it.
    template <bool APPLY, typename DEVICE>
    std::vector<rendering::raytracing::SceneLight> effective_scene_lights(DEVICE& device, const rendering::raytracing::Scene& scene){
        std::vector<rendering::raytracing::SceneLight> lights;
        if constexpr (APPLY) {
            lights = scene.lights;
            for(const auto& instance : scene.instances){
                for(const auto& object_light : scene.objects[instance.object].lights){
                    rendering::raytracing::SceneLight light = object_light;
                    if(!instance.identity){
                        transform_point(instance.transform, object_light.position, light.position);
                        transform_vector(instance.transform, object_light.direction, light.direction);
                        const float length = sqrtf(light.direction[0]*light.direction[0] + light.direction[1]*light.direction[1] + light.direction[2]*light.direction[2]);
                        if(length > 0){
                            light.direction[0] /= length;
                            light.direction[1] /= length;
                            light.direction[2] /= length;
                        }
                    }
                    lights.push_back(light);
                }
            }
            log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "Scene lights: ", lights.size());
            for (size_t li = 0; li < lights.size(); li++) {
                auto& sl = lights[li];
                log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "  light ", li, ": pos=(", sl.position[0], ",", sl.position[1], ",", sl.position[2],
                    ") dir=(", sl.direction[0], ",", sl.direction[1], ",", sl.direction[2],
                    ") color=(", sl.color[0], ",", sl.color[1], ",", sl.color[2], ")");
            }
        }
        return lights;
    }
    } // namespace rendering::raytracing::detail

    namespace rendering::raytracing::detail{
        // content-hash dedup insert: on a miss the library takes ownership of the bundle's scene
        // (the bundle's metadata stays with the caller for producer-side use)
        template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
        typename SPEC::TI library_insert(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library, rendering::Bundle<T>& bundle, bool& is_new){
            validate_semantic_segmentation<SPEC>(device, bundle.scene, library.pool);
            using TI = typename SPEC::TI;
            utils::assert_exit(device, !bundle.metadata.content_hash.empty(), "library: bundle has no content hash — load it through a dataset loader");
            for(TI scene_i = 0; scene_i < (TI)library.metadata.size(); scene_i++){
                if(library.metadata[scene_i].content_hash == bundle.metadata.content_hash){
                    is_new = false;
                    log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "library: scene ", bundle.metadata.content_hash, " shares build ", scene_i, " (content hash match)");
                    return scene_i;
                }
            }
            is_new = true;
            utils::assert_exit(device, !bundle.scene.objects.empty(), "library: bundle scene is empty (already inserted elsewhere?)");
            library.scenes.push_back(std::move(bundle.scene));
            bundle.scene = {};
            library.metadata.push_back({});
            auto& stored = library.metadata.back();
            for(int d = 0; d < 3; d++){
                stored.center[d] = (typename SPEC::T)bundle.metadata.center[d];
                stored.half_extent[d] = (typename SPEC::T)bundle.metadata.half_extent[d];
            }
            stored.max_ray_length = (typename SPEC::T)bundle.metadata.max_ray_length;
            stored.content_hash = bundle.metadata.content_hash;
            library.assets.push_back(nullptr);
            return (TI)(library.scenes.size() - 1);
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
    typename SPEC::TI insert(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library, rendering::Bundle<T>& bundle){
        bool is_new = false;
        return rendering::raytracing::detail::library_insert(device, library, bundle, is_new);
    }

    // appends a root part (geometry in the part frame, placed by transform within the assembly)
    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::ObjectAssembly& assembly, const rendering::raytracing::Object& object, const float transform[12]){
        rendering::raytracing::ObjectAssembly::Part part;
        part.object = assembly.objects.size();
        std::memcpy(part.transform, transform, sizeof(part.transform));
        assembly.objects.push_back(object);
        assembly.parts.push_back(part);
        return assembly.parts.size() - 1;
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Object& object, const float transform[12]){
        scene.objects.push_back(object);
        scene.instances.push_back({scene.objects.size() - 1, {}, rendering::raytracing::detail::transform_is_identity(transform)});
        std::memcpy(scene.instances.back().transform, transform, 12 * sizeof(float));
        return scene.instances.size() - 1;
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Object& object){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, object, identity);
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Mesh& mesh){
        rendering::raytracing::Object object;
        object.meshes.push_back(mesh);
        return add(device, scene, object);
    }


    template <typename DEVICE>
    rendering::raytracing::Placement add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::ObjectAssembly& assembly, const float transform[12]){
        rendering::raytracing::Placement placement{scene.instances.size(), assembly.parts.size()};
        const size_t object_base = scene.objects.size();
        scene.objects.insert(scene.objects.end(), assembly.objects.begin(), assembly.objects.end());
        for(const auto& part : assembly.parts){
            float composed[12];
            rendering::raytracing::detail::compose_transforms(transform, part.transform, composed);
            scene.instances.push_back({object_base + part.object, {}, rendering::raytracing::detail::transform_is_identity(composed)});
            std::memcpy(scene.instances.back().transform, composed, sizeof(composed));
        }
        return placement;
    }

    template <typename DEVICE>
    rendering::raytracing::Placement add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::ObjectAssembly& assembly){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, assembly, identity);
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::ObjectAssembly& assembly){
        pool.assemblies.push_back(assembly);
        return {pool.assemblies.size() - 1};
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::Object& object){
        rendering::raytracing::ObjectAssembly assembly;
        assembly.objects.push_back(object);
        assembly.parts.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}});
        return add(device, pool, assembly);
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::Mesh& mesh){
        rendering::raytracing::Object object;
        object.meshes.push_back(mesh);
        return add(device, pool, object);
    }

    // Validation predicates for boundaries (language bindings, C interface) that must not trip
    // the fail-fast asserts inside the verbs: check first, then call.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_attach(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        if(camera >= SPEC::NUM_CAMERAS || overlay.index >= SPEC::NUM_OVERLAYS){
            return false;
        }
        const TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index || row[slot] == rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY){
                return true;
            }
        }
        return false;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_spawn(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, rendering::raytracing::AssetHandle asset){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_spawn requires an overlay-enabled renderer specification");
        if(overlay.index >= SPEC::NUM_OVERLAYS || asset.index >= renderer.assets.size()){
            return false;
        }
        return rendering::raytracing::detail::first_fit_slot<SPEC>(renderer, overlay.index, renderer.assets[asset.index].num_parts) < SPEC::MAX_OVERLAY_INSTANCES;
    }

    // Overlay verbs mutate host-side truth on the renderer and mark it dirty; update(device,
    // renderer) is the single point where the backend consumes it. All bookkeeping is
    // deterministic: slot allocation is a first-fit scan, so identical call sequences yield
    // identical slots (and therefore identical global instance ids).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void attach(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "attach: overlay index out of range");
        constexpr TI INVALID = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
        TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        TI free_slot = SPEC::MAX_OVERLAYS_PER_CAMERA;
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index){
                return; // a camera's attachments form a set: attach is idempotent
            }
            if(row[slot] == INVALID && free_slot == SPEC::MAX_OVERLAYS_PER_CAMERA){
                free_slot = slot;
            }
        }
        utils::assert_exit(device, free_slot < SPEC::MAX_OVERLAYS_PER_CAMERA, "attach: camera already holds MAX_OVERLAYS_PER_CAMERA overlays");
        row[free_slot] = (TI)overlay.index;
        renderer.attachments_dirty = true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void detach(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "detach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index){
                row[slot] = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
                renderer.attachments_dirty = true;
                return;
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    rendering::raytracing::OverlayPlacement spawn(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, rendering::raytracing::AssetHandle asset, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "spawn requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        if(overlay.index >= SPEC::NUM_OVERLAYS){
            utils::assert_exit(device, false, "spawn: overlay index out of range");
            return {0, 0, 0};
        }
        if(asset.index >= renderer.assets.size()){
            utils::assert_exit(device, false, "spawn: asset handle out of range");
            return {0, 0, 0};
        }
        auto& state = renderer.overlays[overlay.index];
        const auto& record = renderer.assets[asset.index];

        const TI first_slot = rendering::raytracing::detail::first_fit_slot<SPEC>(renderer, overlay.index, record.num_parts);
        if(first_slot >= SPEC::MAX_OVERLAY_INSTANCES){
            utils::assert_exit(device, false, "spawn: overlay capacity exceeded");
            return {0, 0, 0};
        }

        for(TI part = 0; part < record.num_parts; part++){
            auto& slot = state.slots[first_slot + part];
            slot.object = renderer.asset_part_objects[record.first_part + part];
            std::memcpy(slot.rest, &renderer.asset_part_transforms[(record.first_part + part) * 12], sizeof(slot.rest));
            slot.active = true;
            const size_t row = (size_t)overlay.index * SPEC::MAX_OVERLAY_INSTANCES + first_slot + part;
            float world[12];
            rendering::compose_transforms(transform, slot.rest, world);
            rendering::raytracing::detail::stage_transform(renderer, row, world);
            if constexpr (SPEC::HAS_FLOW_OVERLAYS){
                rendering::raytracing::detail::stage_previous(renderer, row, world); // a fresh object has no motion of its own: pure camera flow
            }
        }
        state.dirty = true;
        return {(size_t)first_slot, (size_t)record.num_parts, (size_t)record.first_part};
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void despawn(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement){
        static_assert(SPEC::ENABLE_OVERLAYS, "despawn requires an overlay-enabled renderer specification");
        auto& state = renderer.overlays[overlay.index];
        for(size_t part = 0; part < placement.num_parts; part++){
            state.slots[placement.first_slot + part].active = false;
        }
        state.dirty = true;
    }

    // poses: placement.num_parts geometry-to-world transforms (12 floats each) in asset part
    // order — the output of rig::evaluate for the placement's assembly. Only the current instant
    // changes: the flow pair advances through advance_flow (or spawn)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float* poses){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transforms requires an overlay-enabled renderer specification");
        rendering::raytracing::detail::assert_placement(device, renderer, overlay, placement);
        for(size_t part = 0; part < placement.num_parts; part++){
            rendering::raytracing::detail::stage_transform(renderer, overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part, poses + part * 12);
        }
    }

    // samples: [MOTION_BLUR_SAMPLES][placement.num_parts][12] poses at the shutter times
    // (sample + 0.5) / MOTION_BLUR_SAMPLES the camera lerp uses; close: [placement.num_parts][12]
    // shutter-close poses, the steady state segmentation, probes and the next frame see — the
    // output of rig::sample for the placement's assembly
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float* samples, const float* close){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_motion_transforms requires a dynamic-motion-blur renderer specification");
        rendering::raytracing::detail::assert_placement(device, renderer, overlay, placement);
        for(size_t part = 0; part < placement.num_parts; part++){
            rendering::raytracing::detail::stage_motion(renderer, overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part, samples + part * 12, placement.num_parts * 12, close + part * 12);
        }
    }

    // rigid move of the whole assembly: every part at pose ∘ rest
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float pose[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transform requires an overlay-enabled renderer specification");
        rendering::raytracing::detail::assert_placement(device, renderer, overlay, placement);
        const auto& state = renderer.overlays[overlay.index];
        for(size_t part = 0; part < placement.num_parts; part++){
            float world[12];
            rendering::compose_transforms(pose, state.slots[placement.first_slot + part].rest, world);
            rendering::raytracing::detail::stage_transform(renderer, overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part, world);
        }
    }

    // rigid move with shutter-open/close poses: constant-twist interpolation of the assembly pose
    // into every motion sample, the close pose as the steady state
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float open[12], const float close[12]){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_transform_pair requires a dynamic-motion-blur renderer specification");
        rendering::raytracing::detail::assert_placement(device, renderer, overlay, placement);
        float poses[SPEC::MOTION_BLUR_SAMPLES][12];
        for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
            rendering::screw_interpolate(open, close, ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES, poses[sample]);
        }
        const auto& state = renderer.overlays[overlay.index];
        for(size_t part = 0; part < placement.num_parts; part++){
            const float* rest = state.slots[placement.first_slot + part].rest;
            float samples[SPEC::MOTION_BLUR_SAMPLES][12], world_close[12];
            for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                rendering::compose_transforms(poses[sample], rest, samples[sample]);
            }
            rendering::compose_transforms(close, rest, world_close);
            rendering::raytracing::detail::stage_motion(renderer, overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part, &samples[0][0], 12, world_close);
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "transforms requires an overlay-enabled renderer specification");
        return renderer.transforms;
    }

    // the previous instant's overlay poses (flow): producer input under the transforms()
    // ownership contract, snapshotted from transforms by advance_flow
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_previous(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW_OVERLAYS, "transforms_previous requires a flow renderer specification with overlays");
        return renderer.transforms_previous;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_motion(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "transforms_motion requires a dynamic-motion-blur renderer specification");
        return renderer.transforms_motion;
    }

    // camera input tensors, backend-native residency like transforms: device memory on OptiX,
    // host on generic, shared/mapped on Metal/Vulkan. Producers write them via rlt::copy or
    // kernels; the launch verbs consume them directly. Under motion blur the pair is
    // cameras_open (shutter open) and cameras_close (shutter close, aliasing cameras) — both
    // must be written each step (identical values for a blur-free frame). Under flow,
    // cameras_previous holds the previous instant (snapshotted from cameras by advance_flow).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_open(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_open requires a motion-blur renderer specification");
        return renderer.cameras_open;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_close(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_close requires a motion-blur renderer specification");
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_previous(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW, "cameras_previous requires a flow-capable renderer specification");
        return renderer.cameras_previous;
    }

    // output tensors, same backend-native residency as the inputs: consumers on the device read
    // them in place (zero-copy); host readers stage through a memory-domain copy at readback
    // boundaries — copy(renderer.device, device, ...) — which orders itself after the
    // renderer's in-flight work
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& frame_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_RGB, "frame_buffer requires an RGB-capable renderer specification");
        return renderer.frame_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_DEPTH, "depth_buffer requires a depth-capable renderer specification");
        return renderer.depth_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& segmentation_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_SEGMENTATION, "segmentation_buffer requires a segmentation-capable renderer specification");
        return renderer.segmentation_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& normals_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_NORMALS, "normals_buffer requires a normals-capable renderer specification");
        return renderer.normals_buffer;
    }

    // flow on the current grid: p_previous - p_current in pixels, with a FlowStatus per pixel
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW, "flow_buffer requires a flow-capable renderer specification");
        return renderer.flow_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_status_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW, "flow_status_buffer requires a flow-capable renderer specification");
        return renderer.flow_status_buffer;
    }

    // forward flow on the previous grid: q_current - q_previous in pixels, with a FlowStatus per pixel
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_forward_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW_FORWARD, "flow_forward_buffer requires OUTPUT_FLOW_FORWARD in the renderer specification");
        return renderer.flow_forward_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_forward_status_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW_FORWARD, "flow_forward_status_buffer requires OUTPUT_FLOW_FORWARD in the renderer specification");
        return renderer.flow_forward_status_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        return renderer.collision_results;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& observation(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_OBSERVATION, "observation requires OUTPUT_OBSERVATION in the renderer specification");
        return renderer.observation;
    }

    // decodes a rendered segmentation id back to the object it references. Single owner of the
    // global id layout: scene instances occupy [0, S); overlay o's slot s sits at S + o*CAP + s;
    // object indices count scene objects first, then each pool assembly's objects in
    // registration order. Returns nullptr for the miss sentinel and out-of-range ids.
    // This layout is cross-backend API surface: the generic flat instance array, Metal's user-ID
    // descriptors, and OptiX's user instance ids all realize it identically, and segmentation
    // consumers depend on that equivalence — treat any change to it as breaking.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    const rendering::raytracing::Object* segmentation_object(DEVICE& device, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, uint32_t id){
        static_assert(!SPEC::SEMANTIC_SEGMENTATION, "segmentation_object expects instance IDs, not semantic class IDs");
        if(id == 0xFFFFFFFFu){
            return nullptr;
        }
        if(id < scene.instances.size()){
            return &scene.objects[scene.instances[id].object];
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            const size_t relative = id - scene.instances.size();
            const size_t overlay = relative / SPEC::MAX_OVERLAY_INSTANCES;
            const size_t slot = relative % SPEC::MAX_OVERLAY_INSTANCES;
            if(overlay >= SPEC::NUM_OVERLAYS){
                return nullptr;
            }
            size_t object = renderer.overlays[overlay].slots[slot].object;
            if(object < scene.objects.size()){
                return &scene.objects[object];
            }
            object -= scene.objects.size();
            for(const auto& assembly : pool.assemblies){
                if(object < assembly.objects.size()){
                    return &assembly.objects[object];
                }
                object -= assembly.objects.size();
            }
        }
        return nullptr;
    }

    inline void make_transform(const float position[3], const float orientation_wxyz[4], float out[12]){
        const float w = orientation_wxyz[0], x = orientation_wxyz[1], y = orientation_wxyz[2], z = orientation_wxyz[3];
        out[0] = 1 - 2*(y*y + z*z); out[1] = 2*(x*y - w*z);     out[2] = 2*(x*z + w*y);     out[3] = position[0];
        out[4] = 2*(x*y + w*z);     out[5] = 1 - 2*(x*x + z*z); out[6] = 2*(y*z - w*x);     out[7] = position[1];
        out[8] = 2*(x*z - w*y);     out[9] = 2*(y*z + w*x);     out[10] = 1 - 2*(x*x + y*y); out[11] = position[2];
    }

    inline void compose_transforms(const float a[12], const float b[12], float out[12]){
        rendering::raytracing::detail::compose_transforms(a, b, out);
    }


    namespace rendering::raytracing::detail{
        template <typename SPEC, typename DEVICE>
        std::vector<float> generate_probe_direction_vectors(DEVICE& device){
            std::vector<float> dirs;
            dirs.reserve(SPEC::NUM_PROBES * 3);

            const float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;

            for(int i = 0; i < (int)SPEC::NUM_PROBES; i++){
                float theta = 2.0f * (float)M_PI * i / golden_ratio;
                float cos_inc = 1.0f - 2.0f * (i + 0.5f) / SPEC::NUM_PROBES;
                float sin_inc = sqrtf(1.0f - cos_inc * cos_inc);

                const float dir[3] = {sin_inc * cosf(theta), sin_inc * sinf(theta), cos_inc};
                const float inv_len = 1.0f / sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
                dirs.push_back(dir[0] * inv_len);
                dirs.push_back(dir[1] * inv_len);
                dirs.push_back(dir[2] * inv_len);
            }

            log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "Generated ", SPEC::NUM_PROBES, " probe directions per camera");
            return dirs;
        }

        // one distinct color per instance id (golden-ratio hue hash); miss sentinel renders black
        inline uint32_t segmentation_id_to_rgba(uint32_t id){
            if(id == 0xFFFFFFFFu){
                return 0xFF000000u;
            }
            const float hue = std::fmod((float)id * 0.61803398875f, 1.0f) * 6.0f;
            const float descending = 1.0f - std::fabs(std::fmod(hue, 2.0f) - 1.0f);
            float r = 0, g = 0, b = 0;
            switch((int)hue){
                case 0: r = 1; g = descending; break;
                case 1: r = descending; g = 1; break;
                case 2: g = 1; b = descending; break;
                case 3: g = descending; b = 1; break;
                case 4: r = descending; b = 1; break;
                default: r = 1; b = descending; break;
            }
            return 0xFF000000u | ((uint32_t)(b * 255) << 16) | ((uint32_t)(g * 255) << 8) | (uint32_t)(r * 255);
        }
        // pinned normals encoding, the single owner shared by save verbs and the golden corpus:
        // each component maps through round((clamp(n, -1, 1) * 0.5 + 0.5) * 255). The miss value
        // (0,0,0) encodes to (128,128,128), which no unit normal can reach (a unit vector has a
        // component of magnitude >= 1/sqrt(3)).
        inline uint32_t normal_to_rgba(const float normal[3]){
            uint32_t rgba = 0xFF000000u;
            for(int component = 0; component < 3; component++){
                const float clamped = fminf(fmaxf(normal[component], -1.f), 1.f);
                const uint32_t value = (uint32_t)lroundf((clamped * 0.5f + 0.5f) * 255.f);
                rgba |= value << (8 * component);
            }
            return rgba;
        }
        // pinned flow-status false colour shared by the save verbs and the golden corpus (ABGR
        // packed like the frame buffer): VALID white, MISS black, BEHIND magenta, OUT_OF_FRAME
        // blue, OCCLUDED orange; unknown codes red
        // direction -> hue, magnitude -> saturation against magnitude_scale (pixels); the review
        // writers pass the per-image maximum, live viewers a fixed scale so the image does not flicker
        inline uint32_t flow_to_rgba(float u, float v, float magnitude_scale){
            constexpr float TWO_PI = 6.28318530717958647692f;
            const float magnitude = std::sqrt(u * u + v * v);
            const float saturation = magnitude_scale > 0 ? std::fmin(magnitude / magnitude_scale, 1.f) : 0.f;
            const float hue = (std::atan2(v, u) / TWO_PI + 0.5f) * 6.f;
            const float descending = 1.f - std::fabs(std::fmod(hue, 2.f) - 1.f);
            float red = 0, green = 0, blue = 0;
            switch((int)hue % 6){
                case 0: red = 1; green = descending; break;
                case 1: red = descending; green = 1; break;
                case 2: green = 1; blue = descending; break;
                case 3: green = descending; blue = 1; break;
                case 4: red = descending; blue = 1; break;
                default: red = 1; blue = descending; break;
            }
            const auto channel = [&](float value){ return (uint32_t)((1.f - saturation * (1.f - value)) * 255.f); };
            return 0xFF000000u | (channel(blue) << 16) | (channel(green) << 8) | channel(red);
        }

        inline uint32_t flow_status_to_rgba(uint32_t status){
            switch(status){
                case FlowStatus::VALID: return 0xFFFFFFFFu;
                case FlowStatus::MISS: return 0xFF000000u;
                case FlowStatus::BEHIND: return 0xFFFF00FFu;
                case FlowStatus::OUT_OF_FRAME: return 0xFFFF4020u;
                case FlowStatus::OCCLUDED: return 0xFF0090FFu;
                default: return 0xFF0000FFu;
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
