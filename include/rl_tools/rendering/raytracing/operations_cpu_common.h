#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H

#include "renderer.h"
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

#define RL_TOOLS_RENDERING_RAYTRACING_LOG(message) do { std::cout << "\033[0;34m" << "#rl_tools::rendering::raytracing: " << message << "\033[0m" << std::endl; } while(false)
#define RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR(message) do { std::cerr << "\033[0;31m" << "#rl_tools::rendering::raytracing: " << message << "\033[0m" << std::endl; } while(false)

#ifndef RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // backend-generic launch+sync composite, defined in operations_cpu_post.h; declared here so
    // backend-internal callers (init) resolve it before the definition is included
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer);

    namespace rendering::raytracing::detail{
        template <typename BACKEND>
        void announce_backend(){
            std::fprintf(
                stderr,
                "#rl_tools::rendering::raytracing: backend=%s\n",
                rendering::raytracing::backends::name<BACKEND>()
            );
            std::fflush(stderr);
        }

        template <typename SPEC, typename BACKEND>
        void announce_backend(const rendering::raytracing::Renderer<SPEC, BACKEND>&){
            announce_backend<BACKEND>();
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

    // stages host-verb writes (the per-slot transform_entry mirrors) into the transforms tensor;
    // backends whose tensor is host-resident call this at the top of update(). Device producers
    // write the tensor directly and are not staged — a dirty overlay row is owned by the host.
    template <typename SPEC, typename BACKEND>
    void flush_overlay_transforms(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        float* transforms = data(renderer.transforms);
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_state = renderer.overlays[overlay];
            if(!overlay_state.dirty) continue;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                std::memcpy(transforms + ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12, overlay_state.slots[slot].transform_entry, 12 * sizeof(float));
            }
            overlay_state.dirty = false;
        }
    }

    // world = pose ∘ part_local ∘ articulation: the root slot's tensor entry carries the
    // placement pose, non-root entries articulate their part in the part frame. The
    // transforms_base overload lets the dynamic-motion-blur loop compose from a per-sample
    // slab of transforms_motion, which shares the transforms tensor layout.
    template <typename SPEC, typename BACKEND>
    void compose_overlay_slot_transform(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const float* transforms_base, typename SPEC::TI overlay, typename SPEC::TI slot_index, float out[12]){
        const auto& slot = renderer.overlays[overlay].slots[slot_index];
        const float* row = transforms_base + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES * 12;
        float composed[12];
        compose_transforms(row + (size_t)slot.pose_slot * 12, slot.part_local, composed);
        if(slot_index == slot.pose_slot){
            std::memcpy(out, composed, sizeof(composed));
        }
        else{
            compose_transforms(composed, row + (size_t)slot_index * 12, out);
        }
    }

    template <typename SPEC, typename BACKEND>
    void compose_overlay_slot_transform(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI overlay, typename SPEC::TI slot_index, float out[12]){
        compose_overlay_slot_transform(renderer, data(renderer.transforms), overlay, slot_index, out);
    }

    // invert_transform lives in transforms_generic.h (shared with the device kernels)

    // writes one entry into every motion sample of the staging mirror (constant across the
    // shutter = sharp); the single-pose verbs route through this so a dynamic-motion-blur spec
    // driven only by them renders identically to camera-only blur
    template <typename SPEC, typename BACKEND>
    void stage_motion_entry(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI overlay, typename SPEC::TI slot, const float entry[12]){
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(typename SPEC::TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                std::memcpy(renderer.transforms_motion_staging.data() + sample * SLAB + ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12, entry, 12 * sizeof(float));
            }
            renderer.transforms_motion_dirty[overlay] = true;
        }
    }

    // composes the flow shutter-delta table (world_open ∘ world_close⁻¹ per slot; identity for
    // inactive slots) from open/close entry slabs sharing the transforms-tensor layout.
    // only_dirty mirrors the flush_overlay_transforms ownership contract: a clean overlay row
    // may be producer-written and must not be clobbered from the host mirrors.
    template <typename SPEC, typename BACKEND>
    void compose_flow_deltas_from_slabs(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const float* open_slab, const float* close_slab, float* deltas, bool only_dirty){
        using TI = typename SPEC::TI;
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            if(only_dirty && !renderer.overlays[overlay].dirty) continue;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                float* delta = deltas + ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12;
                if(!renderer.overlays[overlay].slots[slot].active){
                    std::memcpy(delta, IDENTITY_TRANSFORM, 12 * sizeof(float));
                    continue;
                }
                float world_open[12], world_close[12], world_close_inverse[12];
                compose_overlay_slot_transform(renderer, open_slab, overlay, slot, world_open);
                compose_overlay_slot_transform(renderer, close_slab, overlay, slot, world_close);
                invert_transform(world_close, world_close_inverse);
                compose_transforms(world_open, world_close_inverse, delta);
            }
        }
    }

    // CPU expansion of the transforms_pair tensor for backends whose tensors are host-resident
    // (generic/Vulkan); OptiX runs the same math on-device (overlay_accel_expand_motion). The
    // slerp into transforms_motion only applies under dynamic motion blur; flow-only
    // specifications expand the pair into transforms (close) and flow_deltas.
    // Producer-style: writes the tensors directly with no dirty flags, so the host-verb flush
    // never clobbers it — per overlay, use either the set_transform* verbs or the pair path.
    template <typename SPEC, typename BACKEND>
    void expand_motion_transforms_host(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        const float* pairs = data(renderer.transforms_pair);
        float* transforms = data(renderer.transforms);
        for(size_t slot = 0; slot < SLOTS; slot++){
            const float* close = pairs + (SLOTS + slot) * 12;
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                const float* open = pairs + slot * 12;
                float* transforms_motion = data(renderer.transforms_motion);
                for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                    const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                    slerp_transform(open, close, shutter_t, transforms_motion + ((size_t)sample * SLOTS + slot) * 12);
                }
            }
            std::memcpy(transforms + slot * 12, close, 12 * sizeof(float));
        }
        if constexpr (SPEC::HAS_FLOW){
            compose_flow_deltas_from_slabs(renderer, pairs, pairs + SLOTS * 12, data(renderer.flow_deltas), false);
        }
    }

    // host-verb path, called before flush_overlay_transforms consumes the dirty flags: the
    // slabs come from the per-slot mirrors the verbs maintain
    template <typename SPEC, typename BACKEND>
    void compose_flow_deltas(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, float* deltas){
        using TI = typename SPEC::TI;
        std::vector<float> open_slab((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12);
        std::vector<float> close_slab(open_slab.size());
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                const size_t offset = ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12;
                std::memcpy(open_slab.data() + offset, renderer.overlays[overlay].slots[slot].transform_entry_open, 12 * sizeof(float));
                std::memcpy(close_slab.data() + offset, renderer.overlays[overlay].slots[slot].transform_entry, 12 * sizeof(float));
            }
        }
        compose_flow_deltas_from_slabs(renderer, open_slab.data(), close_slab.data(), deltas, true);
    }

    // stages host-verb writes into the transforms_motion tensor, mirroring
    // flush_overlay_transforms; backends with a host-resident tensor call it in update()
    template <typename SPEC, typename BACKEND>
    void flush_overlay_motion_transforms(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
        float* transforms_motion = data(renderer.transforms_motion);
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            if(!renderer.transforms_motion_dirty[overlay]) continue;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const size_t offset = sample * SLAB + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES * 12;
                std::memcpy(transforms_motion + offset, renderer.transforms_motion_staging.data() + offset, (size_t)SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
            }
            renderer.transforms_motion_dirty[overlay] = false;
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
        for(auto& overlay_state : renderer.overlays){
            for(auto& slot : overlay_state.slots){
                slot.active = false;
            }
            overlay_state.dirty = true;
        }
        for(TI attachment_i = 0; attachment_i < SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA; attachment_i++){
            renderer.attachments[attachment_i] = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
        }
        renderer.attachments_dirty = true;
    }

    // resolved-configuration echo: makes a silently-defaulted (e.g. misspelled) fringe
    // config member visible on the first run
    template <typename SPEC>
    void announce_configuration(){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("config: " << SPEC::NUM_CAMERAS << " camera(s) " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT
            << " outputs[rgb=" << SPEC::HAS_RGB << " depth=" << SPEC::HAS_DEPTH << " segmentation=" << SPEC::HAS_SEGMENTATION << (SPEC::SEMANTIC_SEGMENTATION ? " (semantic)" : "") << " normals=" << SPEC::HAS_NORMALS << " flow=" << SPEC::HAS_FLOW << "]"
            << " probes=" << SPEC::NUM_PROBES
            << " motion_blur_samples=" << (SPEC::ENABLE_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 0)
            << " anti_aliasing_grid=" << (SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 0)
            << " overlays=" << SPEC::NUM_OVERLAYS << "x" << SPEC::MAX_OVERLAY_INSTANCES << " overlays_per_camera=" << SPEC::MAX_OVERLAYS_PER_CAMERA);
    }

    // Lights as uploaded to the device: only the PBR tiers consume punctual lights. Object lights
    // are authored in the object's local frame and follow the instance placing it.
    template <bool APPLY>
    std::vector<rendering::raytracing::SceneLight> effective_scene_lights(const rendering::raytracing::Scene& scene){
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
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Scene lights: " << lights.size());
            for (size_t li = 0; li < lights.size(); li++) {
                auto& sl = lights[li];
                RL_TOOLS_RENDERING_RAYTRACING_LOG("  light " << li << ": pos=(" << sl.position[0] << "," << sl.position[1] << "," << sl.position[2]
                    << ") dir=(" << sl.direction[0] << "," << sl.direction[1] << "," << sl.direction[2]
                    << ") color=(" << sl.color[0] << "," << sl.color[1] << "," << sl.color[2] << ")");
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
            using TI = typename SPEC::TI;
            utils::assert_exit(device, !bundle.metadata.content_hash.empty(), "library: bundle has no content hash — load it through a dataset loader");
            for(TI scene_i = 0; scene_i < (TI)library.metadata.size(); scene_i++){
                if(library.metadata[scene_i].content_hash == bundle.metadata.content_hash){
                    is_new = false;
                    RL_TOOLS_RENDERING_RAYTRACING_LOG("library: scene " << bundle.metadata.content_hash << " shares build " << scene_i << " (content hash match)");
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

        for(TI part = 0; part < record.num_parts && first_slot + part < SPEC::MAX_OVERLAY_INSTANCES; part++){
            auto& slot = state.slots[first_slot + part];
            slot.object = renderer.asset_part_objects[record.first_part + part];
            slot.pose_slot = first_slot;
            std::memcpy(slot.part_local, &renderer.asset_part_transforms[(record.first_part + part) * 12], sizeof(slot.part_local));
            std::memcpy(slot.transform_entry, part == 0 ? transform : rendering::raytracing::detail::IDENTITY_TRANSFORM, sizeof(slot.transform_entry));
            std::memcpy(slot.transform_entry_open, slot.transform_entry, sizeof(slot.transform_entry_open));
            slot.active = true;
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, first_slot + part, slot.transform_entry);
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

    // per-part: part 0 sets the placement pose, other parts articulate in their part frame
    // (world = pose ∘ part_local ∘ articulation)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, typename SPEC::TI part, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transform requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot + part].transform_entry, transform, 12 * sizeof(float));
        std::memcpy(state.slots[placement.first_slot + part].transform_entry_open, transform, 12 * sizeof(float));
        state.dirty = true;
        rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), transform);
    }

    // rigid move: sets the placement pose and resets per-part articulation state
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transform requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot].transform_entry, transform, 12 * sizeof(float));
        std::memcpy(state.slots[placement.first_slot].transform_entry_open, transform, 12 * sizeof(float));
        rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)placement.first_slot, transform);
        for(size_t part = 1; part < placement.num_parts; part++){
            std::memcpy(state.slots[placement.first_slot + part].transform_entry, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            std::memcpy(state.slots[placement.first_slot + part].transform_entry_open, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), rendering::raytracing::detail::IDENTITY_TRANSFORM);
        }
        state.dirty = true;
    }

    // dynamic motion blur: per-part shutter-open/close entries, slerped into every motion sample
    // at the same midpoint shutter times the camera lerp uses; the close entry also becomes the
    // steady-state transform (segmentation, probes, and the next frame render at shutter close)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, typename SPEC::TI part, const float open[12], const float close[12]){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "set_transform_pair requires a dynamic-motion-blur or flow renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot + part].transform_entry, close, 12 * sizeof(float));
        std::memcpy(state.slots[placement.first_slot + part].transform_entry_open, open, 12 * sizeof(float));
        state.dirty = true;
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                float entry[12];
                rendering::raytracing::detail::slerp_transform(open, close, shutter_t, entry);
                std::memcpy(renderer.transforms_motion_staging.data() + sample * SLAB + ((size_t)overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part) * 12, entry, 12 * sizeof(float));
            }
            renderer.transforms_motion_dirty[overlay.index] = true;
        }
    }

    // rigid move with shutter-open/close poses: resets articulation in every sample
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float open[12], const float close[12]){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "set_transform_pair requires a dynamic-motion-blur or flow renderer specification");
        using TI = typename SPEC::TI;
        set_transform_pair(device, renderer, overlay, placement, (TI)0, open, close);
        auto& state = renderer.overlays[overlay.index];
        for(size_t part = 1; part < placement.num_parts; part++){
            std::memcpy(state.slots[placement.first_slot + part].transform_entry, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            std::memcpy(state.slots[placement.first_slot + part].transform_entry_open, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), rendering::raytracing::detail::IDENTITY_TRANSFORM);
        }
        state.dirty = true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "transforms requires an overlay-enabled renderer specification");
        return renderer.transforms;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_motion(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "transforms_motion requires a dynamic-motion-blur renderer specification");
        return renderer.transforms_motion;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "transforms_pair requires a dynamic-motion-blur or flow renderer specification");
        return renderer.transforms_pair;
    }

    // camera input tensors, backend-native residency like transforms: device memory on OptiX,
    // host on generic, shared/mapped on Metal/Vulkan. Producers write them via rlt::copy or
    // kernels; the launch verbs consume them directly. Under motion blur the pair is
    // cameras_open (shutter open) and cameras_close (shutter close, aliasing cameras) — both
    // must be written each step (identical values for a blur-free frame).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_open(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_open requires a camera-pair renderer specification (motion blur or flow)");
        return renderer.cameras_open;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_close(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_close requires a camera-pair renderer specification (motion blur or flow)");
        return renderer.cameras;
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

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW, "flow_buffer requires a flow-capable renderer specification");
        return renderer.flow_buffer;
    }

    // producer input like transforms: a clean (non-dirty) overlay's rows are producer-owned and
    // never clobbered by the host-verb composition in update(); the pair expansion writes it too
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& flow_deltas(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW && SPEC::ENABLE_OVERLAYS, "flow_deltas requires a flow renderer specification with overlays");
        return renderer.flow_deltas;
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
        template <typename SPEC>
        std::vector<float> generate_probe_direction_vectors(){
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

            RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << SPEC::NUM_PROBES << " probe directions per camera");
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
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
