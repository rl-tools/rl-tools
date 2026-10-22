#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_OPERATIONS_CPU_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_OPERATIONS_CPU_COMMON_H

#include "hyperlux.h"
#include "segmentation_cpu.h"
#include "transforms_generic.h"
#include "pose_generic.h"

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


#include "outputs/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace hyperlux{
        inline constexpr char LOG_PREFIX[] = "#rl_tools::hyperlux: ";
    }

    template <typename SPEC, typename DEVICE>
    void validate_semantic_segmentation(DEVICE& device, const hyperlux::Scene& scene, const hyperlux::AssetPool& pool){
        if constexpr(SPEC::HAS_SEMANTIC_SEGMENTATION){
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
    void update(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer);

    // one past the largest joint coordinate (0 for a rigid assembly)
    template <typename DEVICE>
    size_t num_coordinates(DEVICE& device, const hyperlux::ObjectAssembly& assembly){
        size_t count = 0;
        for(const auto& part : assembly.parts){
            if(part.joint.type != hyperlux::Joint::Type::FIXED && part.joint.coordinate + 1 > count){
                count = part.joint.coordinate + 1;
            }
        }
        return count;
    }

    // what registering an assembly requires: parts with finite frames on existing objects,
    // parents listed first, joints with finite nonzero axes and finite pivots, coordinates below
    // Joint::MAX_COORDINATES and every one below num_coordinates driving at least one joint
    template <typename DEVICE>
    bool can_add(DEVICE& device, const hyperlux::AssetPool&, const hyperlux::ObjectAssembly& assembly){
        if(assembly.parts.empty()){
            return false;
        }
        bool driven[hyperlux::Joint::MAX_COORDINATES] = {};
        size_t count = 0;
        for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
            const auto& part = assembly.parts[part_i];
            if(part.object >= assembly.objects.size() || (part.parent != hyperlux::ObjectAssembly::Part::NO_PARENT && part.parent >= part_i) || !hyperlux::is_finite(part.transform, 12)){
                return false;
            }
            const auto& joint = part.joint;
            if(joint.type == hyperlux::Joint::Type::FIXED){
                continue;
            }
            if(joint.type != hyperlux::Joint::Type::REVOLUTE && joint.type != hyperlux::Joint::Type::PRISMATIC){
                return false;
            }
            const float norm = std::sqrt(joint.axis[0] * joint.axis[0] + joint.axis[1] * joint.axis[1] + joint.axis[2] * joint.axis[2]);
            if(joint.coordinate >= hyperlux::Joint::MAX_COORDINATES || !hyperlux::is_finite(joint.axis, 3) || !hyperlux::is_finite(joint.pivot, 3) || !hyperlux::is_finite(norm) || norm <= 0){
                return false;
            }
            driven[joint.coordinate] = true;
            count = joint.coordinate + 1 > count ? joint.coordinate + 1 : count;
        }
        for(size_t coordinate = 0; coordinate < count; coordinate++){
            if(!driven[coordinate]){
                return false;
            }
        }
        return true;
    }

    namespace hyperlux::detail{
        template <typename DEVICE, typename SPEC, typename BACKEND>
        void announce_backend(DEVICE& device, const HyperLux<SPEC, BACKEND>&){
            log(device, device.logger, devices::logging::Verbosity::info, hyperlux::LOG_PREFIX, "backend=", hyperlux::backends::name<BACKEND>());
        }
    }

    // =========================================================================
    // Default cube geometry
    // =========================================================================
    namespace hyperlux::constants{
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

    namespace hyperlux::detail{

    inline constexpr float IDENTITY_TRANSFORM[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    // Host staging of the overlay transform inputs. The verbs write the staging mirrors and mark
    // rows pending; update() publishes exactly the pending rows into the resident tensors in the
    // backend's memory domain and clears the marks. Producers write the resident tensors
    // directly and are never touched by the publish — a row belongs to whoever wrote it last.
    template <typename SPEC, typename BACKEND>
    void malloc_overlay_staging(HyperLux<SPEC, BACKEND>& renderer){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        renderer.transforms_staging.assign(SLOTS * 12, 0.0f);
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            renderer.transforms_previous_staging.assign(SLOTS * 12, 0.0f);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            renderer.transforms_motion_staging.assign((size_t)SPEC::MOTION_BLUR_SAMPLES * SLOTS * 12, 0.0f);
        }
    }

    template <typename SPEC, typename BACKEND>
    void stage_previous(HyperLux<SPEC, BACKEND>& renderer, size_t slot, const float pose[12]){
        std::memcpy(renderer.transforms_previous_staging.data() + slot * 12, pose, 12 * sizeof(float));
        renderer.transforms_previous_pending[slot] = true;
    }

    template <typename SPEC, typename BACKEND>
    hyperlux::PoseSink<SPEC> staging_sink(HyperLux<SPEC, BACKEND>& renderer){
        hyperlux::PoseSink<SPEC> sink;
        sink.transforms = renderer.transforms_staging.data();
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            sink.transforms_motion = renderer.transforms_motion_staging.data();
        }
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            sink.transforms_previous = renderer.transforms_previous_staging.data();
        }
        sink.parts = renderer.asset_parts.data();
        return sink;
    }

    template <typename SPEC, typename BACKEND>
    void stage_pose(HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement){
        for(size_t part = 0; part < placement.num_parts; part++){
            const size_t slot = overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part;
            renderer.transforms_pending[slot] = true;
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                renderer.transforms_motion_pending[slot] = true;
            }
        }
    }

    // publish for backends whose resident tensors are host-addressable (after their in-flight
    // wait); OptiX scatters the same pending rows on its stream (optix::publish_rows)
    template <typename SPEC, typename BACKEND>
    void publish_overlay_transforms(HyperLux<SPEC, BACKEND>& renderer){
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
    void advance_flow_staging(HyperLux<SPEC, BACKEND>& renderer){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        for(size_t slot = 0; slot < SLOTS; slot++){
            if(renderer.transforms_pending[slot]){
                stage_previous(renderer, slot, renderer.transforms_staging.data() + slot * 12);
            }
        }
    }

    template <typename SPEC, typename BACKEND>
    bool placement_spawned(const HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement){
        if(overlay.index >= SPEC::NUM_OVERLAYS || placement.num_parts == 0 || placement.first_slot >= SPEC::MAX_OVERLAY_INSTANCES || placement.num_parts > SPEC::MAX_OVERLAY_INSTANCES - placement.first_slot){
            return false;
        }
        for(size_t part = 0; part < placement.num_parts; part++){
            if(!renderer.overlays[overlay.index].slots[placement.first_slot + part].active){
                return false;
            }
        }
        return true;
    }

    // spawned and holding the asset parts it names (replace hands out a new placement)
    template <typename SPEC, typename BACKEND>
    bool placement_valid(const HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement){
        if(!placement_spawned(renderer, overlay, placement) || placement.first_part > renderer.asset_parts.size() || placement.num_parts > renderer.asset_parts.size() - placement.first_part){
            return false;
        }
        for(size_t part = 0; part < placement.num_parts; part++){
            if(renderer.overlays[overlay.index].slots[placement.first_slot + part].object != renderer.asset_parts[placement.first_part + part].object){
                return false;
            }
        }
        return true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void assert_placement(DEVICE& device, const HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement){
        utils::assert_exit(device, placement_valid(renderer, overlay, placement), "overlay verb: placement out of range, not spawned or replaced");
    }

    // joint coordinates move a placement only through its asset's joints, which bind when the
    // assembly enters the pool
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void assert_coordinates(DEVICE& device, const HyperLux<SPEC, BACKEND>& renderer, const hyperlux::OverlayPlacement& placement, bool coordinates){
        bool articulated = false;
        for(size_t part = 0; part < placement.num_parts; part++){
            articulated = articulated || renderer.asset_parts[placement.first_part + part].joint.type != hyperlux::Joint::Type::FIXED;
        }
        utils::assert_exit(device, !coordinates || articulated, "set_pose: joint coordinates for a placement whose asset has no joints");
    }

    // per-slot flow delta table [2][SLOTS][12] from two transforms-layout slabs: slab 0 carries a
    // current-instant point to the previous instant (world_previous ∘ world_current⁻¹, the
    // backward pass), slab 1 the inverse (forward pass); identity for inactive slots. Backends
    // with host-resident tensors call it in update() after the flush; OptiX runs the same
    // composition on-device (overlay_accel_compose_flow_deltas).
    template <typename SPEC, typename BACKEND>
    void compose_flow_deltas(const HyperLux<SPEC, BACKEND>& renderer, const float* current_slab, const float* previous_slab, float* deltas){
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

    // an assembly's parts as the pose evaluation reads them (see AssetPart), objects offset by
    // object_base; requires can_add
    inline void flatten_parts(const hyperlux::ObjectAssembly& assembly, size_t object_base, std::vector<hyperlux::AssetPart>& parts){
        using hyperlux::AssetPart;
        const size_t first = parts.size();
        for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
            const auto& part = assembly.parts[part_i];
            AssetPart flat;
            flat.object = object_base + part.object;
            flat.joint = part.joint;
            if(flat.joint.type != hyperlux::Joint::Type::FIXED){
                const float* axis = flat.joint.axis;
                const float norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
                for(auto& component : flat.joint.axis){
                    component /= norm;
                }
            }
            if(part.parent != hyperlux::ObjectAssembly::Part::NO_PARENT){
                const auto& parent = parts[first + part.parent];
                flat.reference = assembly.parts[part.parent].joint.type != hyperlux::Joint::Type::FIXED ? part.parent : parent.reference;
            }
            if(flat.reference == AssetPart::NO_REFERENCE){
                std::memcpy(flat.origin, part.transform, sizeof(flat.origin));
            }
            else{
                float reference_inverse[12];
                invert_transform(assembly.parts[flat.reference].transform, reference_inverse);
                compose_transforms(reference_inverse, part.transform, flat.origin);
            }
            parts.push_back(flat);
        }
    }

    // flattens the asset pool for overlay spawns: appends pool objects to the combined object
    // list and records the asset parts on the renderer
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void register_pool_assets(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, const hyperlux::AssetPool& pool, std::vector<const hyperlux::Object*>& all_objects){
        using TI = typename SPEC::TI;
        renderer.assets.clear();
        renderer.asset_parts.clear();
        for(const auto& assembly : pool.assemblies){
            utils::assert_exit(device, can_add(device, pool, assembly), "asset: invalid parts or joints (see can_add)");
            typename HyperLux<SPEC, BACKEND>::AssetRecord record;
            record.first_part = (TI)renderer.asset_parts.size();
            record.num_parts = (TI)assembly.parts.size();
            record.num_coordinates = (TI)num_coordinates(device, assembly);
            utils::assert_exit(device, record.num_parts <= SPEC::MAX_OVERLAY_INSTANCES, "asset has more parts than the overlay capacity");
            flatten_parts(assembly, all_objects.size(), renderer.asset_parts);
            for(const auto& object : assembly.objects){
                all_objects.push_back(&object);
            }
            renderer.assets.push_back(record);
        }
    }

    // Deterministic first-fit is a contract, not an implementation detail: the chosen slot defines
    // the global instance id (segmentation output), which must be reproducible across runs and
    // identical across backends. Do not replace with a free-list or best-fit strategy.
    template <typename SPEC, typename BACKEND>
    typename SPEC::TI first_fit_slot(const HyperLux<SPEC, BACKEND>& renderer, size_t overlay, typename SPEC::TI num_parts){
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
    void reset_overlay_state(HyperLux<SPEC, BACKEND>& renderer){
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
            renderer.attachments[attachment_i] = HyperLux<SPEC, BACKEND>::INVALID_OVERLAY;
        }
        renderer.attachments_dirty = true;
    }

    // resolved-configuration echo at debug verbosity: makes a silently-defaulted (e.g.
    // misspelled) fringe config member visible
    template <typename SPEC, typename DEVICE>
    void announce_configuration(DEVICE& device){
        log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::LOG_PREFIX, "config: ", SPEC::NUM_CAMERAS, " camera(s) ", SPEC::CAM_WIDTH, "x", SPEC::CAM_HEIGHT,
            " outputs[rgb=", SPEC::HAS_RGB, " depth=", SPEC::HAS_DEPTH, " segmentation=", SPEC::HAS_SEGMENTATION, (SPEC::HAS_SEMANTIC_SEGMENTATION ? " (semantic)" : ""), " normals=", SPEC::HAS_NORMALS, " flow=", SPEC::HAS_FLOW, " flow_forward=", SPEC::HAS_FLOW_FORWARD, "]",
            " probes=", SPEC::NUM_PROBES,
            " motion_blur_samples=", (SPEC::ENABLE_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 0),
            " anti_aliasing_grid=", (SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 0),
            " overlays=", SPEC::NUM_OVERLAYS, "x", SPEC::MAX_OVERLAY_INSTANCES, " overlays_per_camera=", SPEC::MAX_OVERLAYS_PER_CAMERA);
    }

    // resolves a Scene::Environment into the concrete ambient + miss color pair the shaders
    // consume. DEFAULT reproduces the historical constants exactly (golden-pinned); GRADIENT and
    // EQUIRECT fall back to a solid horizon background until the shaders consume them.
    template <typename SPEC>
    void resolve_environment(const hyperlux::Scene::Environment& environment, typename SPEC::T ambient_color[3], typename SPEC::T miss_color_0[3], typename SPEC::T miss_color_1[3]){
        using T = typename SPEC::T;
        using Environment = hyperlux::Scene::Environment;
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
    std::vector<hyperlux::SceneLight> effective_scene_lights(DEVICE& device, const hyperlux::Scene& scene){
        std::vector<hyperlux::SceneLight> lights;
        if constexpr (APPLY) {
            lights = scene.lights;
            for(const auto& instance : scene.instances){
                for(const auto& object_light : scene.objects[instance.object].lights){
                    hyperlux::SceneLight light = object_light;
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
            log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::LOG_PREFIX, "Scene lights: ", lights.size());
            for(size_t li = 0; li < lights.size(); li++){
                const auto& sl = lights[li];
                log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::LOG_PREFIX, "  light ", li, ": pos=(", sl.position[0], ",", sl.position[1], ",", sl.position[2],
                    ") dir=(", sl.direction[0], ",", sl.direction[1], ",", sl.direction[2],
                    ") color=(", sl.color[0], ",", sl.color[1], ",", sl.color[2], ")");
            }
        }
        return lights;
    }
    } // namespace hyperlux::detail

    namespace hyperlux::detail{
        // content-hash dedup insert: on a miss the library takes ownership of the bundle's scene
        // (the bundle's metadata stays with the caller for producer-side use)
        template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
        typename SPEC::TI library_insert(DEVICE& device, hyperlux::AssetLibrary<SPEC, BACKEND>& library, hyperlux::Bundle<T>& bundle, bool& is_new){
            validate_semantic_segmentation<SPEC>(device, bundle.scene, library.pool);
            using TI = typename SPEC::TI;
            utils::assert_exit(device, !bundle.metadata.content_hash.empty(), "library: bundle has no content hash — load it through a dataset loader");
            for(TI scene_i = 0; scene_i < (TI)library.metadata.size(); scene_i++){
                if(library.metadata[scene_i].content_hash == bundle.metadata.content_hash){
                    is_new = false;
                    log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::LOG_PREFIX, "library: scene ", bundle.metadata.content_hash, " shares build ", scene_i, " (content hash match)");
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
    typename SPEC::TI insert(DEVICE& device, hyperlux::AssetLibrary<SPEC, BACKEND>& library, hyperlux::Bundle<T>& bundle){
        bool is_new = false;
        return hyperlux::detail::library_insert(device, library, bundle, is_new);
    }

    // appends a root part (geometry in the part frame, placed by transform within the assembly)
    template <typename DEVICE>
    size_t add(DEVICE& device, hyperlux::ObjectAssembly& assembly, const hyperlux::Object& object, const float transform[12]){
        hyperlux::ObjectAssembly::Part part;
        part.object = assembly.objects.size();
        std::memcpy(part.transform, transform, sizeof(part.transform));
        assembly.objects.push_back(object);
        assembly.parts.push_back(part);
        return assembly.parts.size() - 1;
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, hyperlux::Scene& scene, const hyperlux::Object& object, const float transform[12]){
        scene.objects.push_back(object);
        scene.instances.push_back({scene.objects.size() - 1, {}, hyperlux::detail::transform_is_identity(transform)});
        std::memcpy(scene.instances.back().transform, transform, 12 * sizeof(float));
        return scene.instances.size() - 1;
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, hyperlux::Scene& scene, const hyperlux::Object& object){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, object, identity);
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, hyperlux::Scene& scene, const hyperlux::Mesh& mesh){
        hyperlux::Object object;
        object.meshes.push_back(mesh);
        return add(device, scene, object);
    }


    template <typename DEVICE>
    hyperlux::Placement add(DEVICE& device, hyperlux::Scene& scene, const hyperlux::ObjectAssembly& assembly, const float transform[12]){
        hyperlux::Placement placement{scene.instances.size(), assembly.parts.size()};
        const size_t object_base = scene.objects.size();
        scene.objects.insert(scene.objects.end(), assembly.objects.begin(), assembly.objects.end());
        for(const auto& part : assembly.parts){
            float composed[12];
            hyperlux::detail::compose_transforms(transform, part.transform, composed);
            scene.instances.push_back({object_base + part.object, {}, hyperlux::detail::transform_is_identity(composed)});
            std::memcpy(scene.instances.back().transform, composed, sizeof(composed));
        }
        return placement;
    }

    template <typename DEVICE>
    hyperlux::Placement add(DEVICE& device, hyperlux::Scene& scene, const hyperlux::ObjectAssembly& assembly){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, assembly, identity);
    }

    template <typename DEVICE>
    hyperlux::AssetHandle add(DEVICE& device, hyperlux::AssetPool& pool, const hyperlux::ObjectAssembly& assembly){
        pool.assemblies.push_back(assembly);
        return {pool.assemblies.size() - 1};
    }

    template <typename DEVICE>
    hyperlux::AssetHandle add(DEVICE& device, hyperlux::AssetPool& pool, const hyperlux::Object& object){
        hyperlux::ObjectAssembly assembly;
        add(device, assembly, object, hyperlux::detail::IDENTITY_TRANSFORM);
        return add(device, pool, assembly);
    }

    template <typename DEVICE>
    hyperlux::AssetHandle add(DEVICE& device, hyperlux::AssetPool& pool, const hyperlux::Mesh& mesh){
        hyperlux::Object object;
        object.meshes.push_back(mesh);
        return add(device, pool, object);
    }

    // Validation predicates for boundaries (language bindings, C interface) that must not trip
    // the fail-fast asserts inside the verbs: check first, then call.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_attach(DEVICE& device, const HyperLux<SPEC, BACKEND>& renderer, typename SPEC::TI camera, hyperlux::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        if(camera >= SPEC::NUM_CAMERAS || overlay.index >= SPEC::NUM_OVERLAYS){
            return false;
        }
        const TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index || row[slot] == HyperLux<SPEC, BACKEND>::INVALID_OVERLAY){
                return true;
            }
        }
        return false;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_spawn(DEVICE& device, const HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, hyperlux::AssetHandle asset){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_spawn requires an overlay-enabled renderer specification");
        if(overlay.index >= SPEC::NUM_OVERLAYS || asset.index >= renderer.assets.size()){
            return false;
        }
        return hyperlux::detail::first_fit_slot<SPEC>(renderer, overlay.index, renderer.assets[asset.index].num_parts) < SPEC::MAX_OVERLAY_INSTANCES;
    }

    // Overlay verbs mutate host-side truth on the renderer and mark it dirty; update(device,
    // renderer) is the single point where the backend consumes it (OptiX producers may also
    // rewrite the objects of active rows on the device, see overlay_structure, which host truth
    // and segmentation_object then do not reflect). All bookkeeping is deterministic: slot
    // allocation is a first-fit scan, so identical call sequences yield identical slots (and
    // therefore identical global instance ids).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void attach(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, typename SPEC::TI camera, hyperlux::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        utils::assert_exit(device, camera < SPEC::NUM_CAMERAS, "attach: camera index out of range");
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "attach: overlay index out of range");
        constexpr TI INVALID = HyperLux<SPEC, BACKEND>::INVALID_OVERLAY;
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
    void detach(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, typename SPEC::TI camera, hyperlux::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "detach requires an overlay-enabled renderer specification");
        utils::assert_exit(device, camera < SPEC::NUM_CAMERAS, "detach: camera index out of range");
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "detach: overlay index out of range");
        using TI = typename SPEC::TI;
        TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index){
                row[slot] = HyperLux<SPEC, BACKEND>::INVALID_OVERLAY;
                renderer.attachments_dirty = true;
                return;
            }
        }
    }

    // moves a placement: its rows from the pose and the joint coordinates (nullptr: the rest
    // state) through the asset's joints. Only the current instant changes: the flow pair advances
    // through advance_flow (or spawn)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_pose(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement, const hyperlux::Pose& pose, const float* coordinates = nullptr){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_pose requires an overlay-enabled renderer specification");
        hyperlux::detail::assert_placement(device, renderer, overlay, placement);
        hyperlux::detail::assert_coordinates(device, renderer, placement, coordinates != nullptr);
        set_pose(device, hyperlux::detail::staging_sink(renderer), overlay, placement, pose, coordinates);
        hyperlux::detail::stage_pose(renderer, overlay, placement);
    }

    // shutter-open/close states for dynamic motion blur, see the pose sink overload
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_pose(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement, const hyperlux::Pose& open, const hyperlux::Pose& close, const float* coordinates_open = nullptr, const float* coordinates_close = nullptr){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_pose with shutter-open/close states requires a dynamic-motion-blur renderer specification");
        hyperlux::detail::assert_placement(device, renderer, overlay, placement);
        hyperlux::detail::assert_coordinates(device, renderer, placement, coordinates_open != nullptr || coordinates_close != nullptr);
        set_pose(device, hyperlux::detail::staging_sink(renderer), overlay, placement, open, close, coordinates_open, coordinates_close);
        hyperlux::detail::stage_pose(renderer, overlay, placement);
    }

    // the resident rows for device producers (set_pose and advance_flow on the sink inside their
    // kernels) and for readers, in the backend's memory domain; valid from init until free
    template <typename DEVICE, typename SPEC, typename BACKEND>
    hyperlux::PoseSink<SPEC> pose_sink(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "pose_sink requires an overlay-enabled renderer specification");
        hyperlux::PoseSink<SPEC> sink;
        sink.transforms = data(renderer.transforms);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            sink.transforms_motion = data(renderer.transforms_motion);
        }
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            sink.transforms_previous = data(renderer.transforms_previous);
        }
        sink.parts = renderer.asset_parts_resident != nullptr ? renderer.asset_parts_resident : renderer.asset_parts.data();
        return sink;
    }

    // flow: the overlay's previous instant becomes the current instant as the host last wrote it
    // (an episode reset: no motion of its own in the next frame); its rows go pending for the
    // next update like set_pose's
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void advance_flow(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay){
        static_assert(SPEC::HAS_FLOW_OVERLAYS, "advance_flow of an overlay requires a flow renderer specification with overlays");
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "advance_flow: overlay index out of range");
        advance_flow(device, hyperlux::detail::staging_sink(renderer), overlay);
        for(size_t slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
            renderer.transforms_previous_pending[overlay.index * SPEC::MAX_OVERLAY_INSTANCES + slot] = true;
        }
    }

    // places an asset at pose and coordinates (nullptr: the rest state) in both flow instants: a
    // fresh object has no motion of its own
    template <typename DEVICE, typename SPEC, typename BACKEND>
    hyperlux::OverlayPlacement spawn(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, hyperlux::AssetHandle asset, const hyperlux::Pose& pose = {}, const float* coordinates = nullptr, unsigned int segmentation_class = hyperlux::segmentation::INHERIT){
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
        if(!hyperlux::segmentation::placement_class_valid(segmentation_class)){
            utils::assert_exit(device, false, "spawn: reserved segmentation class");
            return {0, 0, 0};
        }
        auto& state = renderer.overlays[overlay.index];
        const auto& record = renderer.assets[asset.index];

        const TI first_slot = hyperlux::detail::first_fit_slot<SPEC>(renderer, overlay.index, record.num_parts);
        if(first_slot >= SPEC::MAX_OVERLAY_INSTANCES){
            utils::assert_exit(device, false, "spawn: overlay capacity exceeded");
            return {0, 0, 0};
        }

        for(TI part = 0; part < record.num_parts; part++){
            auto& slot = state.slots[first_slot + part];
            slot.object = renderer.asset_parts[record.first_part + part].object;
            slot.active = true;
            slot.segmentation_class = segmentation_class;
        }
        state.dirty = true;
        const hyperlux::OverlayPlacement placement{(size_t)first_slot, (size_t)record.num_parts, (size_t)record.first_part};
        set_pose(device, renderer, overlay, placement, pose, coordinates);
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            for(size_t part = 0; part < placement.num_parts; part++){
                const size_t row = overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part;
                hyperlux::detail::stage_previous(renderer, row, renderer.transforms_staging.data() + row * 12);
            }
        }
        return placement;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void despawn(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement){
        static_assert(SPEC::ENABLE_OVERLAYS, "despawn requires an overlay-enabled renderer specification");
        utils::assert_exit(device, hyperlux::detail::placement_spawned(renderer, overlay, placement), "overlay verb: placement out of range or not spawned");
        auto& state = renderer.overlays[overlay.index];
        for(size_t part = 0; part < placement.num_parts; part++){
            state.slots[placement.first_slot + part].active = false;
        }
        state.dirty = true;
    }

    // swaps the asset occupying a slot run in place: the same slots and ids, the new asset's
    // objects and joints for the next set_pose (nothing changes when the run already holds it);
    // the asset must have the run's part count
    template <typename DEVICE, typename SPEC, typename BACKEND>
    hyperlux::OverlayPlacement replace(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, const hyperlux::OverlayPlacement& placement, hyperlux::AssetHandle asset){
        static_assert(SPEC::ENABLE_OVERLAYS, "replace requires an overlay-enabled renderer specification");
        utils::assert_exit(device, hyperlux::detail::placement_spawned(renderer, overlay, placement), "overlay verb: placement out of range or not spawned");
        if(asset.index >= renderer.assets.size() || renderer.assets[asset.index].num_parts != placement.num_parts){
            utils::assert_exit(device, false, "replace: the asset must exist and have the placement's part count");
            return placement;
        }
        const auto& record = renderer.assets[asset.index];
        auto& state = renderer.overlays[overlay.index];
        if(state.slots[placement.first_slot].object != renderer.asset_parts[record.first_part].object){
            for(size_t part = 0; part < placement.num_parts; part++){
                state.slots[placement.first_slot + part].object = renderer.asset_parts[record.first_part + part].object;
            }
            state.dirty = true;
        }
        return {placement.first_slot, placement.num_parts, (size_t)record.first_part};
    }

    // the overlay's state as the renderer holds it: the slots from the host state, the rows from
    // the resident tensor (what the last update published, or what a device producer wrote since).
    // A snapshot is meaningful only between renderers built from the same scene and pool, whose
    // global object ids agree, at equal capacity
    template <typename DEVICE, typename SPEC, typename BACKEND, typename SNAPSHOT_SPEC>
    void copy(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay, hyperlux::OverlaySnapshot<SNAPSHOT_SPEC>& snapshot){
        static_assert(SPEC::ENABLE_OVERLAYS, "copy of an overlay requires an overlay-enabled renderer specification");
        static_assert(SNAPSHOT_SPEC::MAX_OVERLAY_INSTANCES == SPEC::MAX_OVERLAY_INSTANCES, "copy: the snapshot needs the renderer's overlay capacity");
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "copy: overlay index out of range");
        const auto& state = renderer.overlays[overlay.index];
        for(size_t slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
            snapshot.slots[slot] = {state.slots[slot].active ? 1u : 0u, (unsigned int)state.slots[slot].object, state.slots[slot].segmentation_class};
        }
        std::memcpy(snapshot.rows, data(renderer.transforms) + overlay.index * SPEC::MAX_OVERLAY_INSTANCES * 12, SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
    }

    // restores an overlay from a snapshot of a renderer built from the same scene and pool: its
    // slots into the host state (the next update rebuilds the structure) and its rows through the
    // host mirror, pending for that update; under dynamic motion blur every sample takes the rows,
    // a snapshot being a sharp instant
    template <typename DEVICE, typename SPEC, typename BACKEND, typename SNAPSHOT_SPEC>
    void copy(DEVICE& device, const hyperlux::OverlaySnapshot<SNAPSHOT_SPEC>& snapshot, HyperLux<SPEC, BACKEND>& renderer, hyperlux::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "copy of an overlay requires an overlay-enabled renderer specification");
        static_assert(SNAPSHOT_SPEC::MAX_OVERLAY_INSTANCES == SPEC::MAX_OVERLAY_INSTANCES, "copy: the snapshot needs the renderer's overlay capacity");
        using TI = typename SPEC::TI;
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "copy: overlay index out of range");
        auto& state = renderer.overlays[overlay.index];
        const size_t base = overlay.index * SPEC::MAX_OVERLAY_INSTANCES;
        for(size_t slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
            state.slots[slot].active = snapshot.slots[slot].active != 0;
            state.slots[slot].object = (TI)snapshot.slots[slot].object;
            state.slots[slot].segmentation_class = snapshot.slots[slot].segmentation_class;
            renderer.transforms_pending[base + slot] = true;
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                renderer.transforms_motion_pending[base + slot] = true;
            }
        }
        state.dirty = true;
        std::memcpy(renderer.transforms_staging.data() + base * 12, snapshot.rows, SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                std::memcpy(renderer.transforms_motion_staging.data() + (sample * SLOTS + base) * 12, snapshot.rows, SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
            }
        }
    }

    // camera input tensors, backend-native residency like transforms: device memory on OptiX,
    // host on generic, shared/mapped on Metal/Vulkan. Producers write them via rlt::copy or
    // kernels; the launch verbs consume them directly. Under motion blur the pair is
    // cameras_open (shutter open) and cameras_close (shutter close, aliasing cameras) — both
    // must be written each step (identical values for a blur-free frame). Under flow,
    // cameras_previous holds the previous instant (snapshotted from cameras by advance_flow).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_open(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_open requires a motion-blur renderer specification");
        return renderer.cameras_open;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_close(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_CAMERA_PAIR, "cameras_close requires a motion-blur renderer specification");
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_previous(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_FLOW, "cameras_previous requires a flow-capable renderer specification");
        return renderer.cameras_previous;
    }

    // output tensors, same backend-native residency as the inputs: consumers on the device read
    // them in place (zero-copy); host readers stage through a memory-domain copy at readback
    // boundaries — copy(renderer.device, device, ...) — which orders itself after the
    // renderer's in-flight work
    template <typename DEVICE, typename SPEC, typename BACKEND, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT auto& output(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, OUTPUT descriptor){
        return output(device, renderer.outputs, descriptor);
    }
    template <typename DEVICE, typename SPEC, typename BACKEND, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT const auto& output(DEVICE& device, const HyperLux<SPEC, BACKEND>& renderer, OUTPUT descriptor){
        return output(device, renderer.outputs, descriptor);
    }


    // decodes a rendered segmentation id back to the object it references. Single owner of the
    // global id layout: scene instances occupy [0, S); overlay o's slot s sits at S + o*CAP + s;
    // object indices count scene objects first, then each pool assembly's objects in
    // registration order. Returns nullptr for the miss sentinel and out-of-range ids.
    // This layout is cross-backend API surface: the generic flat instance array, Metal's user-ID
    // descriptors, and OptiX's user instance ids all realize it identically, and segmentation
    // consumers depend on that equivalence — treat any change to it as breaking.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    const hyperlux::Object* segmentation_object(DEVICE& device, const hyperlux::Scene& scene, const hyperlux::AssetPool& pool, const HyperLux<SPEC, BACKEND>& renderer, uint32_t id){
        static_assert(SPEC::HAS_INSTANCE_SEGMENTATION, "segmentation_object requires an instance segmentation output");
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
        hyperlux::detail::compose_transforms(a, b, out);
    }


    namespace hyperlux::detail{
        // probe directions are camera-relative FLU (x = principal axis, y = image left, z = image
        // up): every backend rotates them per camera and replaces probe 0 by the principal axis
        template <typename SPEC, typename DEVICE>
        std::vector<float> generate_probe_direction_vectors(DEVICE& device){
            std::vector<float> dirs;
            dirs.reserve(SPEC::NUM_PROBES * 3);

            // double with the spiral angle reduced modulo one turn: as a float angle of up to ~245 rad the
            // directions moved by one ulp (1.5e-5 rad) between fast-math and strict toolchains
            const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;

            for(int i = 0; i < (int)SPEC::NUM_PROBES; i++){
                const double theta = 2.0 * math::PI<double> * std::fmod(i / golden_ratio, 1.0);
                const double cos_inc = 1.0 - 2.0 * (i + 0.5) / SPEC::NUM_PROBES;
                const double sin_inc = std::sqrt(1.0 - cos_inc * cos_inc);
                dirs.push_back((float)(sin_inc * std::cos(theta)));
                dirs.push_back((float)(sin_inc * std::sin(theta)));
                dirs.push_back((float)cos_inc);
            }

            log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::LOG_PREFIX, "Generated ", SPEC::NUM_PROBES, " probe directions per camera");
            return dirs;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
