#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CPU_H

#include "world.h"
#include "rig/operations_cpu.h"
#include "pose.h"

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    auto& get_renderer(DEVICE&, rl::environments::hyperdrone::World<SPEC>& world) {
        return world.slots[world.active_slot].renderer;
    }

    template <typename DEVICE, typename SPEC>
    const auto& get_renderer(DEVICE&, const rl::environments::hyperdrone::World<SPEC>& world) {
        return world.slots[world.active_slot].renderer;
    }

    namespace rl::environments::hyperdrone {
        // task wrappers extend the instance Parameters/State by derivation, so the World verbs
        // accept any element type derived from the base contract types
        template <typename WORLD, typename PARAMETER_SPEC, typename STATE_SPEC>
        constexpr bool check_world_instance_tensors(){
            static_assert(utils::typing::is_base_of_v<typename WORLD::Parameters, typename PARAMETER_SPEC::T>);
            static_assert(utils::typing::is_base_of_v<typename WORLD::State, typename STATE_SPEC::T>);
            static_assert(get<0>(typename PARAMETER_SPEC::SHAPE{}) == WORLD::INSTANCES);
            static_assert(get<0>(typename STATE_SPEC::SHAPE{}) == WORLD::INSTANCES);
            return true;
        }
    }

    template <typename DEVICE, typename SPEC, typename RESET_SPEC>
    void request_render(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, const Tensor<RESET_SPEC>& reset_mask){
        static_assert(length(typename RESET_SPEC::SHAPE{}) == 1 && get<0>(typename RESET_SPEC::SHAPE{}) == rl::environments::hyperdrone::World<SPEC>::INSTANCES);
        if(static_cast<const void*>(data(reset_mask)) != static_cast<const void*>(data(world.render_reset))){
            auto reset = reset_mask;
            binary_operation(device, tensor::operations::binary::LogicalOr{}, reset, world.render_reset);
        }
        world.render_pending = true;
    }

    template <typename DEVICE, typename SPEC>
    void request_render(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world){
        world.render_pending = true;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world) {
        malloc(device, world.dynamics);
        malloc(device, world.drone_parameters);
        malloc(device, world.drone_bodies);
    }

    // pools an assembly once per path in the shared library; must happen before any hot slot
    // flattens the pool (wrapper init overloads register before forwarding, the World's own
    // drones register before its slots build)
    template <typename DEVICE, typename SPEC>
    rendering::raytracing::AssetHandle register_pool_asset(DEVICE& device, rl::environments::hyperdrone::SharedContext<SPEC>& shared, const std::string& path) {
        auto& render_device = get_rendering_device(device);
        for (size_t asset_i = 0; asset_i < shared.pool_asset_paths.size(); asset_i++) {
            if (shared.pool_asset_paths[asset_i] == path) {
                return shared.pool_asset_handles[asset_i];
            }
        }
        rendering::raytracing::ObjectAssembly assembly;
        bool loaded = load<typename SPEC::SHADING, SPEC::OUTPUT_RGB>(render_device, assembly, path);
        utils::assert_exit(device, loaded, "hyperdrone::register_pool_asset: failed to load the asset");
        auto handle = add(render_device, shared.library.pool, assembly);
        shared.pool_asset_paths.push_back(path);
        shared.pool_asset_handles.push_back(handle);
        return handle;
    }

    // appends a rigid entity kind: its pooled asset and the rig derived from it; kinds take
    // consecutive per-instance slots in registration order, ahead of the drone region
    template <typename DEVICE, typename SPEC>
    typename SPEC::TI register_entity_kind(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::SharedContext& shared, const std::string& path) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        auto& render_device = get_rendering_device(device);
        typename WORLD::EntityKind kind{};
        kind.asset = register_pool_asset(device, shared, path);
        bool rig_ok = init(render_device, kind.rig, shared.library.pool.assemblies[kind.asset.index]);
        utils::assert_exit(device, rig_ok, "hyperdrone::register_entity_kind: the asset does not fit the entity rig");
        for (const auto& registered : world.entity_kinds) {
            kind.slot_offset += registered.rig.num_parts;
        }
        utils::assert_exit(device, kind.slot_offset + kind.rig.num_parts <= SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE, "hyperdrone::register_entity_kind: the entity kinds exceed MAX_ENTITY_SLOTS_PER_INSTANCE");
        world.entity_kinds.push_back(kind);
        return (TI)world.entity_kinds.size() - 1;
    }

    // builds the drone table once from the registered entries; SELF_VISIBLE pools one mesh per
    // distinct assembly content and matches each entry's rig to its nominal rotors
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::environments::hyperdrone::SharedContext<SPEC>& shared) {
        using TI = typename SPEC::TI;
        using DYNAMICS_ENV = typename rl::environments::hyperdrone::Parameters<SPEC>::DYNAMICS_ENV;
        auto& render_device = get_rendering_device(device);
        utils::assert_exit(device, shared.num_drones == 0, "hyperdrone::SharedContext: the drone table is built once");
        utils::assert_exit(device, !shared.drones.empty(), "hyperdrone::SharedContext: register at least one drone");
        utils::assert_exit(device, shared.drones.size() <= SPEC::MAX_DRONES, "hyperdrone::SharedContext: more drones than Specification::MAX_DRONES");
        if constexpr (SPEC::SELF_VISIBLE) {
            std::vector<std::string> content_hashes;
            for (auto& drone : shared.drones) {
                std::string path, content_hash;
                utils::assert_exit(device, !drone.asset.empty(), "hyperdrone::SharedContext: every drone needs an asset when SELF_VISIBLE");
                utils::assert_exit(device, rendering::datasets::resolve_reference(render_device, drone.asset, path), "hyperdrone::SharedContext: failed to resolve the drone asset");
                utils::assert_exit(device, rendering::datasets::content_hash(render_device, path, content_hash), "hyperdrone::SharedContext: failed to hash the drone asset");
                drone.asset = path;
                rl::environments::hyperdrone::DroneBody<SPEC> body;
                while (body.mesh < (TI)content_hashes.size() && content_hashes[body.mesh] != content_hash) {
                    body.mesh++;
                }
                if (body.mesh == (TI)content_hashes.size()) {
                    shared.drone_meshes.push_back(register_pool_asset(device, shared, path));
                    content_hashes.push_back(content_hash);
                }
                const auto& assembly = shared.library.pool.assemblies[shared.drone_meshes[body.mesh].index];
                const auto& rotors = drone.parameters.dynamics.dynamics;
                const bool rig_ok = init(render_device, body.rig, assembly, rotors.rotor_positions, rotors.rotor_torque_directions, (TI)DYNAMICS_ENV::ACTION_DIM);
                utils::assert_exit(device, rig_ok, "hyperdrone::SharedContext: a drone asset needs a body and one prop_* per rotor, each hub nearest a distinct rotor with a nonzero reaction torque");
                utils::assert_exit(device, shared.drone_bodies.empty() || body.rig.num_parts == shared.drone_bodies[0].rig.num_parts, "hyperdrone::SharedContext: all drone assets need the same part count");
                shared.drone_bodies.push_back(body);
            }
            shared.drone_parts = shared.drone_bodies[0].rig.num_parts;
        }
        shared.num_drones = (TI)shared.drones.size();
    }

    // the shared AssetLibrary and SceneSet are owned by the caller (typically the
    // MultiEnvironment); each World builds one hot slot per scene of its partition, each
    // initialized exactly once
    template <typename DEVICE, typename SPEC, typename DATASET>
    void init(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename SPEC::TI first_scene, typename SPEC::TI num_scenes, typename SPEC::TI member_index) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        auto& render_device = get_rendering_device(device);
        init(device, world.dynamics);
        world.member_index = member_index;
        world.episode_counter = 0;
        world.history_step = 0;
        world.flow_scene_version = 0;
        utils::assert_exit(device, num_scenes > 0, "hyperdrone::World: empty scene partition");
        utils::assert_exit(device, first_scene + num_scenes <= (TI)corpus.references.size(), "hyperdrone::World: scene partition out of range");
        if (shared.num_drones == 0) {
            if (shared.drones.empty()) {
                // no registration: the first World's nominal is the single drone (no assembly)
                shared.drones.push_back({"", {world.dynamics.parameters, world.parameters.camera}});
            }
            init(device, shared);
        }
        world.num_drones = shared.num_drones;
        world.dynamics.parameters = shared.drones[0].parameters.dynamics;
        world.parameters.camera = shared.drones[0].parameters.camera;
        {
            std::vector<rl::environments::hyperdrone::DroneParameters<SPEC>> parameters_staging(SPEC::MAX_DRONES);
            std::vector<rl::environments::hyperdrone::DroneBody<SPEC>> bodies_staging(SPEC::MAX_DRONES);
            for (TI drone_i = 0; drone_i < world.num_drones; drone_i++) {
                parameters_staging[drone_i] = shared.drones[drone_i].parameters;
                if constexpr (SPEC::SELF_VISIBLE) {
                    bodies_staging[drone_i] = shared.drone_bodies[drone_i];
                }
            }
            Tensor<typename WORLD::DRONE_PARAMETERS_SPEC> parameters_alias;
            parameters_alias._data = parameters_staging.data();
            copy(render_device, device, parameters_alias, world.drone_parameters);
            Tensor<typename WORLD::DRONE_BODIES_SPEC> bodies_alias;
            bodies_alias._data = bodies_staging.data();
            copy(render_device, device, bodies_alias, world.drone_bodies);
        }
        if constexpr (SPEC::SELF_VISIBLE) {
            world.drone_meshes = shared.drone_meshes;
            world.drone_parts = shared.drone_parts;
            world.drone_slot_offset = 0;
            for (const auto& kind : world.entity_kinds) {
                world.drone_slot_offset += kind.rig.num_parts;
            }
            const TI required_slots = world.drone_slot_offset + WORLD::N_AGENTS * world.drone_parts;
            const std::string slot_message = "hyperdrone::World: MAX_ENTITY_SLOTS_PER_INSTANCE must be at least " + std::to_string(required_slots) + " for the entity kinds and the drones";
            utils::assert_exit(device, required_slots <= SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE, slot_message.c_str());
        }
        world.slots.resize(num_scenes);
        for (TI slot_i = 0; slot_i < num_scenes; slot_i++) {
            auto& slot = world.slots[slot_i];
            slot.corpus_index = first_scene + slot_i;
            malloc(render_device, slot.renderer, shared.library);
            rendering::Bundle<T> bundle;
            using LOAD_MODE = utils::typing::conditional_t<SPEC::OUTPUT_SEGMENTATION, mode::rendering::SceneRoots<>, mode::Default<>>;
            const bool scene_loaded = load<typename SPEC::SHADING, SPEC::OUTPUT_RGB>(render_device, dataset, corpus, slot.corpus_index, bundle, Mode<LOAD_MODE>{});
            utils::assert_exit(device, scene_loaded, "hyperdrone::World: failed to load scene");
            slot.metadata = bundle.metadata;
            const TI scene_id = insert(render_device, shared.library, bundle);
            init(render_device, slot.renderer, shared.library, scene_id);
            generate_probe_directions(render_device, slot.renderer);
            rendering::datasets::annotations::FreeSpaceParameters<T, TI> free_space_parameters{};
            rendering::datasets::annotations::annotate(render_device, slot.annotations, slot.metadata, slot.renderer, free_space_parameters, shared.annotation_cache);
            utils::assert_exit(device, slot.annotations.num_positions > 0, "hyperdrone::World: scene has no valid indoor positions");
            if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
                // pinned deterministic spawn order (instance-major, registration order, then the
                // drone region with the first mesh) so segmentation ids are stable across runs and
                // backends and every row sits where the producers write it; publishing here leaves
                // no pending rows behind, so the per-step producers own their rows from the first
                // frame on
                const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
                for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                    const rendering::raytracing::OverlayIndex overlay{instance_i};
                    for (TI view_i = 0; view_i < WORLD::N_VIEWS; view_i++) {
                        attach(render_device, slot.renderer, instance_i * WORLD::N_VIEWS + view_i, overlay);
                    }
                    for (const auto& kind : world.entity_kinds) {
                        const auto placement = spawn(render_device, slot.renderer, overlay, kind.asset, identity);
                        utils::assert_exit(device, placement.first_slot == kind.slot_offset && placement.num_parts == kind.rig.num_parts, "hyperdrone::World: entity placement deviates from the registration layout");
                    }
                    if constexpr (SPEC::SELF_VISIBLE) {
                        for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                            const auto placement = spawn(render_device, slot.renderer, overlay, world.drone_meshes[0], identity);
                            utils::assert_exit(device, placement.first_slot == world.drone_slot_offset + agent_i * world.drone_parts, "hyperdrone::World: drone placement deviates from the drone region");
                        }
                    }
                }
                update(render_device, slot.renderer);
            }
        }
        world.active_slot = 0;
        world.scene_version = 1;
        malloc(device, world.history);
        malloc(device, world.prev_cameras);
        malloc(device, world.episode_start);
        malloc(device, world.render_reset);
        malloc(device, world.active_annotations);
        malloc(render_device, world.camera_staging_close);
        malloc(render_device, world.camera_staging_previous);
        malloc(render_device, world.camera_staging_open);
        set_all(device, world.episode_start, (TI)0);
        set_all(device, world.render_reset, false);
        world.render_pending = false;
        {
            Tensor<typename WORLD::ACTIVE_ANNOTATIONS_SPEC> annotations_alias;
            annotations_alias._data = &world.slots[0].annotations;
            copy(render_device, device, annotations_alias, world.active_annotations);
        }
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world) {
        auto& render_device = get_rendering_device(device);
        if (!world.slots.empty()) {
            free(device, world.history);
            free(device, world.prev_cameras);
            free(device, world.episode_start);
            free(device, world.render_reset);
            free(device, world.active_annotations);
            free(render_device, world.camera_staging_close);
            free(render_device, world.camera_staging_previous);
            free(render_device, world.camera_staging_open);
        }
        for (auto& slot : world.slots) {
            free(render_device, slot.renderer);
        }
        world.slots.clear();
        free(device, world.drone_parameters);
        free(device, world.drone_bodies);
        free(device, world.dynamics);
    }

    // Scene changes require the caller to reset all instances before observing again.
    template <typename DEVICE, typename SPEC>
    void select_scene(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename SPEC::TI slot_i) {
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        auto& render_device = get_rendering_device(device);
        utils::assert_exit(device, slot_i < world.slots.size(), "hyperdrone::select_scene: slot out of range");
        world.active_slot = slot_i;
        world.scene_version++;
        Tensor<typename WORLD::ACTIVE_ANNOTATIONS_SPEC> annotations_alias;
        annotations_alias._data = &world.slots[world.active_slot].annotations;
        copy(render_device, device, annotations_alias, world.active_annotations);
        request_render(device, world);
    }

    template <typename DEVICE, typename SPEC>
    void rotate_scene(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world) {
        utils::assert_exit(device, !world.slots.empty(), "hyperdrone::rotate_scene: empty scene partition");
        select_scene(device, world, (world.active_slot + 1) % world.slots.size());
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters) {
        parameters = world.parameters;
        initial_parameters(device, world.dynamics, parameters.dynamics);
    }

    namespace rl::environments::hyperdrone {
        // device-capable single-instance sampling shared by the CPU loops and the CUDA kernels.
        // The episode parameters split into the drone entry, the drone's own (dynamics, shutter)
        // and the view (camera, brightness) so task wrappers can source the view from elsewhere
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::TI _sample_drone(DEVICE& device, typename SPEC::TI num_drones, RNG& rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            if (num_drones <= 1) {
                return 0;
            }
            const TI drone_i = (TI)(random::uniform_real_distribution(device.random, (T)0, (T)1, rng) * (T)num_drones);
            return drone_i < num_drones ? drone_i : num_drones - 1;
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_dynamics_parameters(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, const DroneParameters<SPEC>& drone, Parameters<SPEC>& parameters, RNG& rng) {
            // the l2f sampler copies its environment's nominal before randomizing: hand it the entry's
            auto drone_dynamics = dynamics;
            drone_dynamics.parameters = drone.dynamics;
            sample_initial_parameters(device, drone_dynamics, parameters.dynamics, rng);
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_view_parameters(DEVICE& device, const Camera<typename SPEC::T>& camera, Parameters<SPEC>& parameters, RNG& rng) {
            using T = typename SPEC::T;
            parameters.camera = camera;
            parameters.camera.fov = camera.fov + SPEC::CAMERA_FOV_RANDOMIZATION_RANGE * random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
            CameraRandomization<T> randomization;
            for (unsigned axis_i = 0; axis_i < 3; axis_i++) {
                randomization.offset_body_range[axis_i] = SPEC::CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE;
                randomization.rotation_body_range[axis_i] = SPEC::CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE;
            }
            randomize_camera_mount(device, randomization, parameters.camera.mount, rng);
            parameters.brightness_scale = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng) * (T)2 - (T)1) * SPEC::BRIGHTNESS_RANDOMIZATION_RANGE;
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_shutter_fraction(DEVICE& device, Parameters<SPEC>& parameters, RNG& rng) {
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                parameters.shutter_fraction = random::uniform_real_distribution(device.random, SPEC::SHUTTER_FRACTION_MIN, SPEC::SHUTTER_FRACTION_MAX, rng);
            }
        }
        template <typename DEVICE, typename SPEC, typename DRONE_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_initial_parameters(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, const Parameters<SPEC>& defaults, const Tensor<DRONE_SPEC>& drones, typename SPEC::TI num_drones, Parameters<SPEC>& parameters, RNG& rng) {
            parameters = defaults;
            parameters.drone = _sample_drone<DEVICE, SPEC>(device, num_drones, rng);
            const auto& drone = get_ref(device, drones, parameters.drone);
            _sample_dynamics_parameters<DEVICE, SPEC>(device, dynamics, drone, parameters, rng);
            _sample_view_parameters<DEVICE, SPEC>(device, drone.camera, parameters, rng);
            _sample_shutter_fraction<DEVICE, SPEC>(device, parameters, rng);
        }
        // agent view of the (possibly multi-agent) per-instance state
        template <typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT const typename World<SPEC>::DYNAMICS_ENV::State& _agent_state(const typename World<SPEC>::State& state, typename SPEC::TI agent_i) {
            if constexpr (SPEC::N_AGENTS == 1) {
                return state;
            } else {
                return state.agent_states[agent_i];
            }
        }
        template <typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT typename World<SPEC>::DYNAMICS_ENV::State& _agent_state(typename World<SPEC>::State& state, typename SPEC::TI agent_i) {
            if constexpr (SPEC::N_AGENTS == 1) {
                return state;
            } else {
                return state.agent_states[agent_i];
            }
        }
        // the drone rig's coordinates (prop phases, one per rotor) of one agent at shutter open and
        // close: the open phase trails the close phase by the spin during the shutter fraction;
        // zero without a rotor-phase state
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _drone_coordinates(DEVICE& device, const Parameters<SPEC>& parameters, const typename World<SPEC>::DYNAMICS_ENV::State& agent_state, float open[World<SPEC>::DYNAMICS_ENV::ACTION_DIM], float close[World<SPEC>::DYNAMICS_ENV::ACTION_DIM]) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using WORLD = World<SPEC>;
            constexpr TI ROTORS = WORLD::DYNAMICS_ENV::ACTION_DIM;
            for (TI rotor_i = 0; rotor_i < ROTORS; rotor_i++) {
                close[rotor_i] = 0;
                if constexpr (world::HasRotorPhase<typename WORLD::DYNAMICS_ENV::State>::VALUE) {
                    close[rotor_i] = (float)agent_state.rotor_phase[rotor_i];
                }
                open[rotor_i] = close[rotor_i];
                if constexpr (world::HasRotorPhase<typename WORLD::DYNAMICS_ENV::State>::VALUE && WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                    const T dt = parameters.dynamics.integration.dt;
                    open[rotor_i] = (float)((T)agent_state.rotor_phase[rotor_i] - agent_state.rpm[rotor_i] * (T)2 * math::PI<T> / (T)60 * dt * parameters.shutter_fraction);
                }
            }
        }
        template <typename SPEC>
        rendering::raytracing::OverlayPlacement _entity_placement(const World<SPEC>& world, const typename World<SPEC>::EntityKind& kind) {
            return {kind.slot_offset, kind.rig.num_parts, world.slots[world.active_slot].renderer.assets[kind.asset.index].first_part};
        }
        // the host counterpart of the device drone rows: swaps the agent's region to the body of
        // the instance's drone entry (a no-op when it already holds it) and poses it
        template <typename DEVICE, typename SPEC, typename RENDERER>
        void _drone_rows(DEVICE& device, const World<SPEC>& world, RENDERER& renderer, typename SPEC::TI instance_i, typename SPEC::TI agent_i, const DroneBody<SPEC>& body, const float base_open[12], const float base_close[12], const float coordinates_open[], const float coordinates_close[]) {
            const rendering::raytracing::OverlayIndex overlay{instance_i};
            const rendering::raytracing::OverlayPlacement region{world.drone_slot_offset + agent_i * world.drone_parts, world.drone_parts, 0};
            const auto placement = replace(device, renderer, overlay, region, world.drone_meshes[body.mesh]);
            if constexpr (World<SPEC>::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                set_transform_pair(device, renderer, overlay, placement, body.rig, base_open, base_close, coordinates_open, coordinates_close);
            } else {
                set_transform(device, renderer, overlay, placement, body.rig, base_close, coordinates_close);
            }
        }
        // scene-under-drone: the dynamics state stays near the origin; the sampled indoor
        // position and yaw place the scene via the parameters
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_dynamics_state(DEVICE& device, typename World<SPEC>::DYNAMICS_ENV& dynamics, Parameters<SPEC>& parameters, typename World<SPEC>::State& state, RNG& rng) {
            using TI = typename SPEC::TI;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                sample_initial_state(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), rng);
            }
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_scene_placement(DEVICE& device, const typename World<SPEC>::ANNOTATIONS& annotations, Parameters<SPEC>& parameters, RNG& rng) {
            using T = typename SPEC::T;
            auto indoor_pos = rendering::datasets::annotations::sample_free_position(device, annotations, rng);
            parameters.scene_translation[0] = indoor_pos.position[0];
            parameters.scene_translation[1] = indoor_pos.position[1];
            parameters.scene_translation[2] = indoor_pos.position[2];
            T scene_yaw = random::uniform_real_distribution(device.random, (T)0, (T)2 * math::PI<T>, rng);
            parameters.scene_yaw_cos = math::cos(device.math, scene_yaw);
            parameters.scene_yaw_sin = math::sin(device.math, scene_yaw);
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_initial_state(DEVICE& device, typename World<SPEC>::DYNAMICS_ENV& dynamics, const typename World<SPEC>::ANNOTATIONS& annotations, Parameters<SPEC>& parameters, typename World<SPEC>::State& state, RNG& rng) {
            _sample_dynamics_state<DEVICE, SPEC>(device, dynamics, parameters, state, rng);
            _sample_scene_placement<DEVICE, SPEC>(device, annotations, parameters, rng);
        }
        // device-capable per-instance dynamics verbs shared by the CPU loops and the CUDA
        // kernels: agents step independently, the reward is the cooperative mean, termination is
        // any-agent (including pairwise proximity)
        template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T _step(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, Parameters<SPEC>& parameters, const typename World<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename World<SPEC>::State& next_state, RNG& rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using WORLD = World<SPEC>;
            constexpr TI PER_AGENT_ACTION_DIM = WORLD::DYNAMICS_ENV::ACTION_DIM;
            T dt = 0;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, PER_AGENT_ACTION_DIM, false>> agent_action;
                for (TI action_i = 0; action_i < PER_AGENT_ACTION_DIM; action_i++) {
                    set(agent_action, 0, action_i, get(action, 0, agent_i * PER_AGENT_ACTION_DIM + action_i));
                }
                T agent_dt = step(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), agent_action, _agent_state<SPEC>(next_state, agent_i), rng);
                dt = agent_i == 0 ? agent_dt : dt;
            }
            return dt;
        }
        template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T _reward(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, Parameters<SPEC>& parameters, const typename World<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename World<SPEC>::State& next_state, RNG& rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using WORLD = World<SPEC>;
            constexpr TI PER_AGENT_ACTION_DIM = WORLD::DYNAMICS_ENV::ACTION_DIM;
            T sum = 0;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, PER_AGENT_ACTION_DIM, false>> agent_action;
                for (TI action_i = 0; action_i < PER_AGENT_ACTION_DIM; action_i++) {
                    set(agent_action, 0, action_i, get(action, 0, agent_i * PER_AGENT_ACTION_DIM + action_i));
                }
                sum += reward(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), agent_action, _agent_state<SPEC>(next_state, agent_i), rng);
            }
            return sum / (T)SPEC::N_AGENTS;
        }
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT bool _terminated(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, const Parameters<SPEC>& parameters, const typename World<SPEC>::State& state, RNG& rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                if (terminated(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), rng)) {
                    return true;
                }
            }
            if constexpr (SPEC::N_AGENTS > 1) {
                for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                    for (TI other_i = agent_i + 1; other_i < SPEC::N_AGENTS; other_i++) {
                        const auto& a = _agent_state<SPEC>(state, agent_i);
                        const auto& b = _agent_state<SPEC>(state, other_i);
                        T distance_squared = 0;
                        for (TI dim = 0; dim < 3; dim++) {
                            T delta = a.position[dim] - b.position[dim];
                            distance_squared += delta * delta;
                        }
                        if (distance_squared < SPEC::AGENT_COLLISION_DISTANCE * SPEC::AGENT_COLLISION_DISTANCE) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
        template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _observe_dynamics(DEVICE& device, const typename World<SPEC>::DYNAMICS_ENV& dynamics, Parameters<SPEC>& parameters, const typename World<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng) {
            using TI = typename SPEC::TI;
            using WORLD = World<SPEC>;
            constexpr TI PER_AGENT_DIM = WORLD::DYNAMICS_ENV::Observation::DIM;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                auto agent_observation = view(device, observation, matrix::ViewSpec<1, PER_AGENT_DIM>{}, 0, agent_i * PER_AGENT_DIM);
                observe(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), typename WORLD::DYNAMICS_ENV::Observation{}, agent_observation, rng);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, RNG& rng) {
        rl::environments::hyperdrone::_sample_initial_parameters<DEVICE, SPEC>(device, world.dynamics, world.parameters, world.drone_parameters, world.num_drones, parameters, rng);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, typename rl::environments::hyperdrone::World<SPEC>::State& state, RNG& rng) {
        rl::environments::hyperdrone::_sample_initial_state<DEVICE, SPEC, RNG>(device, world.dynamics, world.slots[world.active_slot].annotations, parameters, state, rng);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::World<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::hyperdrone::World<SPEC>::State& next_state, RNG& rng) {
        return rl::environments::hyperdrone::_step<DEVICE, SPEC>(device, world.dynamics, parameters, state, action, next_state, rng);
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::World<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::hyperdrone::World<SPEC>::State& next_state, RNG& rng) {
        return rl::environments::hyperdrone::_reward<DEVICE, SPEC>(device, world.dynamics, parameters, state, action, next_state, rng);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool terminated(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, const typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::World<SPEC>::State& state, RNG& rng) {
        return rl::environments::hyperdrone::_terminated<DEVICE, SPEC>(device, world.dynamics, parameters, state, rng);
    }

    template <typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::World<SPEC>::State& state, const typename rl::environments::hyperdrone::World<SPEC>::ObservationPrivileged& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng) {
        rl::environments::hyperdrone::_observe_dynamics<DEVICE, SPEC>(device, world.dynamics, parameters, state, observation, rng);
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                sample_initial_parameters(device, world, get_ref(device, parameters, instance_i), rng);
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                sample_initial_state(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
        request_render(device, world, reset_mask);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void step(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(get<1>(typename ACTION_SPEC::SHAPE{}) == WORLD::ACTION_DIM);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, WORLD::ACTION_DIM, false>> action_matrix;
            for (TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++) {
                set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
            }
            step(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng);
        }
        request_render(device, world);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void reward(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, WORLD::ACTION_DIM, false>> action_matrix;
            for (TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++) {
                set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
            }
            set(device, rewards, reward(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng), instance_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void terminated(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            set(device, terminated_flags, terminated(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng), instance_i);
        }
    }
    // dynamics-side observations (privileged and reduced state-branch chains); the visual
    // Image observation has its own overload below
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !utils::typing::is_same_v<OBSERVATION_TYPE, typename rl::environments::hyperdrone::World<SPEC>::Observation>, bool>::type = true>
    void observe(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            auto observation_slice = view(device, observations, instance_i);
            auto observation_matrix = matrix_view(device, observation_slice);
            if constexpr (utils::typing::is_same_v<OBSERVATION_TYPE, typename WORLD::ObservationPrivileged>) {
                rl::environments::hyperdrone::_observe_dynamics<DEVICE, SPEC>(device, world.dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), observation_matrix, rng);
            } else {
                static_assert(WORLD::N_AGENTS == 1, "multi-agent Worlds expose only the privileged observation on the dynamics side");
                observe(device, world.dynamics, get_ref(device, parameters, instance_i).dynamics, get_ref(device, states, instance_i), observation_type, observation_matrix, rng);
            }
        }
    }

    // the render verb: pose production -> render launch -> scatter into the frame history.
    // Reset semantics ride the mask: reset instances restart their shutter pair and their
    // episode's history window
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask_input) {
        request_render(device, world, reset_mask_input);
        auto& reset_mask = world.render_reset;
        auto& renderer = get_renderer(device, world);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(SPEC::OUTPUT_RGB, "hyperdrone::World::render requires the RGB observation output");
        constexpr TI INSTANCES = WORLD::INSTANCES;
        constexpr TI NUM_CAMERAS = INSTANCES * WORLD::N_VIEWS;
        const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);

        for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
            const TI instance_i = camera_i / WORLD::N_VIEWS;
            const TI agent_i = camera_i % WORLD::N_VIEWS;
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& agent_state = rl::environments::hyperdrone::_agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
            set(device, world.camera_staging_close, rl::environments::hyperdrone::make_camera<DEVICE, T>(device, instance_parameters.camera, agent_state.orientation, agent_state.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin), camera_i);
        }
        copy(device, renderer.device, world.camera_staging_close, cameras(device, renderer));
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            copy(device, device, world.prev_cameras, world.camera_staging_previous);
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const TI instance_i = camera_i / WORLD::N_VIEWS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                bool reset = world.history_step == 0 || get(device, reset_mask, instance_i);
                const auto camera_close = get(device, world.camera_staging_close, camera_i);
                set(device, world.camera_staging_open, reset ? camera_close : rl::environments::hyperdrone::interpolate_camera(camera_close, get(device, world.camera_staging_previous, camera_i), instance_parameters.shutter_fraction), camera_i);
            }
            copy(device, renderer.device, world.camera_staging_open, cameras_open(device, renderer));
            copy(device, device, world.camera_staging_close, world.prev_cameras);
        }
        if constexpr (SPEC::SELF_VISIBLE) {
            for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                utils::assert_exit(device, instance_parameters.drone < world.num_drones, "hyperdrone::World::render: drone entry out of range");
                const auto& body = get_ref(device, world.drone_bodies, instance_parameters.drone);
                for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                    const auto& agent_state = rl::environments::hyperdrone::_agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                    float base[12];
                    rl::environments::hyperdrone::rig::make_body_transform(device, agent_state.orientation, agent_state.position, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, base);
                    float coordinates_open[WORLD::DYNAMICS_ENV::ACTION_DIM], coordinates_close[WORLD::DYNAMICS_ENV::ACTION_DIM];
                    rl::environments::hyperdrone::_drone_coordinates<DEVICE, SPEC>(device, instance_parameters, agent_state, coordinates_open, coordinates_close);
                    // the body is posed at shutter close; only the props blur
                    rl::environments::hyperdrone::_drone_rows(device, world, renderer, instance_i, agent_i, body, base, base, coordinates_open, coordinates_close);
                }
            }
        }
        if constexpr (SPEC::OUTPUT_FLOW){
            const bool first = world.flow_scene_version != world.scene_version;
            if(!first){
                copy(renderer.device, device, cameras_previous(device, renderer), world.camera_staging_previous);
            }
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                if(first || get(device, reset_mask, instance_i)){
                    for(TI agent_i = 0; agent_i < WORLD::N_VIEWS; agent_i++){
                        const TI camera_i = instance_i * WORLD::N_VIEWS + agent_i;
                        set(device, world.camera_staging_previous, get(device, world.camera_staging_close, camera_i), camera_i);
                    }
                    if constexpr (WORLD::RENDERER_SPEC::HAS_FLOW_OVERLAYS){
                        for(TI slot = 0; slot < SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE; slot++){
                            const TI row = instance_i * SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE + slot;
                            rendering::raytracing::detail::stage_previous(renderer, row, renderer.transforms_staging.data() + row * 12);
                        }
                    }
                }
            }
            copy(device, renderer.device, world.camera_staging_previous, cameras_previous(device, renderer));
        }
        if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
            // wrapper render overloads staged their entity poses pre-forward; publish them
            update(device, renderer);
        }
        render(device, renderer);
        if constexpr (SPEC::OUTPUT_FLOW){
            advance_flow(device, renderer);
            world.flow_scene_version = world.scene_version;
        }

        const TI history_slot = world.history_step % SPEC::HISTORY_LENGTH;
        {
            constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
            std::vector<float> observation_staging(NUM_CAMERAS * CAM_PIXELS * 3);
            std::vector<float> frame_staging(NUM_CAMERAS * WORLD::FRAME_DIM);
            {
                Tensor<tensor::Specification<float, TI, typename decltype(renderer.observation)::SPEC::SHAPE>> observation_alias;
                observation_alias._data = observation_staging.data();
                copy(renderer.device, device, renderer.observation, observation_alias);
            }
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const auto& instance_parameters = get_ref(device, parameters, camera_i / WORLD::N_VIEWS);
                const float scale = (float)instance_parameters.brightness_scale;
                for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                    for (TI channel_i = 0; channel_i < 3; channel_i++) {
                        float value = observation_staging[(camera_i * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
                        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                        frame_staging[camera_i * WORLD::FRAME_DIM + pixel_i * WORLD::IMAGE_CHANNELS + channel_i] = value;
                    }
                }
            }
            auto history_row = view(device, world.history, history_slot);
            Tensor<tensor::Specification<float, TI, tensor::Shape<TI, NUM_CAMERAS, WORLD::FRAME_DIM>>> frame_alias;
            frame_alias._data = frame_staging.data();
            copy(device, device, frame_alias, history_row);
        }
        {
            std::vector<TI> episode_start_staging(INSTANCES);
            Tensor<typename WORLD::EPISODE_START_SPEC> episode_start_alias;
            episode_start_alias._data = episode_start_staging.data();
            copy(device, device, world.episode_start, episode_start_alias);
            for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
                if (get(device, reset_mask, instance_i)) {
                    episode_start_staging[instance_i] = world.history_step;
                }
            }
            copy(device, device, episode_start_alias, world.episode_start);
        }
        world.history_step++;
        set_all(device, world.render_reset, false);
        world.render_pending = false;
    }

    // the per-step visual observation is the latest rendered frame
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::World<SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(rl::environments::hyperdrone::check_world_instance_tensors<WORLD, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::World::observe: no frame available");
        const TI history_slot = (world.history_step - 1) % SPEC::HISTORY_LENGTH;
        std::vector<float> frame_staging(WORLD::INSTANCES * WORLD::N_VIEWS * WORLD::FRAME_DIM);
        auto history_row = view(device, world.history, history_slot);
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, WORLD::INSTANCES * WORLD::N_VIEWS, WORLD::FRAME_DIM>>> frame_alias;
        frame_alias._data = frame_staging.data();
        copy(device, device, history_row, frame_alias);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            for (TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++) {
                set(device, observations, (T)frame_staging[instance_i * WORLD::OBSERVATION_DIM + dim_i], instance_i, dim_i);
            }
        }
    }

    // MultiEnvironment over Worlds: owns the shared AssetLibrary and SceneSet; scenes are
    // partitioned in contiguous blocks across the members
    namespace rl::environments::hyperdrone {
        template <typename MULTI_ENVIRONMENT, typename SPEC>
        constexpr typename MULTI_ENVIRONMENT::TI instances_per_environment(){
            static_assert(get<0>(typename SPEC::SHAPE{}) == MULTI_ENVIRONMENT::INSTANCES, "instance tensors must cover all members' instances contiguously");
            return MULTI_ENVIRONMENT::INSTANCES_PER_ENVIRONMENT;
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS>
    void malloc(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env) {
        using TI = typename MEMBER::TI;
        auto& render_device = get_rendering_device(device);
        malloc(render_device, env.shared.library);
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            malloc(device, env.environments[environment_i]);
        }
    }
    template <typename DEVICE, typename SPEC, typename DATASET>
    void init_segmentation_taxonomy(DEVICE& device, rl::environments::hyperdrone::SharedContext<SPEC>& shared, const DATASET&){
        if constexpr (SPEC::SEMANTIC_SEGMENTATION){
            auto& render_device = get_rendering_device(device);
            std::shared_ptr<const rendering::SegmentationTaxonomy> taxonomy;
            std::string taxonomy_source;
            std::vector<rendering::Object*> objects;
            const std::string expected = DATASET::SEGMENTATION_TAXONOMY;
            auto collect = [&](auto& container, const std::string& source, bool scene){
                auto entries = segmentation_objects(render_device, container);
                const auto candidate = segmentation_taxonomy(render_device, entries);
                if(scene && !expected.empty() && (!candidate || candidate->name != expected)){
                    throw std::invalid_argument("hyperdrone: " + source + " requires segmentation taxonomy '" + expected + "'");
                }
                if(candidate){
                    if(taxonomy && (candidate->name != taxonomy->name || candidate->classes != taxonomy->classes)){
                        throw std::invalid_argument("hyperdrone: incompatible segmentation taxonomies in " + taxonomy_source + " and " + source + "; remap classes before loading");
                    }
                    taxonomy = candidate;
                    taxonomy_source = source;
                }
                objects.insert(objects.end(), entries.begin(), entries.end());
            };
            for(size_t scene_i = 0; scene_i < shared.library.scenes.size(); scene_i++){
                collect(shared.library.scenes[scene_i], "scene " + shared.library.metadata[scene_i].content_hash, true);
            }
            collect(shared.library.pool, "overlay assets", false);
            validate_semantic_segmentation(render_device, objects);
            for(auto* object : objects){
                if(object->segmentation_taxonomy){
                    object->segmentation_taxonomy = taxonomy;
                }
            }
            shared.segmentation_taxonomy = std::move(taxonomy);
        }
    }

    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename DATASET>
    void init(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, const DATASET& dataset) {
        using TI = typename MEMBER::TI;
        auto& render_device = get_rendering_device(device);
        typename DATASET::Corpus corpus;
        enumerate(render_device, dataset, corpus);
        const TI total_scenes = (TI)corpus.references.size();
        utils::assert_exit(device, total_scenes >= NUMBER_OF_ENVIRONMENTS, "hyperdrone: fewer scenes than environments");
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            const TI first = environment_i * total_scenes / NUMBER_OF_ENVIRONMENTS;
            const TI last = (environment_i + 1) * total_scenes / NUMBER_OF_ENVIRONMENTS;
            env.environments[environment_i].rng_offset = environment_i * MEMBER::INSTANCES;  // disjoint per-instance RNG partitions on devices with per-instance engines
            init(device, env.environments[environment_i], env.shared, dataset, corpus, first, last - first, environment_i);
        }
        init_segmentation_taxonomy(device, env.shared, dataset);
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS>
    void free(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env) {
        using TI = typename MEMBER::TI;
        auto& render_device = get_rendering_device(device);
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            free(device, env.environments[environment_i]);
        }
        free(render_device, env.shared.library);
        env.shared.segmentation_taxonomy.reset();
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, PARAMETER_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            sample_initial_parameters(device, env.environments[environment_i], parameters_block, reset_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            sample_initial_state(device, env.environments[environment_i], parameters_block, states_block, reset_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto observations_block = view_range(device, observations, environment_i * M, tensor::ViewSpec<0, M>{});
            observe(device, env.environments[environment_i], parameters_block, states_block, observation_type, observations_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto actions_block = view_range(device, actions, environment_i * M, tensor::ViewSpec<0, M>{});
            auto next_states_block = view_range(device, next_states, environment_i * M, tensor::ViewSpec<0, M>{});
            step(device, env.environments[environment_i], parameters_block, states_block, actions_block, next_states_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto actions_block = view_range(device, actions, environment_i * M, tensor::ViewSpec<0, M>{});
            auto next_states_block = view_range(device, next_states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto rewards_block = view_range(device, rewards, environment_i * M, tensor::ViewSpec<0, M>{});
            reward(device, env.environments[environment_i], parameters_block, states_block, actions_block, next_states_block, rewards_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto terminated_block = view_range(device, terminated_flags, environment_i * M, tensor::ViewSpec<0, M>{});
            terminated(device, env.environments[environment_i], parameters_block, states_block, terminated_block, rng);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask) {
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            render(device, env.environments[environment_i], parameters_block, states_block, reset_block);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS>
    void rotate_scene(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env) {
        using TI = typename MEMBER::TI;
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            rotate_scene(device, env.environments[environment_i]);
        }
    }

    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, const rl::environments::hyperdrone::Camera<typename SPEC::T>& camera){
        using rl::environments::l2f::detail::lossless_to_string;
        const auto& pose = camera.mount.pose;
        std::string json_string = "{\"mount\": {\"pose\": {\"position\": [";
        for (unsigned dim_i = 0; dim_i < 3; dim_i++) {
            json_string += lossless_to_string(pose.position[dim_i]) + (dim_i < 2 ? ", " : "");
        }
        json_string += "], \"orientation\": [";
        for (unsigned dim_i = 0; dim_i < 4; dim_i++) {
            json_string += lossless_to_string(pose.orientation[dim_i]) + (dim_i < 3 ? ", " : "");
        }
        json_string += "]}}, \"fov\": " + lossless_to_string(camera.fov) + "}";
        return json_string;
    }
    template <typename DEVICE, typename SPEC>
    void from_json(DEVICE& device, const rl::environments::hyperdrone::World<SPEC>& world, const nlohmann::json& json_object, rl::environments::hyperdrone::Camera<typename SPEC::T>& camera){
        using T = typename SPEC::T;
        const nlohmann::json& pose = json_object.at("mount").at("pose");
        for (unsigned dim_i = 0; dim_i < 3; dim_i++) {
            camera.mount.pose.position[dim_i] = pose.at("position").at(dim_i).get<T>();
        }
        for (unsigned dim_i = 0; dim_i < 4; dim_i++) {
            camera.mount.pose.orientation[dim_i] = pose.at("orientation").at(dim_i).get<T>();
        }
        camera.fov = json_object.at("fov").get<T>();
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
