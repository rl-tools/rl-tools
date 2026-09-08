#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CPU_H

#include "world.h"
#include "rig/operations_cpu.h"
#include "pose.h"

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

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
    }

    // deduplicated (by path) registration of an entity asset into the shared library's pool;
    // must happen before any hot slot is built (the wrapper init overloads run pre-forward, so
    // the chain guarantees it)
    template <typename DEVICE, typename SHARED_CONTEXT, typename SHADING, bool HAS_RGB>
    rendering::raytracing::AssetHandle register_pool_asset(DEVICE& device, SHARED_CONTEXT& shared, const std::string& path) {
        for (size_t asset_i = 0; asset_i < shared.pool_asset_paths.size(); asset_i++) {
            if (shared.pool_asset_paths[asset_i] == path) {
                return shared.pool_asset_handles[asset_i];
            }
        }
        rendering::raytracing::ObjectAssembly assembly;
        bool loaded = load<SHADING, HAS_RGB>(device, assembly, path);
        utils::assert_exit(device, loaded, "hyperdrone::register_pool_asset: failed to load assembly");
        auto handle = add(device, shared.library.pool, assembly);
        shared.pool_asset_paths.push_back(path);
        shared.pool_asset_handles.push_back(handle);
        return handle;
    }

    // the shared AssetLibrary and SceneSet are owned by the caller (typically the
    // MultiEnvironment); each World builds one hot slot per scene of its partition, each
    // initialized exactly once — rotation only repoints the renderer view
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
        utils::assert_exit(device, num_scenes > 0, "hyperdrone::World: empty scene partition");
        utils::assert_exit(device, first_scene + num_scenes <= (TI)corpus.references.size(), "hyperdrone::World: scene partition out of range");
        if constexpr (SPEC::SELF_VISIBLE) {
            utils::assert_exit(device, !world.drone_asset_path.empty(), "hyperdrone::World: drone_asset_path must be set before init when SELF_VISIBLE");
            auto drone_asset = register_pool_asset<decltype(render_device), typename WORLD::SharedContext, typename SPEC::SHADING, SPEC::OUTPUT_RGB>(render_device, shared, world.drone_asset_path);
            rendering::raytracing::ObjectAssembly assembly;
            bool loaded = load<typename SPEC::SHADING, SPEC::OUTPUT_RGB>(render_device, assembly, world.drone_asset_path);
            utils::assert_exit(device, loaded, "hyperdrone::World: failed to load the drone assembly");
            bool rig_ok = init(render_device, world.drone_rig, assembly);
            utils::assert_exit(device, rig_ok, "hyperdrone::World: drone asset does not follow the body/prop_* convention");
            world.drone_entity_kind_index = (TI)world.entity_kinds.size();
            for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                world.entity_kinds.push_back({drone_asset, (TI)assembly.parts.size(), 0});
            }
        }
        world.slots.resize(num_scenes);
        for (TI slot_i = 0; slot_i < num_scenes; slot_i++) {
            auto& slot = world.slots[slot_i];
            slot.corpus_index = first_scene + slot_i;
            malloc(render_device, slot.renderer, shared.library);
            rendering::Bundle<T> bundle;
            const bool scene_loaded = load<typename SPEC::SHADING, SPEC::OUTPUT_RGB>(render_device, dataset, corpus, slot.corpus_index, bundle);
            utils::assert_exit(device, scene_loaded, "hyperdrone::World: failed to load scene");
            slot.metadata = bundle.metadata;
            const TI scene_id = insert(render_device, shared.library, bundle);
            init(render_device, slot.renderer, shared.library, scene_id);
            generate_probe_directions(render_device, slot.renderer);
            rendering::datasets::annotations::FreeSpaceParameters<T, TI> free_space_parameters{};
            rendering::datasets::annotations::annotate(render_device, slot.annotations, slot.metadata, slot.renderer, free_space_parameters, shared.annotation_cache);
            utils::assert_exit(device, slot.annotations.num_positions > 0, "hyperdrone::World: scene has no valid indoor positions");
            if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
                // pinned deterministic spawn order (instance-major, then registration order) so
                // segmentation ids are stable across runs and backends
                const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
                slot.entity_placements.clear();
                for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                    for (TI view_i = 0; view_i < WORLD::N_VIEWS; view_i++) {
                        attach(render_device, slot.renderer, instance_i * WORLD::N_VIEWS + view_i, rendering::raytracing::OverlayIndex{instance_i});
                    }
                    for (auto& kind : world.entity_kinds) {
                        slot.entity_placements.push_back(spawn(render_device, slot.renderer, rendering::raytracing::OverlayIndex{instance_i}, kind.asset, identity));
                    }
                }
            }
        }
        world.active_slot = 0;
        world.renderer = world.slots[0].renderer;
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
        if constexpr (SPEC::SELF_VISIBLE) {
            malloc(device, world.drone_pose_staging);
        }
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
            if constexpr (SPEC::SELF_VISIBLE) {
                free(device, world.drone_pose_staging);
            }
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
        free(device, world.dynamics);
    }

    // deterministic round-robin over this World's partition; the caller must force a reset of
    // all instances afterwards (everything downstream flows through the reset path)
    template <typename DEVICE, typename SPEC>
    void rotate_scene(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world) {
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        auto& render_device = get_rendering_device(device);
        world.active_slot = (world.active_slot + 1) % world.slots.size();
        world.renderer = world.slots[world.active_slot].renderer;
        Tensor<typename WORLD::ACTIVE_ANNOTATIONS_SPEC> annotations_alias;
        annotations_alias._data = &world.slots[world.active_slot].annotations;
        copy(render_device, device, annotations_alias, world.active_annotations);
        request_render(device, world);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, rl::environments::hyperdrone::World<SPEC>& world, typename rl::environments::hyperdrone::World<SPEC>::Parameters& parameters) {
        parameters = world.parameters;
        initial_parameters(device, world.dynamics, parameters.dynamics);
    }

    namespace rl::environments::hyperdrone {
        // device-capable single-instance sampling shared by the CPU loop and the CUDA kernels
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_initial_parameters(DEVICE& device, typename World<SPEC>::DYNAMICS_ENV& dynamics, const Parameters<SPEC>& defaults, Parameters<SPEC>& parameters, RNG& rng) {
            using T = typename SPEC::T;
            parameters = defaults;
            sample_initial_parameters(device, dynamics, parameters.dynamics, rng);
            parameters.fov = defaults.fov + SPEC::CAMERA_FOV_RANDOMIZATION_RANGE * random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
            CameraRandomization<T> randomization;
            for (unsigned axis_i = 0; axis_i < 3; axis_i++) {
                randomization.offset_body_range[axis_i] = SPEC::CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE;
                randomization.rotation_body_range[axis_i] = SPEC::CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE;
            }
            randomize_camera_mount(device, randomization, parameters.camera_mount, rng);
            parameters.brightness_scale = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng) * (T)2 - (T)1) * SPEC::BRIGHTNESS_RANDOMIZATION_RANGE;
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                parameters.shutter_fraction = random::uniform_real_distribution(device.random, SPEC::SHUTTER_FRACTION_MIN, SPEC::SHUTTER_FRACTION_MAX, rng);
            }
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
        // scene-under-drone: the dynamics state stays near the origin; the sampled indoor
        // position and yaw place the scene via the parameters
        template <typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_initial_state(DEVICE& device, typename World<SPEC>::DYNAMICS_ENV& dynamics, const typename World<SPEC>::ANNOTATIONS& annotations, Parameters<SPEC>& parameters, typename World<SPEC>::State& state, RNG& rng) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            for (TI agent_i = 0; agent_i < SPEC::N_AGENTS; agent_i++) {
                sample_initial_state(device, dynamics, parameters.dynamics, _agent_state<SPEC>(state, agent_i), rng);
            }
            auto indoor_pos = rendering::datasets::annotations::sample_free_position(device, annotations, rng);
            parameters.scene_translation[0] = indoor_pos.position[0];
            parameters.scene_translation[1] = indoor_pos.position[1];
            parameters.scene_translation[2] = indoor_pos.position[2];
            T scene_yaw = random::uniform_real_distribution(device.random, (T)0, (T)2 * math::PI<T>, rng);
            parameters.scene_yaw_cos = math::cos(device.math, scene_yaw);
            parameters.scene_yaw_sin = math::sin(device.math, scene_yaw);
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
        rl::environments::hyperdrone::_sample_initial_parameters<DEVICE, SPEC, RNG>(device, world.dynamics, world.parameters, parameters, rng);
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
            set(device, world.camera_staging_close, rl::environments::hyperdrone::make_camera<DEVICE, T>(device, instance_parameters.camera_mount, instance_parameters.fov, agent_state.orientation, agent_state.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin), camera_i);
        }
        copy(device, world.renderer.device, world.camera_staging_close, cameras(device, world.renderer));
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            copy(device, device, world.prev_cameras, world.camera_staging_previous);
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const TI instance_i = camera_i / WORLD::N_VIEWS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                bool reset = world.history_step == 0 || get(device, reset_mask, instance_i);
                const auto camera_close = get(device, world.camera_staging_close, camera_i);
                set(device, world.camera_staging_open, reset ? camera_close : rl::environments::hyperdrone::interpolate_camera(camera_close, get(device, world.camera_staging_previous, camera_i), instance_parameters.shutter_fraction), camera_i);
            }
            copy(device, world.renderer.device, world.camera_staging_open, cameras_open(device, world.renderer));
            copy(device, device, world.camera_staging_close, world.prev_cameras);
        }
        if constexpr (SPEC::SELF_VISIBLE) {
            auto& slot = world.slots[world.active_slot];
            const TI kinds = (TI)world.entity_kinds.size();
            for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                    const auto& agent_state = rl::environments::hyperdrone::_agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                    float body[12];
                    rl::environments::hyperdrone::rig::make_body_transform(device, agent_state.orientation, agent_state.position, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, body);
                    const auto& placement = slot.entity_placements[instance_i * kinds + world.drone_entity_kind_index + agent_i];
                    T phase_open[4] = {0, 0, 0, 0};
                    T phase_close[4] = {0, 0, 0, 0};
                    if constexpr (rl::environments::hyperdrone::world::HasRotorPhase<typename WORLD::DYNAMICS_ENV::State>::VALUE) {
                        for (TI rotor_i = 0; rotor_i < 4; rotor_i++) {
                            phase_close[rotor_i] = agent_state.rotor_phase[rotor_i];
                            phase_open[rotor_i] = phase_close[rotor_i];
                        }
                        if constexpr (WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                            const T dt = instance_parameters.dynamics.integration.dt;
                            for (TI rotor_i = 0; rotor_i < 4; rotor_i++) {
                                phase_open[rotor_i] -= agent_state.rpm[rotor_i] * (T)2 * math::PI<T> / (T)60 * dt * instance_parameters.shutter_fraction;
                            }
                        }
                    }
                    // the body rides its agent's camera (FPV mount): open == close, motion blur
                    // comes from the props alone
                    if constexpr (WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                        set_transform_pair(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, world.drone_rig, body, body, phase_open, phase_close);
                    } else {
                        set_transform(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, body);
                        for (TI prop_i = 0; prop_i < world.drone_rig.num_props; prop_i++) {
                            float spin[12];
                            rl::environments::hyperdrone::rig::prop_spin_transform(device, world.drone_rig.prop_pivots[prop_i][0], world.drone_rig.prop_pivots[prop_i][1], world.drone_rig.prop_directions[prop_i] * phase_close[prop_i], spin);
                            set_transform(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, world.drone_rig.prop_parts[prop_i], spin);
                        }
                    }
                }
            }
        }
        if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
            // wrapper render overloads wrote their entity poses pre-forward; publish them
            update(device, world.renderer);
        }
        render(device, world.renderer);

        const TI history_slot = world.history_step % SPEC::HISTORY_LENGTH;
        {
            constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
            std::vector<float> observation_staging(NUM_CAMERAS * CAM_PIXELS * 3);
            std::vector<float> frame_staging(NUM_CAMERAS * WORLD::FRAME_DIM);
            {
                Tensor<tensor::Specification<float, TI, typename decltype(world.renderer.observation)::SPEC::SHAPE>> observation_alias;
                observation_alias._data = observation_staging.data();
                copy(world.renderer.device, device, world.renderer.observation, observation_alias);
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
            init(device, env.environments[environment_i], env.shared, dataset, corpus, first, last - first, environment_i);
        }
    }
    template <typename DEVICE, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS>
    void free(DEVICE& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env) {
        using TI = typename MEMBER::TI;
        auto& render_device = get_rendering_device(device);
        for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
            free(device, env.environments[environment_i]);
        }
        free(render_device, env.shared.library);
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
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
