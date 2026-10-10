#pragma once
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <cstddef>
#include <string>

namespace hyperdrone_env_user {
    namespace orbiter {
        using T = float;
        using TI = std::size_t;

        struct BaseSpecification: rl_tools::rl::environments::hyperdrone::presets::X500FPV<T, TI> {
            static constexpr TI INSTANCES_PER_ENVIRONMENT = 4;
            static constexpr TI CAM_WIDTH = 64;
            static constexpr TI CAM_HEIGHT = 64;
            using SHADING = rl_tools::rendering::raytracing::Low;
            static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 16;  // drone rig parts + the orbiter
            static constexpr bool ENABLE_MOTION_BLUR = true;
            static constexpr TI MOTION_BLUR_SAMPLES = 8;
        };

        template <typename T_NEXT_WORLD>
        struct Specification {
            using NEXT_WORLD = T_NEXT_WORLD;
            using T = typename NEXT_WORLD::T;
            using TI = typename NEXT_WORLD::TI;
            static constexpr T ORBIT_RADIUS = 1.5;
            static constexpr T ORBIT_RATE = 1.0;  // revolutions per second
            static constexpr T REWARD_DISTANCE_SCALE = 1.0;
            static constexpr T COLLISION_RADIUS = 0.15;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
        struct ComponentSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename T_SPEC>
        struct ParametersOrbiter: T_SPEC::NEXT_COMPONENT {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            T orbit_center[3] = {0, 0, 0};  // scene frame, set to the drone's spawn position at reset
        };
        template <typename T_SPEC>
        struct StateOrbiter: T_SPEC::NEXT_COMPONENT {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool REQUIRES_INTEGRATION = false;
            static constexpr typename SPEC::TI DIM = 1 + NEXT_COMPONENT::DIM;
            T orbit_phase = 0;
        };

        template <typename T_TASK_SPEC>
        struct World: T_TASK_SPEC::NEXT_WORLD {
            using TASK_SPEC = T_TASK_SPEC;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using T = typename NEXT_WORLD::T;
            using TI = typename NEXT_WORLD::TI;
            static_assert(NEXT_WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0, "the orbiter task needs entity slots: set MAX_ENTITY_SLOTS_PER_INSTANCE in the base Specification");
            using Parameters = ParametersOrbiter<ComponentSpecification<T, TI, typename NEXT_WORLD::Parameters>>;
            using State = StateOrbiter<ComponentSpecification<T, TI, typename NEXT_WORLD::State>>;
            std::string orbiter_asset_path;
            TI entity_kind_index = 0;
        };

        template <typename DEVICE, typename TASK_SPEC>
        void drone_scene_position(DEVICE&, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename TASK_SPEC::T out[3]){
            out[0] = parameters.scene_yaw_cos * state.position[0] - parameters.scene_yaw_sin * state.position[1] + parameters.scene_translation[0];
            out[1] = parameters.scene_yaw_sin * state.position[0] + parameters.scene_yaw_cos * state.position[1] + parameters.scene_translation[1];
            out[2] = state.position[2] + parameters.scene_translation[2];
        }
        template <typename DEVICE, typename TASK_SPEC>
        void orbiter_position(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, typename TASK_SPEC::T phase, typename TASK_SPEC::T out[3]){
            using T = typename TASK_SPEC::T;
            out[0] = parameters.orbit_center[0] + TASK_SPEC::ORBIT_RADIUS * rl_tools::math::cos(device.math, phase);
            out[1] = parameters.orbit_center[1] + TASK_SPEC::ORBIT_RADIUS * rl_tools::math::sin(device.math, phase);
            out[2] = parameters.orbit_center[2];
        }
        template <typename DEVICE, typename TASK_SPEC>
        void orbiter_pose(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, typename TASK_SPEC::T phase, float out[12]){
            using T = typename TASK_SPEC::T;
            T position[3];
            orbiter_position<DEVICE, TASK_SPEC>(device, parameters, phase, position);
            for(unsigned row = 0; row < 3; row++){
                for(unsigned column = 0; column < 3; column++){
                    out[row * 4 + column] = row == column ? 1.0f : 0.0f;
                }
                out[row * 4 + 3] = (float)position[row];
            }
        }
        template <typename DEVICE, typename TASK_SPEC>
        typename TASK_SPEC::T orbiter_distance(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename TASK_SPEC::T phase){
            using T = typename TASK_SPEC::T;
            T drone[3], object[3];
            drone_scene_position<DEVICE, TASK_SPEC>(device, parameters, state, drone);
            orbiter_position<DEVICE, TASK_SPEC>(device, parameters, phase, object);
            T sum = 0;
            for(unsigned dim = 0; dim < 3; dim++){
                T difference = drone[dim] - object[dim];
                sum += difference * difference;
            }
            return rl_tools::math::sqrt(device.math, sum);
        }
    }
    using WORLD = orbiter::World<orbiter::Specification<rl_tools::rl::environments::hyperdrone::World<orbiter::BaseSpecification>>>;
}

namespace rl_tools {
    template <typename DEVICE, typename TASK_SPEC>
    bool set_asset(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, const std::string& name, const std::string& path){
        if(name == "orbiter"){
            world.orbiter_asset_path = path;
            return true;
        }
        return set_asset(device, static_cast<typename TASK_SPEC::NEXT_WORLD&>(world), name, path);
    }
    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index){
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        utils::assert_exit(device, !world.orbiter_asset_path.empty(), "orbiter: the object asset must be set before init (pass assets={'orbiter': ...})");
        world.entity_kind_index = register_entity_kind(device, world, shared, world.orbiter_asset_path);
        init(device, static_cast<NEXT_WORLD&>(world), shared, dataset, corpus, first_scene, num_scenes, member_index);
    }

    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, typename hyperdrone_env_user::orbiter::World<TASK_SPEC>::Parameters& parameters, typename hyperdrone_env_user::orbiter::World<TASK_SPEC>::State& state, RNG& rng){
        using T = typename TASK_SPEC::T;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        sample_initial_state(device, static_cast<NEXT_WORLD&>(world), static_cast<typename NEXT_WORLD::Parameters&>(parameters), static_cast<typename NEXT_WORLD::State&>(state), rng);
        // the orbit is centered on the freshly sampled spawn and its phase aligned with the
        // sampled facing, so the object starts in front of the camera (a wall or the sampled
        // attitude can still hide it) and crosses the view once per revolution
        hyperdrone_env_user::orbiter::drone_scene_position<DEVICE, TASK_SPEC>(device, parameters, state, parameters.orbit_center);
        const T qw = state.orientation[0], qx = state.orientation[1], qy = state.orientation[2], qz = state.orientation[3];
        const T forward_x = 1 - 2 * (qy * qy + qz * qz);
        const T forward_y = 2 * (qx * qy + qw * qz);
        const T scene_forward_x = parameters.scene_yaw_cos * forward_x - parameters.scene_yaw_sin * forward_y;
        const T scene_forward_y = parameters.scene_yaw_sin * forward_x + parameters.scene_yaw_cos * forward_y;
        state.orbit_phase = math::atan2(device.math, scene_forward_y, scene_forward_x);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using TI = typename TASK_SPEC::TI;
        using WORLD = hyperdrone_env_user::orbiter::World<TASK_SPEC>;
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename WORLD::State>);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                sample_initial_state(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
        request_render(device, world, reset_mask);
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC>
    void prepare_entity_poses(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states){
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = hyperdrone_env_user::orbiter::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        const auto& kind = world.entity_kinds[world.entity_kind_index];
        const auto placement = rl::environments::hyperdrone::_entity_placement(world, kind);
        const float* no_coordinates = nullptr;  // a rigid entity: its rig has no coordinates
        auto& renderer = get_renderer(device, world);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            float pose[12];
            hyperdrone_env_user::orbiter::orbiter_pose<DEVICE, TASK_SPEC>(device, instance_parameters, state.orbit_phase, pose);
            if constexpr (NEXT_WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                constexpr T TWO_PI = (T)2 * math::PI<T>;
                const T dt = instance_parameters.agents[0].dynamics.integration.dt;
                float pose_open[12];
                T phase_open = state.orbit_phase - TWO_PI * TASK_SPEC::ORBIT_RATE * dt * instance_parameters.shutter_fraction;
                hyperdrone_env_user::orbiter::orbiter_pose<DEVICE, TASK_SPEC>(device, instance_parameters, phase_open, pose_open);
                set_transform_pair(device, renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, kind.rig, pose_open, pose, no_coordinates, no_coordinates);
            } else {
                set_transform(device, renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, kind.rig, pose, no_coordinates);
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = hyperdrone_env_user::orbiter::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr T TWO_PI = (T)2 * math::PI<T>;
        if constexpr (NEXT_WORLD::SPEC::HISTORY_LENGTH > 1 || NEXT_WORLD::SPEC::OUTPUT_FLOW){
            if(world.render_pending){
                prepare_entity_poses(device, world, parameters, states);
            }
        }
        step(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const T dt = instance_parameters.agents[0].dynamics.integration.dt;
            T phase = get_ref(device, states, instance_i).orbit_phase + TWO_PI * TASK_SPEC::ORBIT_RATE * dt;
            phase = phase - math::floor(device.math, phase / TWO_PI) * TWO_PI;
            get_ref(device, next_states, instance_i).orbit_phase = phase;
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask){
        if(!world.render_pending){
            return;
        }
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        request_render(device, world, reset_mask);
        prepare_entity_poses(device, world, parameters, states);
        render(device, static_cast<NEXT_WORLD&>(world), parameters, states, world.render_reset);
    }
    // a frame requested by reset or step and observed without an explicit render goes through
    // the task's render, so the object moves with it
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename TASK_SPEC::NEXT_WORLD::Observation observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, observation_type, observations, rng);
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = hyperdrone_env_user::orbiter::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        reward(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rewards, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& next_state = get_ref(device, next_states, instance_i);
            T distance = hyperdrone_env_user::orbiter::orbiter_distance<DEVICE, TASK_SPEC>(device, instance_parameters, next_state, next_state.orbit_phase);
            set(device, rewards, get(device, rewards, instance_i) - TASK_SPEC::REWARD_DISTANCE_SCALE * distance, instance_i);
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(DEVICE& device, hyperdrone_env_user::orbiter::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = hyperdrone_env_user::orbiter::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        terminated(device, static_cast<NEXT_WORLD&>(world), parameters, states, terminated_flags, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            T distance = hyperdrone_env_user::orbiter::orbiter_distance<DEVICE, TASK_SPEC>(device, instance_parameters, state, state.orbit_phase);
            if (distance < TASK_SPEC::COLLISION_RADIUS) {
                set(device, terminated_flags, true, instance_i);
            }
        }
    }
}

// ADL entry points, mirroring the shipped tasks: member-lifecycle verbs carry no Tensor
// argument, so composite code only finds a task's overloads through the task type's own
// namespace at instantiation
namespace hyperdrone_env_user::orbiter {
    template <typename DEVICE, typename TASK_SPEC>
    bool set_asset(DEVICE& device, World<TASK_SPEC>& world, const std::string& name, const std::string& path){
        return ::rl_tools::set_asset(device, world, name, path);
    }
    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index){
        ::rl_tools::init(device, world, shared, dataset, corpus, first_scene, num_scenes, member_index);
    }
}