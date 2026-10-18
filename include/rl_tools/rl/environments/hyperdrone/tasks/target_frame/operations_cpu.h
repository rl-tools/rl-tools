#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_CPU_H

#include "target_frame.h"
#include "../../operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    namespace rl::environments::hyperdrone::tasks::target_frame {
        // device-capable sampling shared by the CPU loops and the CUDA kernels
        // a setup carries its drone entry: the entry's camera renders the target and the adopting
        // instance flies the entry
        template <typename DEVICE, typename TASK_SPEC, typename DRONE_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_setup(DEVICE& device, const typename World<TASK_SPEC>::NEXT_WORLD::ANNOTATIONS& annotations, const typename World<TASK_SPEC>::NEXT_WORLD::Parameters& defaults, const Tensor<DRONE_SPEC>& drones, typename TASK_SPEC::TI num_drones, typename World<TASK_SPEC>::Parameters& setup, RNG& rng){
            using T = typename TASK_SPEC::T;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            auto& view = static_cast<typename NEXT_WORLD::Parameters&>(setup);
            view = defaults;
            view.agents[0].drone = hyperdrone::_sample_drone<DEVICE, BASE_SPEC>(device, num_drones, rng);
            hyperdrone::_sample_camera_parameters<DEVICE, BASE_SPEC>(device, get_ref(device, drones, view.agents[0].drone), view.agents[0], rng);
            hyperdrone::_sample_brightness<DEVICE, BASE_SPEC>(device, view, rng);
            hyperdrone::_sample_scene_placement<DEVICE, BASE_SPEC>(device, annotations, view, rng);
            if constexpr (TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE > (T)0){
                setup.target_roll = random::uniform_real_distribution(device.random, -TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng);
                setup.target_pitch = random::uniform_real_distribution(device.random, -TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng);
            } else {
                setup.target_roll = 0;
                setup.target_pitch = 0;
            }
            if constexpr (TASK_SPEC::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > (T)0){
                setup.brightness_mismatch = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng) * (T)2 - (T)1) * TASK_SPEC::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
            } else {
                setup.brightness_mismatch = 1;
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename DRONE_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _adopt_setup(DEVICE& device, const typename World<TASK_SPEC>::NEXT_WORLD::DYNAMICS_ENV& dynamics, const Tensor<DRONE_SPEC>& drones, const typename World<TASK_SPEC>::Parameters& setup, typename World<TASK_SPEC>::Parameters& parameters, RNG& rng){
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            parameters = setup;
            auto& base = static_cast<typename NEXT_WORLD::Parameters&>(parameters);
            hyperdrone::_sample_dynamics_parameters<DEVICE, BASE_SPEC>(device, dynamics, get_ref(device, drones, setup.agents[0].drone), base.agents[0], rng);
            hyperdrone::_sample_shutter_fraction<DEVICE, BASE_SPEC>(device, base, rng);
        }
        RL_TOOLS_FUNCTION_PLACEMENT inline float _target_pixel(float value, float scale){
            value *= scale;
            return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        }
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT auto _target_camera(DEVICE& device, const typename World<TASK_SPEC>::Parameters& setup, typename TASK_SPEC::TI mount, typename TASK_SPEC::T aspect) {
            using T = typename TASK_SPEC::T;
            constexpr auto TARGET_CAMERA = World<TASK_SPEC>::TARGET_CAMERA;
            if(mount == TARGET_CAMERA) {
                return make_target_camera<DEVICE, T>(device, setup.agents[0].cameras[TARGET_CAMERA], aspect, setup.scene_translation, setup.scene_yaw_cos, setup.scene_yaw_sin, setup.target_roll, setup.target_pitch);
            }
            T orientation[4], position[3];
            make_target_body_pose(device, setup.agents[0].cameras[TARGET_CAMERA], setup.target_roll, setup.target_pitch, orientation, position);
            return make_camera<DEVICE, T>(device, setup.agents[0].cameras[mount], orientation, position, aspect, setup.scene_translation, setup.scene_yaw_cos, setup.scene_yaw_sin);
        }
        template <typename TASK_SPEC>
        bool _bank_due(const World<TASK_SPEC>& world){
            return world.bank_scene_version != world.scene_version || world.history_step - world.bank_step >= TASK_SPEC::TARGET_BANK_REFRESH_STEPS;
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        init(device, static_cast<NEXT_WORLD&>(world), shared, dataset, corpus, first_scene, num_scenes, member_index);
        malloc(device, world.setups);
        malloc(device, world.bank_frames);
        malloc(device, world.setup_index);
        malloc(device, world.bank_cursor);
        malloc(device, world.target_frames);
        world.bank_step = 0;
        world.bank_scene_version = 0;
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if (!world.slots.empty()) {
            free(device, world.setups);
            free(device, world.bank_frames);
            free(device, world.setup_index);
            free(device, world.bank_cursor);
            free(device, world.target_frames);
        }
        free(device, static_cast<NEXT_WORLD&>(world));
    }

    // the bank: one setup per instance from the active scene's free space, rendered through the
    // instance cameras in one pass and scaled into bank_frames
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_target_setups(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        for (TI setup_i = 0; setup_i < WORLD::INSTANCES; setup_i++) {
            rl::environments::hyperdrone::tasks::target_frame::_sample_setup<DEVICE, TASK_SPEC>(device, world.slots[world.active_slot].annotations, world.parameters, world.drone_parameters, world.num_drones, get_ref(device, world.setups, setup_i), rng);
        }
    }
    namespace rl::environments::hyperdrone::tasks::target_frame {
        // renders the target view of every per-instance setup (the bank's, or instances' own
        // parameters) and writes the scaled frames of the masked rows (all for a null mask)
        template <typename DEVICE, typename TASK_SPEC, typename SETUPS_SPEC, typename FRAMES_SPEC>
        void _render_targets(DEVICE& device, World<TASK_SPEC>& world, Tensor<SETUPS_SPEC>& setups, Tensor<FRAMES_SPEC>& frames, const bool* mask) {
            using T = typename TASK_SPEC::T;
            using TI = typename TASK_SPEC::TI;
            using WORLD = World<TASK_SPEC>;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            static_assert(utils::typing::is_same_v<typename SETUPS_SPEC::T, typename WORLD::Parameters>);
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            const T aspect = static_cast<T>(BASE_SPEC::CAM_WIDTH) / static_cast<T>(BASE_SPEC::CAM_HEIGHT);
            auto& renderer = get_renderer(device, world);
            if constexpr (BASE_SPEC::SELF_VISIBLE) {
                // the setup's drone at rest at the target pose, overwritten by the next frame's rows
                const float coordinates[NEXT_WORLD::DYNAMICS_ENV::ACTION_DIM] = {};
                for (TI setup_i = 0; setup_i < WORLD::INSTANCES; setup_i++) {
                    const auto& setup = get_ref(device, setups, setup_i);
                    T orientation[4], position[3];
                    make_target_body_pose(device, setup.agents[0].cameras[WORLD::TARGET_CAMERA], setup.target_roll, setup.target_pitch, orientation, position);
                    float base[12];
                    rig::make_body_transform(device, orientation, position, setup.scene_translation, setup.scene_yaw_cos, setup.scene_yaw_sin, base);
                    const auto& body = get_ref(device, world.drone_bodies, setup.agents[0].drone);
                    const float placement_scale = rotor_scale(device, body.rig, setup.agents[0].dynamics.dynamics.rotor_positions, (TI)NEXT_WORLD::DYNAMICS_ENV::ACTION_DIM);
                    hyperdrone::_drone_rows(device, world, renderer, setup_i, (TI)0, body, base, base, coordinates, coordinates, placement_scale);
                }
                update(device, renderer);
            }
            for (TI setup_i = 0; setup_i < WORLD::INSTANCES; setup_i++) {
                const auto& setup = get_ref(device, setups, setup_i);
                for(TI mount = 0; mount < NEXT_WORLD::N_CAMERAS_PER_AGENT; mount++) {
                    set(device, world.camera_staging_close, _target_camera<DEVICE, TASK_SPEC>(device, setup, mount, aspect), setup_i * NEXT_WORLD::N_VIEWS + mount);
                }
            }
            copy(device, renderer.device, world.camera_staging_close, cameras(device, renderer));
            if constexpr (BASE_SPEC::ENABLE_MOTION_BLUR) {
                copy(device, renderer.device, world.camera_staging_close, cameras_open(device, renderer));
            }
            if constexpr (NEXT_WORLD::RENDERER_SPEC::HAS_FLOW){
                if(world.flow_scene_version != world.scene_version){
                    advance_flow(device, renderer);
                    if constexpr (NEXT_WORLD::RENDERER_SPEC::HAS_FLOW_OVERLAYS){
                        update(device, renderer);
                    }
                }
            }
            render(device, renderer);
            using RGB_SPEC = rendering::raytracing::outputs::TensorSpecification<typename WORLD::RENDERER_SPEC, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>>;
            std::vector<float> observation_staging(RGB_SPEC::SIZE);
            {
                Tensor<rendering::raytracing::outputs::TensorSpecification<typename WORLD::RENDERER_SPEC, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>>> observation_alias;
                observation_alias._data = observation_staging.data();
                copy(renderer.device, device, output(device, renderer, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>{}), observation_alias);
            }
            for (TI setup_i = 0; setup_i < WORLD::INSTANCES; setup_i++) {
                if (mask != nullptr && !mask[setup_i]) {
                    continue;
                }
                const auto& setup = get_ref(device, setups, setup_i);
                const TI rgb_row = NEXT_WORLD::observation_rgb_row(setup_i);
                const float scale = (float)(setup.brightness_scale * setup.brightness_mismatch);
                for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                    for (TI channel_i = 0; channel_i < 3; channel_i++) {
                        set(device, frames, _target_pixel(observation_staging[(rgb_row * CAM_PIXELS + pixel_i) * 3 + channel_i], scale), setup_i, pixel_i * NEXT_WORLD::IMAGE_CHANNELS + channel_i);
                    }
                }
            }
        }
    }
    template <typename DEVICE, typename TASK_SPEC>
    void render_target_bank(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world) {
        using TI = typename TASK_SPEC::TI;
        rl::environments::hyperdrone::tasks::target_frame::_render_targets(device, world, world.setups, world.bank_frames, nullptr);
        set(device, world.bank_cursor, (TI)0, (TI)0);
        world.bank_step = world.history_step;
        world.bank_scene_version = world.scene_version;
    }
    // re-renders the target frames of the masked instances from their own parameters, for callers
    // that move instances after the reset adopted a setup (e.g. replayed spawns)
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC>
    void render_target_frames(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask) {
        rl::environments::hyperdrone::tasks::target_frame::_render_targets(device, world, parameters, world.target_frames, data(reset_mask));
    }
    template <typename DEVICE, typename TASK_SPEC, typename TASK_SPEC::TI N_WORLDS, typename PARAMETER_SPEC, typename RESET_SPEC>
    void render_target_frames(DEVICE& device, rl::environments::HyperDrone<rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>, N_WORLDS>& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask) {
        using HYPERDRONE = rl::environments::HyperDrone<rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>, N_WORLDS>;
        using TI = typename TASK_SPEC::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_world<HYPERDRONE, PARAMETER_SPEC>();
        for (TI world_i = 0; world_i < N_WORLDS; world_i++) {
            auto parameters_block = view_range(device, parameters, world_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, world_i * M, tensor::ViewSpec<0, M>{});
            render_target_frames(device, env.worlds[world_i], parameters_block, reset_block);
        }
    }
    // a target-frame reset adopts a setup and copies its frame per instance, which the batch
    // verbs do; the single-instance entry points would silently fall back to the base World's
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Parameters& parameters, RNG& rng) = delete;
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Parameters& parameters, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::State& state, RNG& rng) = delete;

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        static_assert(utils::typing::is_same_v<typename PARAMETER_SPEC::T, typename WORLD::Parameters>);
        if (rl::environments::hyperdrone::tasks::target_frame::_bank_due(world)) {
            sample_target_setups(device, world, rng);
            render_target_bank(device, world);
        }
        TI cursor = get(device, world.bank_cursor, (TI)0);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                const TI index = cursor % WORLD::INSTANCES;
                cursor++;
                set(device, world.setup_index, index, instance_i);
                rl::environments::hyperdrone::tasks::target_frame::_adopt_setup<DEVICE, TASK_SPEC>(device, world.dynamics, world.drone_parameters, get_ref(device, world.setups, index), get_ref(device, parameters, instance_i), rng);
                auto source = view(device, world.bank_frames, index);
                auto destination = view(device, world.target_frames, instance_i);
                copy(device, device, source, destination);
            }
        }
        set(device, world.bank_cursor, cursor, (TI)0);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        using BASE_SPEC = typename TASK_SPEC::NEXT_WORLD::SPEC;
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename WORLD::State>);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                rl::environments::hyperdrone::_sample_dynamics_state<DEVICE, BASE_SPEC>(device, world.dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
        complete_reset(device, world, reset_mask, rng);
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename TASK_SPEC::NEXT_WORLD::Observation observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, observation_type, observations, rng);
    }

    // the composed observation: [stacked frames | target frame | padding]
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using BASE_SPEC = typename NEXT_WORLD::SPEC;
        constexpr TI INSTANCES = NEXT_WORLD::INSTANCES;
        constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
        constexpr TI IMAGE_CHANNELS = NEXT_WORLD::IMAGE_CHANNELS;
        constexpr TI STACK_N = TASK_SPEC::IMAGE_STACK_N;
        constexpr TI STACK_STRIDE = TASK_SPEC::IMAGE_STACK_STRIDE;
        constexpr TI TOTAL_CHANNELS = WORLD::OBSERVATION_CHANNELS;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::tasks::target_frame::observe: no frame available");
        const TI latest = world.history_step - 1;

        const auto* episode_start = data(world.episode_start);
        const auto* targets = data(world.target_frames);
        for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
            for (TI frame_i = 0; frame_i < STACK_N; frame_i++) {
                const TI back = frame_i * STACK_STRIDE;
                TI desired = latest >= back ? latest - back : episode_start[instance_i];
                if (desired < episode_start[instance_i]) {
                    desired = episode_start[instance_i];
                }
                const TI history_slot = desired % BASE_SPEC::HISTORY_LENGTH;
                auto history_row = view(device, world.history, history_slot);
                auto history_instance = view(device, history_row, instance_i);
                const auto* frame = data(history_instance);
                for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                    for (TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++) {
                        set(device, observations, (T)frame[pixel_i * IMAGE_CHANNELS + channel_i], instance_i, pixel_i * TOTAL_CHANNELS + frame_i * IMAGE_CHANNELS + channel_i);
                    }
                }
            }
            for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                for (TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++) {
                    set(device, observations, (T)targets[instance_i * NEXT_WORLD::FRAME_DIM + pixel_i * IMAGE_CHANNELS + channel_i], instance_i, pixel_i * TOTAL_CHANNELS + STACK_N * IMAGE_CHANNELS + channel_i);
                }
                for (TI channel_i = (STACK_N + 1) * IMAGE_CHANNELS; channel_i < TOTAL_CHANNELS; channel_i++) {
                    set(device, observations, (T)0, instance_i, pixel_i * TOTAL_CHANNELS + channel_i);
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// ADL entry points: generic composite code (e.g. the HyperDrone lifecycle fan-out)
// dispatches member verbs without Tensor arguments, so the task's lifecycle overloads must be
// reachable through the member type's namespace
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame {
    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index){
        ::rl_tools::init(device, world, shared, dataset, corpus, first_scene, num_scenes, member_index);
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, World<TASK_SPEC>& world){
        ::rl_tools::free(device, world);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
