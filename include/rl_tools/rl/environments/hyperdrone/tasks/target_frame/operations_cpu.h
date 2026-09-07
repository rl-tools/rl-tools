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
        template <typename DEVICE, typename TASK_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_initial_parameters(DEVICE& device, typename World<TASK_SPEC>::NEXT_WORLD::DYNAMICS_ENV& dynamics, const typename World<TASK_SPEC>::NEXT_WORLD::Parameters& defaults, typename World<TASK_SPEC>::Parameters& parameters, RNG& rng){
            using T = typename TASK_SPEC::T;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            hyperdrone::_sample_initial_parameters<DEVICE, typename NEXT_WORLD::SPEC>(device, dynamics, defaults, static_cast<typename NEXT_WORLD::Parameters&>(parameters), rng);
            parameters.target_pose[0] = 0;
            parameters.target_pose[1] = 0;
            parameters.target_pose[2] = 0;
            parameters.target_pose[3] = 1;
            parameters.target_pose[4] = 0;
            parameters.target_pose[5] = 0;
            parameters.target_pose[6] = 0;
            if constexpr (TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE > (T)0){
                parameters.target_roll = random::uniform_real_distribution(device.random, -TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng);
                parameters.target_pitch = random::uniform_real_distribution(device.random, -TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TASK_SPEC::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng);
            } else {
                parameters.target_roll = 0;
                parameters.target_pitch = 0;
            }
            if constexpr (TASK_SPEC::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > (T)0){
                parameters.brightness_mismatch = (T)1 + (random::uniform_real_distribution(device.random, (T)0, (T)1, rng) * (T)2 - (T)1) * TASK_SPEC::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
            } else {
                parameters.brightness_mismatch = 1;
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        init(device, static_cast<NEXT_WORLD&>(world), shared, dataset, corpus, first_scene, num_scenes, member_index);
        malloc(device, world.target_frames);
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if (!world.slots.empty()) {
            free(device, world.target_frames);
        }
        free(device, static_cast<NEXT_WORLD&>(world));
    }

    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Parameters& parameters, RNG& rng) {
        rl::environments::hyperdrone::tasks::target_frame::_sample_initial_parameters<DEVICE, TASK_SPEC>(device, world.dynamics, world.parameters, parameters, rng);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        static_assert(utils::typing::is_same_v<typename PARAMETER_SPEC::T, typename WORLD::Parameters>);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                sample_initial_parameters(device, world, get_ref(device, parameters, instance_i), rng);
            }
        }
    }

    // reset-conditional second render pass: the target frames are cached at reset (pixel-identical
    // to per-step re-rendering since pose and brightness are fixed per episode), then the student
    // frame render forwards to the base
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask_input) {
        request_render(device, world, reset_mask_input);
        auto& reset_mask = world.render_reset;
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using BASE_SPEC = typename NEXT_WORLD::SPEC;
        constexpr TI INSTANCES = NEXT_WORLD::INSTANCES;
        bool any_reset = false;
        for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
            any_reset = any_reset || get(device, reset_mask, instance_i);
        }
        if (any_reset) {
            const T aspect = static_cast<T>(BASE_SPEC::CAM_WIDTH) / static_cast<T>(BASE_SPEC::CAM_HEIGHT);
            for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                set(device, world.camera_staging_close, rl::environments::hyperdrone::make_target_camera<DEVICE, T>(device, instance_parameters.camera_mount, instance_parameters.fov, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, instance_parameters.target_roll, instance_parameters.target_pitch), instance_i);
            }
            copy(device, world.renderer.device, world.camera_staging_close, cameras(device, world.renderer));
            if constexpr (BASE_SPEC::ENABLE_MOTION_BLUR) {
                copy(device, world.renderer.device, world.camera_staging_close, cameras_open(device, world.renderer));
            }
            render(device, world.renderer);
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            std::vector<float> observation_staging(INSTANCES * CAM_PIXELS * 3);
            {
                Tensor<tensor::Specification<float, TI, typename decltype(world.renderer.observation)::SPEC::SHAPE>> observation_alias;
                observation_alias._data = observation_staging.data();
                copy(world.renderer.device, device, world.renderer.observation, observation_alias);
            }
            std::vector<float> target_staging(INSTANCES * NEXT_WORLD::FRAME_DIM);
            {
                Tensor<typename WORLD::TARGET_FRAMES_SPEC> target_alias;
                target_alias._data = target_staging.data();
                copy(device, device, world.target_frames, target_alias);
                for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
                    if (!get(device, reset_mask, instance_i)) {
                        continue;
                    }
                    const auto& instance_parameters = get_ref(device, parameters, instance_i);
                    const float scale = (float)(instance_parameters.brightness_scale * instance_parameters.brightness_mismatch);
                    for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                        for (TI channel_i = 0; channel_i < 3; channel_i++) {
                            float value = observation_staging[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
                            value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                            target_staging[instance_i * NEXT_WORLD::FRAME_DIM + pixel_i * NEXT_WORLD::IMAGE_CHANNELS + channel_i] = value;
                        }
                    }
                }
                copy(device, device, target_alias, world.target_frames);
            }
        }
        render(device, static_cast<NEXT_WORLD&>(world), parameters, states, reset_mask);
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

        std::vector<TI> episode_start_staging(INSTANCES);
        {
            Tensor<typename NEXT_WORLD::EPISODE_START_SPEC> episode_start_alias;
            episode_start_alias._data = episode_start_staging.data();
            copy(device, device, world.episode_start, episode_start_alias);
        }
        std::vector<float> target_staging(INSTANCES * NEXT_WORLD::FRAME_DIM);
        {
            Tensor<typename WORLD::TARGET_FRAMES_SPEC> target_alias;
            target_alias._data = target_staging.data();
            copy(device, device, world.target_frames, target_alias);
        }
        std::vector<float> frame_staging(NEXT_WORLD::FRAME_DIM);
        for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
            for (TI frame_i = 0; frame_i < STACK_N; frame_i++) {
                const TI back = frame_i * STACK_STRIDE;
                TI desired = latest >= back ? latest - back : episode_start_staging[instance_i];
                if (desired < episode_start_staging[instance_i]) {
                    desired = episode_start_staging[instance_i];
                }
                const TI history_slot = desired % BASE_SPEC::HISTORY_LENGTH;
                {
                    auto history_row = view(device, world.history, history_slot);
                    auto history_instance = view(device, history_row, instance_i);
                    Tensor<tensor::Specification<float, TI, tensor::Shape<TI, NEXT_WORLD::FRAME_DIM>>> frame_alias;
                    frame_alias._data = frame_staging.data();
                    copy(device, device, history_instance, frame_alias);
                }
                for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                    for (TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++) {
                        set(device, observations, (T)frame_staging[pixel_i * IMAGE_CHANNELS + channel_i], instance_i, pixel_i * TOTAL_CHANNELS + frame_i * IMAGE_CHANNELS + channel_i);
                    }
                }
            }
            for (TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++) {
                for (TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++) {
                    set(device, observations, (T)target_staging[instance_i * NEXT_WORLD::FRAME_DIM + pixel_i * IMAGE_CHANNELS + channel_i], instance_i, pixel_i * TOTAL_CHANNELS + STACK_N * IMAGE_CHANNELS + channel_i);
                }
                for (TI channel_i = (STACK_N + 1) * IMAGE_CHANNELS; channel_i < TOTAL_CHANNELS; channel_i++) {
                    set(device, observations, (T)0, instance_i, pixel_i * TOTAL_CHANNELS + channel_i);
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// ADL entry points: generic composite code (e.g. the MultiEnvironment lifecycle fan-out)
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
