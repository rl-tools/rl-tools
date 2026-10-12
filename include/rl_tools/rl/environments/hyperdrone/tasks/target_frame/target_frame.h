#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_TARGET_FRAME_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_TARGET_FRAME_H

#include "../../world.h"
#include "../../observation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame {

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct ParametersSpecification {
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct ParametersTargetFrame: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        T target_roll = 0;                         // per-episode target-camera perturbation
        T target_pitch = 0;
        T brightness_mismatch = 1;                 // per-episode exposure-mismatch factor
    };

    template <typename T_NEXT_WORLD>
    struct Specification {
        using NEXT_WORLD = T_NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        // derive-and-shadow task knobs
        static constexpr TI IMAGE_STACK_N = 1;
        static constexpr TI IMAGE_STACK_STRIDE = 1;
        static constexpr TI PAD_CHANNELS_TO = 0;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = 0;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0;
        // target frames come from a bank of INSTANCES pre-rendered episode setups that is re-rendered
        // every TARGET_BANK_REFRESH_STEPS frames and on scene changes: one extra render pass per
        // refresh, and resets beyond INSTANCES within one refresh reuse setups
        static constexpr TI TARGET_BANK_REFRESH_STEPS = 32;
    };

    // the wrapper is the World: it adds its target bank and the per-instance target cache,
    // bundles its parameters component and observation contribution, and overloads only the
    // verbs it extends — base verbs bind through deduction-from-derived
    template <typename T_TASK_SPEC>
    struct World: T_TASK_SPEC::NEXT_WORLD {
        using TASK_SPEC = T_TASK_SPEC;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        static_assert(NEXT_WORLD::HAS_IMAGE_OBSERVATION, "target-frame tasks require RGB<Float32> in OUTPUTS");
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;

        using Parameters = ParametersTargetFrame<ParametersSpecification<T, TI, typename NEXT_WORLD::Parameters>>;
        using OBSERVATION_COMPOSITION = hyperdrone::observation::Channels<
            hyperdrone::observation::ImageStack<TI, TASK_SPEC::IMAGE_STACK_N, TASK_SPEC::IMAGE_STACK_STRIDE>,
            hyperdrone::observation::TargetImage<TI>,
            hyperdrone::observation::PadChannelsTo<TI, TASK_SPEC::PAD_CHANNELS_TO>>;
        // the composed channel layout is what the policy CNN input shape derives from
        struct Observation: OBSERVATION_COMPOSITION {
            static constexpr TI HEIGHT = NEXT_WORLD::SPEC::CAM_HEIGHT;
            static constexpr TI WIDTH = NEXT_WORLD::SPEC::CAM_WIDTH;
            static constexpr TI CHANNELS = OBSERVATION_COMPOSITION::template CHANNELS<NEXT_WORLD::IMAGE_CHANNELS>;
            static constexpr TI DIM = HEIGHT * WIDTH * CHANNELS;
            using SHAPE = tensor::Shape<TI, HEIGHT, WIDTH, CHANNELS>;
        };
        static constexpr TI OBSERVATION_CHANNELS = Observation::CHANNELS;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static_assert(NEXT_WORLD::SPEC::HISTORY_LENGTH >= TASK_SPEC::IMAGE_STACK_STRIDE * (TASK_SPEC::IMAGE_STACK_N - 1) + 1, "the base World's frame history must cover the image stack");
        static_assert(NEXT_WORLD::N_AGENTS == 1, "the target-frame task requires one agent per instance");
        static_assert(NEXT_WORLD::N_VIEWS == 1, "the target-frame task requires one camera per instance");
        static_assert(TASK_SPEC::TARGET_BANK_REFRESH_STEPS > 0);

        // a setup is everything the target frame depends on (view, scene placement, target
        // perturbation), sampled into the Parameters a resetting instance adopts whole
        using SETUPS_SPEC = tensor::Specification<Parameters, TI, tensor::Shape<TI, NEXT_WORLD::INSTANCES>>;
        using FRAMES_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, NEXT_WORLD::INSTANCES, NEXT_WORLD::FRAME_DIM>>;
        using SETUP_INDEX_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, NEXT_WORLD::INSTANCES>>;
        using BANK_CURSOR_SPEC = tensor::Specification<TI, TI, tensor::Shape<TI, 1>>;
        Tensor<SETUPS_SPEC> setups;
        Tensor<FRAMES_SPEC> bank_frames;
        Tensor<SETUP_INDEX_SPEC> setup_index;  // the setup each instance adopted at its last reset
        Tensor<BANK_CURSOR_SPEC> bank_cursor;  // setups adopted since the refresh: resets take them in order
        Tensor<FRAMES_SPEC> target_frames;     // copied at reset, so a refresh leaves running episodes alone
        TI bank_step = 0;           // history_step at the refresh
        TI bank_scene_version = 0;  // the scene_version the bank was rendered for, 0 = none
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
