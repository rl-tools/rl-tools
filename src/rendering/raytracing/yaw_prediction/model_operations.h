#pragma once

#include "model.h"
#include <rl_tools/nn_models/parallel/operations_generic.h>

namespace rl_tools {

    // ======================== malloc / free — Model ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model) {
        malloc(device, model.early_encoder_a);
        malloc(device, model.early_encoder_b);

        malloc(device, model.mid_conv_a);
        malloc(device, model.mid_conv_b);
        malloc(device, model.late_encoder_a);
        malloc(device, model.late_encoder_b);
        malloc(device, model.head);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        malloc(device, static_cast<rendering::raytracing::yaw_prediction::ModuleForward<SPEC>&>(model));
        malloc(device, model.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model) {
        free(device, model.early_encoder_a);
        free(device, model.early_encoder_b);

        free(device, model.mid_conv_a);
        free(device, model.mid_conv_b);
        free(device, model.late_encoder_a);
        free(device, model.late_encoder_b);
        free(device, model.head);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        free(device, model.output);
        free(device, static_cast<rendering::raytracing::yaw_prediction::ModuleForward<SPEC>&>(model));
    }

    // ======================== malloc / free — State ========================

    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rendering::raytracing::yaw_prediction::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rendering::raytracing::yaw_prediction::State&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, rendering::raytracing::yaw_prediction::State&, rendering::raytracing::yaw_prediction::State&) {}

    // ======================== malloc / free — Buffer ========================

    template<typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer) {
        malloc(device, buffer.buffer_early_a);
        malloc(device, buffer.buffer_early_b);

        malloc(device, buffer.buffer_mid_a);
        malloc(device, buffer.buffer_mid_b);
        malloc(device, buffer.buffer_late_a);
        malloc(device, buffer.buffer_late_b);
        malloc(device, buffer.buffer_head);
        malloc(device, buffer.features_a);
        malloc(device, buffer.features_b);
        malloc(device, buffer.intermediate_mid_a);
        malloc(device, buffer.intermediate_mid_b);
        malloc(device, buffer.intermediate_a);
        malloc(device, buffer.intermediate_b);
        malloc(device, buffer.concatenated);
        malloc(device, buffer.d_features_a);
        malloc(device, buffer.d_features_b);

        malloc(device, buffer.d_kw_for_b);
        malloc(device, buffer.d_mid_a);
        malloc(device, buffer.d_mid_b);
        malloc(device, buffer.d_concatenated);
        malloc(device, buffer.d_output_a);
        malloc(device, buffer.d_output_b);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer) {
        free(device, buffer.buffer_early_a);
        free(device, buffer.buffer_early_b);

        free(device, buffer.buffer_mid_a);
        free(device, buffer.buffer_mid_b);
        free(device, buffer.buffer_late_a);
        free(device, buffer.buffer_late_b);
        free(device, buffer.buffer_head);
        free(device, buffer.features_a);
        free(device, buffer.features_b);
        free(device, buffer.intermediate_mid_a);
        free(device, buffer.intermediate_mid_b);
        free(device, buffer.intermediate_a);
        free(device, buffer.intermediate_b);
        free(device, buffer.concatenated);
        free(device, buffer.d_features_a);
        free(device, buffer.d_features_b);

        free(device, buffer.d_kw_for_b);
        free(device, buffer.d_mid_a);
        free(device, buffer.d_mid_b);
        free(device, buffer.d_concatenated);
        free(device, buffer.d_output_a);
        free(device, buffer.d_output_b);
    }

    // ======================== init_weights ========================

    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, RNG& rng) {
        init_weights(device, model.early_encoder_a, rng);
        init_weights(device, model.early_encoder_b, rng);

        init_weights(device, model.mid_conv_a, rng);
        init_weights(device, model.mid_conv_b, rng);
        init_weights(device, model.late_encoder_a, rng);
        init_weights(device, model.late_encoder_b, rng);
        init_weights(device, model.head, rng);
    }

    // ======================== evaluate (2-input, CPU-compatible) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, OUTPUT& output, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        // 1. Early encoders
        evaluate(device, model.early_encoder_a, input_a, buffer.features_a, buffer.buffer_early_a, rng, mode);
        evaluate(device, model.early_encoder_b, input_b, buffer.features_b, buffer.buffer_early_b, rng, mode);

        // 2. Mid conv A
        evaluate(device, model.mid_conv_a, buffer.features_a, buffer.intermediate_mid_a, buffer.buffer_mid_a, rng, mode);

        // 3. Mid conv B (cross-conv or independent conv)
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, buffer.intermediate_mid_a);
            evaluate(device, model.mid_conv_b, buffer.features_b, kw_4d, buffer.intermediate_mid_b, buffer.buffer_mid_b, rng, mode);
        } else {
            evaluate(device, model.mid_conv_b, buffer.features_b, buffer.intermediate_mid_b, buffer.buffer_mid_b, rng, mode);
        }

        // 4. Late encoders
        evaluate(device, model.late_encoder_a, buffer.intermediate_mid_a, buffer.intermediate_a, buffer.buffer_late_a, rng, mode);
        evaluate(device, model.late_encoder_b, buffer.intermediate_mid_b, buffer.intermediate_b, buffer.buffer_late_b, rng, mode);

        // 5. Concatenate
        nn_models::parallel::_concatenate(device, buffer.intermediate_a, buffer.intermediate_b, buffer.concatenated);

        // 6. Head
        evaluate(device, model.head, buffer.concatenated, output, buffer.buffer_head, rng, mode);
    }

    // ======================== forward (2-input, training) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename BUFFER_SPEC, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        // 1. Early encoders
        forward(device, model.early_encoder_a, input_a, buffer.buffer_early_a, rng, mode);
        forward(device, model.early_encoder_b, input_b, buffer.buffer_early_b, rng, mode);

        // 2. Save features (contiguous copies for backward)
        {
            auto fa = rl_tools::output(device, model.early_encoder_a);
            auto fb = rl_tools::output(device, model.early_encoder_b);
            copy(device, device, fa, buffer.features_a);
            copy(device, device, fb, buffer.features_b);
        }

        // 3. Mid conv A
        forward(device, model.mid_conv_a, buffer.features_a, buffer.buffer_mid_a, rng, mode);

        // 4. Mid conv B (cross-conv or independent conv)
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, rl_tools::output(device, model.mid_conv_a));
            forward(device, model.mid_conv_b, buffer.features_b, kw_4d, buffer.buffer_mid_b, rng, mode);
        } else {
            forward(device, model.mid_conv_b, buffer.features_b, buffer.buffer_mid_b, rng, mode);
        }

        // 5. Late encoders
        {
            auto mid_a_out = rl_tools::output(device, model.mid_conv_a);
            auto mid_b_out = rl_tools::output(device, model.mid_conv_b);
            forward(device, model.late_encoder_a, mid_a_out, buffer.buffer_late_a, rng, mode);
            forward(device, model.late_encoder_b, mid_b_out, buffer.buffer_late_b, rng, mode);
        }

        // 6. Copy late encoder outputs and concatenate
        {
            auto la = rl_tools::output(device, model.late_encoder_a);
            auto lb = rl_tools::output(device, model.late_encoder_b);
            copy(device, device, la, buffer.intermediate_a);
            copy(device, device, lb, buffer.intermediate_b);
        }
        nn_models::parallel::_concatenate(device, buffer.intermediate_a, buffer.intermediate_b, buffer.concatenated);

        // 7. Head
        forward(device, model.head, buffer.concatenated, buffer.buffer_head, rng, mode);
        {
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        }
    }

    // ======================== backward_full (2-input) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer) {
        // 1. Backward head
        backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.buffer_head);

        // 2. Split d_concatenated
        nn_models::parallel::_split(device, buffer.d_concatenated, buffer.d_output_a, buffer.d_output_b);

        // 3. Backward late encoders
        {
            auto mid_a_out = rl_tools::output(device, model.mid_conv_a);
            auto mid_b_out = rl_tools::output(device, model.mid_conv_b);
            backward_full(device, model.late_encoder_a, mid_a_out, buffer.d_output_a, buffer.d_mid_a, buffer.buffer_late_a);
            backward_full(device, model.late_encoder_b, mid_b_out, buffer.d_output_b, buffer.d_mid_b, buffer.buffer_late_b);
        }

        // 4. Backward mid conv B and accumulate cross-conv gradient
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, rl_tools::output(device, model.mid_conv_a));
            backward_full(device, model.mid_conv_b,
                buffer.features_b, kw_4d, buffer.d_mid_b,
                buffer.d_features_b, buffer.d_kw_for_b, buffer.buffer_mid_b);

            // Accumulate kernel weight gradient into d_mid_a
            using TI = typename SPEC::TI;
            constexpr TI TOTAL = product(typename SPEC::MID_CONV_B_OUTPUT_SHAPE{});
            for (TI i = 0; i < TOTAL; i++) {
                auto val = get_flat(device, buffer.d_mid_a, i);
                auto kw_grad = get_flat(device, buffer.d_kw_for_b, i);
                set_flat(device, buffer.d_mid_a, (decltype(val))((float)val + (float)kw_grad), i);
            }
        } else {
            backward_full(device, model.mid_conv_b,
                buffer.features_b, buffer.d_mid_b,
                buffer.d_features_b, buffer.buffer_mid_b);
        }

        // 5. Backward mid conv A
        backward_full(device, model.mid_conv_a,
            buffer.features_a, buffer.d_mid_a,
            buffer.d_features_a, buffer.buffer_mid_a);

        // 6. Backward early encoders
        backward_full(device, model.early_encoder_a, input_a, buffer.d_features_a, d_input_a, buffer.buffer_early_a);
        backward_full(device, model.early_encoder_b, input_b, buffer.d_features_b, d_input_b, buffer.buffer_early_b);
    }

    // ======================== zero_gradient ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        zero_gradient(device, model.early_encoder_a);
        zero_gradient(device, model.early_encoder_b);

        zero_gradient(device, model.mid_conv_a);
        zero_gradient(device, model.mid_conv_b);
        zero_gradient(device, model.late_encoder_a);
        zero_gradient(device, model.late_encoder_b);
        zero_gradient(device, model.head);
    }

    // ======================== update ========================

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.early_encoder_a, optimizer);
        update(device, model.early_encoder_b, optimizer);

        update(device, model.mid_conv_a, optimizer);
        update(device, model.mid_conv_b, optimizer);
        update(device, model.late_encoder_a, optimizer);
        update(device, model.late_encoder_b, optimizer);
        update(device, model.head, optimizer);
    }

    // ======================== _reset_optimizer_state ========================

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, model.early_encoder_a, optimizer);
        _reset_optimizer_state(device, model.early_encoder_b, optimizer);

        _reset_optimizer_state(device, model.mid_conv_a, optimizer);
        _reset_optimizer_state(device, model.mid_conv_b, optimizer);
        _reset_optimizer_state(device, model.late_encoder_a, optimizer);
        _reset_optimizer_state(device, model.late_encoder_b, optimizer);
        _reset_optimizer_state(device, model.head, optimizer);
    }

    // ======================== reset_forward_state ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model) {
        reset_forward_state(device, model.early_encoder_a);
        reset_forward_state(device, model.early_encoder_b);

        reset_forward_state(device, model.mid_conv_a);
        reset_forward_state(device, model.mid_conv_b);
        reset_forward_state(device, model.late_encoder_a);
        reset_forward_state(device, model.late_encoder_b);
        reset_forward_state(device, model.head);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        reset_forward_state(device, static_cast<rendering::raytracing::yaw_prediction::ModuleForward<SPEC>&>(model));
        set_all(device, model.output, 0);
    }

    // ======================== copy ========================

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device,
            const rendering::raytracing::yaw_prediction::ModuleForward<SOURCE_SPEC>& source,
            rendering::raytracing::yaw_prediction::ModuleForward<TARGET_SPEC>& target) {
        copy(source_device, target_device, source.early_encoder_a, target.early_encoder_a);
        copy(source_device, target_device, source.early_encoder_b, target.early_encoder_b);

        copy(source_device, target_device, source.mid_conv_a, target.mid_conv_a);
        copy(source_device, target_device, source.mid_conv_b, target.mid_conv_b);
        copy(source_device, target_device, source.late_encoder_a, target.late_encoder_a);
        copy(source_device, target_device, source.late_encoder_b, target.late_encoder_b);
        copy(source_device, target_device, source.head, target.head);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device,
            const rendering::raytracing::yaw_prediction::ModuleBackward<SOURCE_SPEC>& source,
            rendering::raytracing::yaw_prediction::ModuleBackward<TARGET_SPEC>& target) {
        copy(source_device, target_device,
            static_cast<const rendering::raytracing::yaw_prediction::ModuleForward<SOURCE_SPEC>&>(source),
            static_cast<rendering::raytracing::yaw_prediction::ModuleForward<TARGET_SPEC>&>(target));
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device,
            const rendering::raytracing::yaw_prediction::ModuleGradient<SOURCE_SPEC>& source,
            rendering::raytracing::yaw_prediction::ModuleGradient<TARGET_SPEC>& target) {
        copy(source_device, target_device,
            static_cast<const rendering::raytracing::yaw_prediction::ModuleForward<SOURCE_SPEC>&>(source),
            static_cast<rendering::raytracing::yaw_prediction::ModuleForward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================

    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device,
            rendering::raytracing::yaw_prediction::ModuleForward<S1>& a,
            const rendering::raytracing::yaw_prediction::ModuleForward<S2>& b) {
        auto diff = abs_diff(device, a.early_encoder_a, b.early_encoder_a);
        diff += abs_diff(device, a.early_encoder_b, b.early_encoder_b);

        diff += abs_diff(device, a.mid_conv_a, b.mid_conv_a);
        diff += abs_diff(device, a.mid_conv_b, b.mid_conv_b);
        diff += abs_diff(device, a.late_encoder_a, b.late_encoder_a);
        diff += abs_diff(device, a.late_encoder_b, b.late_encoder_b);
        diff += abs_diff(device, a.head, b.head);
        return diff;
    }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device,
            rendering::raytracing::yaw_prediction::ModuleGradient<S1>& a,
            const rendering::raytracing::yaw_prediction::ModuleGradient<S2>& b) {
        return abs_diff(device,
            static_cast<rendering::raytracing::yaw_prediction::ModuleForward<S1>&>(a),
            static_cast<const rendering::raytracing::yaw_prediction::ModuleForward<S2>&>(b))
            + abs_diff(device, a.output, b.output);
    }

    // ======================== is_nan ========================

    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device,
            rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model,
            const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, model.early_encoder_a, mode)
            || is_nan(device, model.early_encoder_b, mode)

            || is_nan(device, model.mid_conv_a, mode)
            || is_nan(device, model.mid_conv_b, mode)
            || is_nan(device, model.late_encoder_a, mode)
            || is_nan(device, model.late_encoder_b, mode)
            || is_nan(device, model.head, mode);
    }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device,
            rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model,
            const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, static_cast<rendering::raytracing::yaw_prediction::ModuleForward<SPEC>&>(model), mode)
            || is_nan(device, model.output, mode);
    }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, rendering::raytracing::yaw_prediction::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, model.output);
    }

    // ======================== reset (State) ========================

    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const rendering::raytracing::yaw_prediction::ModuleForward<SPEC>&, rendering::raytracing::yaw_prediction::State&, RNG&, Mode<MODE>) {}

    // ======================== abs_diff (State) ========================

    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE&, const rendering::raytracing::yaw_prediction::State&, const rendering::raytracing::yaw_prediction::State&) { return 0; }
}
