#pragma once

#include "model.h"

namespace rl_tools {

    // ======================== malloc / free — Model ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model) {
        malloc(device, model.early_encoder_a);
        malloc(device, model.early_encoder_b);
        malloc(device, model.kernel_gen_a);
        malloc(device, model.kernel_gen_b);
        malloc(device, model.cross_conv_a);
        malloc(device, model.cross_conv_b);
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
        free(device, model.kernel_gen_a);
        free(device, model.kernel_gen_b);
        free(device, model.cross_conv_a);
        free(device, model.cross_conv_b);
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
        malloc(device, buffer.buffer_kg_a);
        malloc(device, buffer.buffer_kg_b);
        malloc(device, buffer.buffer_cross_a);
        malloc(device, buffer.buffer_cross_b);
        malloc(device, buffer.buffer_late_a);
        malloc(device, buffer.buffer_late_b);
        malloc(device, buffer.buffer_head);
        malloc(device, buffer.features_a);
        malloc(device, buffer.features_b);
        malloc(device, buffer.intermediate_a);
        malloc(device, buffer.intermediate_b);
        malloc(device, buffer.concatenated);
        malloc(device, buffer.d_features_a);
        malloc(device, buffer.d_features_b);
        malloc(device, buffer.d_features_temp);
        malloc(device, buffer.d_kw_for_a);
        malloc(device, buffer.d_kw_for_b);
        malloc(device, buffer.d_cross_a);
        malloc(device, buffer.d_cross_b);
        malloc(device, buffer.d_concatenated);
        malloc(device, buffer.d_output_a);
        malloc(device, buffer.d_output_b);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer) {
        free(device, buffer.buffer_early_a);
        free(device, buffer.buffer_early_b);
        free(device, buffer.buffer_kg_a);
        free(device, buffer.buffer_kg_b);
        free(device, buffer.buffer_cross_a);
        free(device, buffer.buffer_cross_b);
        free(device, buffer.buffer_late_a);
        free(device, buffer.buffer_late_b);
        free(device, buffer.buffer_head);
        free(device, buffer.features_a);
        free(device, buffer.features_b);
        free(device, buffer.intermediate_a);
        free(device, buffer.intermediate_b);
        free(device, buffer.concatenated);
        free(device, buffer.d_features_a);
        free(device, buffer.d_features_b);
        free(device, buffer.d_features_temp);
        free(device, buffer.d_kw_for_a);
        free(device, buffer.d_kw_for_b);
        free(device, buffer.d_cross_a);
        free(device, buffer.d_cross_b);
        free(device, buffer.d_concatenated);
        free(device, buffer.d_output_a);
        free(device, buffer.d_output_b);
    }

    // ======================== init_weights ========================

    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, RNG& rng) {
        init_weights(device, model.early_encoder_a, rng);
        init_weights(device, model.early_encoder_b, rng);
        init_weights(device, model.kernel_gen_a, rng);
        init_weights(device, model.kernel_gen_b, rng);
        init_weights(device, model.cross_conv_a, rng);
        init_weights(device, model.cross_conv_b, rng);
        init_weights(device, model.late_encoder_a, rng);
        init_weights(device, model.late_encoder_b, rng);
        init_weights(device, model.head, rng);
    }

    // ======================== zero_gradient ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model) {
        zero_gradient(device, model.early_encoder_a);
        zero_gradient(device, model.early_encoder_b);
        zero_gradient(device, model.kernel_gen_a);
        zero_gradient(device, model.kernel_gen_b);
        zero_gradient(device, model.cross_conv_a);
        zero_gradient(device, model.cross_conv_b);
        zero_gradient(device, model.late_encoder_a);
        zero_gradient(device, model.late_encoder_b);
        zero_gradient(device, model.head);
    }

    // ======================== update ========================

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.early_encoder_a, optimizer);
        update(device, model.early_encoder_b, optimizer);
        update(device, model.kernel_gen_a, optimizer);
        update(device, model.kernel_gen_b, optimizer);
        update(device, model.cross_conv_a, optimizer);
        update(device, model.cross_conv_b, optimizer);
        update(device, model.late_encoder_a, optimizer);
        update(device, model.late_encoder_b, optimizer);
        update(device, model.head, optimizer);
    }

    // ======================== _reset_optimizer_state ========================

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, model.early_encoder_a, optimizer);
        _reset_optimizer_state(device, model.early_encoder_b, optimizer);
        _reset_optimizer_state(device, model.kernel_gen_a, optimizer);
        _reset_optimizer_state(device, model.kernel_gen_b, optimizer);
        _reset_optimizer_state(device, model.cross_conv_a, optimizer);
        _reset_optimizer_state(device, model.cross_conv_b, optimizer);
        _reset_optimizer_state(device, model.late_encoder_a, optimizer);
        _reset_optimizer_state(device, model.late_encoder_b, optimizer);
        _reset_optimizer_state(device, model.head, optimizer);
    }

    // ======================== reset_forward_state ========================

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model) {
        reset_forward_state(device, model.early_encoder_a);
        reset_forward_state(device, model.early_encoder_b);
        reset_forward_state(device, model.kernel_gen_a);
        reset_forward_state(device, model.kernel_gen_b);
        reset_forward_state(device, model.cross_conv_a);
        reset_forward_state(device, model.cross_conv_b);
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
        copy(source_device, target_device, source.kernel_gen_a, target.kernel_gen_a);
        copy(source_device, target_device, source.kernel_gen_b, target.kernel_gen_b);
        copy(source_device, target_device, source.cross_conv_a, target.cross_conv_a);
        copy(source_device, target_device, source.cross_conv_b, target.cross_conv_b);
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
        diff += abs_diff(device, a.kernel_gen_a, b.kernel_gen_a);
        diff += abs_diff(device, a.kernel_gen_b, b.kernel_gen_b);
        diff += abs_diff(device, a.cross_conv_a, b.cross_conv_a);
        diff += abs_diff(device, a.cross_conv_b, b.cross_conv_b);
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
            || is_nan(device, model.kernel_gen_a, mode)
            || is_nan(device, model.kernel_gen_b, mode)
            || is_nan(device, model.cross_conv_a, mode)
            || is_nan(device, model.cross_conv_b, mode)
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
