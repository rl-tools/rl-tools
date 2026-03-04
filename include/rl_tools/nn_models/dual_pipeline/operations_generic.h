#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_DUAL_PIPELINE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_DUAL_PIPELINE_OPERATIONS_GENERIC_H

#include "model.h"
#include "../sequential/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::dual_pipeline{
        // Concatenate a and b along the last dimension into output
        // All tensors must be row-major contiguous
        template <typename DEVICE, typename A_SPEC, typename B_SPEC, typename OUTPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _concatenate(DEVICE& device, const Tensor<A_SPEC>& a, const Tensor<B_SPEC>& b, Tensor<OUTPUT_SPEC>& output){
            using TI = typename DEVICE::index_t;
            using A_SHAPE = typename A_SPEC::SHAPE;
            using B_SHAPE = typename B_SPEC::SHAPE;
            using OUTPUT_SHAPE = typename OUTPUT_SPEC::SHAPE;
            constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
            constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
            constexpr TI LAST_DIM_OUT = get_last(OUTPUT_SHAPE{});
            static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
            constexpr TI TOTAL_A = product(A_SHAPE{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            for(TI i = 0; i < LEADING; i++){
                for(TI j = 0; j < LAST_DIM_A; j++){
                    TI a_idx = i * LAST_DIM_A + j;
                    TI out_idx = i * LAST_DIM_OUT + j;
                    *(data(output) + out_idx) = get_flat(device, a, a_idx);
                }
                for(TI j = 0; j < LAST_DIM_B; j++){
                    TI b_idx = i * LAST_DIM_B + j;
                    TI out_idx = i * LAST_DIM_OUT + LAST_DIM_A + j;
                    *(data(output) + out_idx) = get_flat(device, b, b_idx);
                }
            }
        }

        // Split d_output into d_a and d_b (reverse of concatenate)
        template <typename DEVICE, typename D_OUTPUT_SPEC, typename D_A_SPEC, typename D_B_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _split(DEVICE& device, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_A_SPEC>& d_a, Tensor<D_B_SPEC>& d_b){
            using TI = typename DEVICE::index_t;
            using A_SHAPE = typename D_A_SPEC::SHAPE;
            using B_SHAPE = typename D_B_SPEC::SHAPE;
            using OUTPUT_SHAPE = typename D_OUTPUT_SPEC::SHAPE;
            constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
            constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
            constexpr TI LAST_DIM_OUT = get_last(OUTPUT_SHAPE{});
            static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
            constexpr TI TOTAL_A = product(A_SHAPE{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            for(TI i = 0; i < LEADING; i++){
                for(TI j = 0; j < LAST_DIM_A; j++){
                    TI out_idx = i * LAST_DIM_OUT + j;
                    TI a_idx = i * LAST_DIM_A + j;
                    *(data(d_a) + a_idx) = get_flat(device, d_output, out_idx);
                }
                for(TI j = 0; j < LAST_DIM_B; j++){
                    TI out_idx = i * LAST_DIM_OUT + LAST_DIM_A + j;
                    TI b_idx = i * LAST_DIM_B + j;
                    *(data(d_b) + b_idx) = get_flat(device, d_output, out_idx);
                }
            }
        }
    }

    // ======================== malloc / free ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC>& module){
        malloc(device, module.pipeline_a);
        malloc(device, module.pipeline_b);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC>& module){
        free(device, module.pipeline_a);
        free(device, module.pipeline_b);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::dual_pipeline::ModuleBackward<SPEC>& module){
        malloc(device, static_cast<nn_models::dual_pipeline::ModuleForward<SPEC>&>(module));
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::dual_pipeline::ModuleBackward<SPEC>& module){
        free(device, static_cast<nn_models::dual_pipeline::ModuleForward<SPEC>&>(module));
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& module){
        malloc(device, static_cast<nn_models::dual_pipeline::ModuleBackward<SPEC>&>(module));
        malloc(device, module.output);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& module){
        free(device, static_cast<nn_models::dual_pipeline::ModuleBackward<SPEC>&>(module));
        free(device, module.output);
    }

    // Buffer malloc/free
    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer){
        malloc(device, buffer.buffer_a);
        malloc(device, buffer.buffer_b);
        malloc(device, buffer.intermediate_a);
        malloc(device, buffer.intermediate_b);
        malloc(device, buffer.d_output_a);
        malloc(device, buffer.d_output_b);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer){
        free(device, buffer.buffer_a);
        free(device, buffer.buffer_b);
        free(device, buffer.intermediate_a);
        free(device, buffer.intermediate_b);
        free(device, buffer.d_output_a);
        free(device, buffer.d_output_b);
    }

    // State malloc/free
    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::dual_pipeline::ModuleState<STATE_SPEC>& state){
        malloc(device, state.state_a);
        malloc(device, state.state_b);
    }
    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::dual_pipeline::ModuleState<STATE_SPEC>& state){
        free(device, state.state_a);
        free(device, state.state_b);
    }

    // ======================== init_weights ========================
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC>& module, RNG& rng){
        init_weights(device, module.pipeline_a, rng);
        init_weights(device, module.pipeline_b, rng);
    }

    // ======================== evaluate ========================
    template <typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn_models::dual_pipeline::ModuleForward<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, OUTPUT& output, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, model.pipeline_a, input_a, buffer.intermediate_a, buffer.buffer_a, rng, mode);
        evaluate(device, model.pipeline_b, input_b, buffer.intermediate_b, buffer.buffer_b, rng, mode);
        nn_models::dual_pipeline::_concatenate(device, buffer.intermediate_a, buffer.intermediate_b, output);
    }

    // ======================== forward ========================
    template <typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model.pipeline_a, input_a, buffer.buffer_a, rng, mode);
        forward(device, model.pipeline_b, input_b, buffer.buffer_b, rng, mode);
        auto output_a = rl_tools::output(device, model.pipeline_a);
        auto output_b = rl_tools::output(device, model.pipeline_b);
        copy(device, device, output_a, buffer.intermediate_a);
        copy(device, device, output_b, buffer.intermediate_b);
        nn_models::dual_pipeline::_concatenate(device, buffer.intermediate_a, buffer.intermediate_b, model.output);
    }

    template <typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, OUTPUT& output, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model, input_a, input_b, buffer, rng, mode);
        copy(device, device, model.output, output);
    }

    // ======================== backward_full ========================
    template <typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::dual_pipeline::_split(device, d_output, buffer.d_output_a, buffer.d_output_b);
        backward_full(device, model.pipeline_a, input_a, buffer.d_output_a, d_input_a, buffer.buffer_a, mode);
        backward_full(device, model.pipeline_b, input_b, buffer.d_output_b, d_input_b, buffer.buffer_b, mode);
    }

    // ======================== backward (gradients only) ========================
    template <typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, D_OUTPUT& d_output, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::dual_pipeline::_split(device, d_output, buffer.d_output_a, buffer.d_output_b);
        backward(device, model.pipeline_a, input_a, buffer.d_output_a, buffer.buffer_a, mode);
        backward(device, model.pipeline_b, input_b, buffer.d_output_b, buffer.buffer_b, mode);
    }

    // ======================== backward_input ========================
    template <typename DEVICE, typename SPEC, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, nn_models::dual_pipeline::ModuleBackward<SPEC>& model, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, nn_models::dual_pipeline::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::dual_pipeline::_split(device, d_output, buffer.d_output_a, buffer.d_output_b);
        backward_input(device, model.pipeline_a, buffer.d_output_a, d_input_a, buffer.buffer_a, mode);
        backward_input(device, model.pipeline_b, buffer.d_output_b, d_input_b, buffer.buffer_b, mode);
    }

    // ======================== zero_gradient ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& module){
        zero_gradient(device, module.pipeline_a);
        zero_gradient(device, module.pipeline_b);
    }

    // ======================== update ========================
    template <typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer){
        update(device, module.pipeline_a, optimizer);
        update(device, module.pipeline_b, optimizer);
    }

    // ======================== _reset_optimizer_state ========================
    template <typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer){
        _reset_optimizer_state(device, module.pipeline_a, optimizer);
        _reset_optimizer_state(device, module.pipeline_b, optimizer);
    }

    // ======================== reset_forward_state ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC>& module){
        reset_forward_state(device, module.pipeline_a);
        reset_forward_state(device, module.pipeline_b);
    }

    // ======================== copy ========================
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn_models::dual_pipeline::ModuleForward<SOURCE_SPEC>& source, nn_models::dual_pipeline::ModuleForward<TARGET_SPEC>& target){
        copy(source_device, target_device, source.pipeline_a, target.pipeline_a);
        copy(source_device, target_device, source.pipeline_b, target.pipeline_b);
    }

    // ======================== abs_diff ========================
    template <typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC_A>& a, const nn_models::dual_pipeline::ModuleForward<SPEC_B>& b){
        using T = typename SPEC_A::TYPE_POLICY::DEFAULT;
        return static_cast<T>(abs_diff(device, a.pipeline_a, b.pipeline_a)) + static_cast<T>(abs_diff(device, a.pipeline_b, b.pipeline_b));
    }

    // ======================== is_nan ========================
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn_models::dual_pipeline::ModuleForward<SPEC>& model, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return is_nan(device, model.pipeline_a, mode) || is_nan(device, model.pipeline_b, mode);
    }

    // ======================== output ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn_models::dual_pipeline::ModuleGradient<SPEC>& model){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, model.output);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, const nn_models::dual_pipeline::ModuleGradient<SPEC>& model){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, model.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
