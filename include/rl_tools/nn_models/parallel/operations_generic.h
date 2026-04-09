#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_OPERATIONS_GENERIC_H

#include "model.h"
#include "../sequential/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::parallel{
        namespace detail{
            template <typename TI>
            RL_TOOLS_FUNCTION_PLACEMENT void _fill_tuple(utils::Tuple<TI>&){}
            template <typename TI, typename HEAD, typename... TAIL, typename FIRST, typename... REST>
            RL_TOOLS_FUNCTION_PLACEMENT void _fill_tuple(utils::Tuple<TI, HEAD, TAIL...>& tuple, const FIRST& first, const REST&... rest){
                tuple.content = first;
                _fill_tuple(static_cast<utils::Tuple<TI, TAIL...>&>(tuple), rest...);
            }
        }

        template <typename FIRST, typename... REST>
        RL_TOOLS_FUNCTION_PLACEMENT utils::Tuple<typename FIRST::SPEC::SHAPE::TI, FIRST, REST...> pack_inputs(const FIRST& first, const REST&... rest){
            using TI = typename FIRST::SPEC::SHAPE::TI;
            utils::Tuple<TI, FIRST, REST...> result{};
            detail::_fill_tuple(result, first, rest...);
            return result;
        }

        // 2-tensor concatenation (convenience for custom models that manage their own buffers)
        template <typename DEVICE, typename A_SPEC, typename B_SPEC, typename OUTPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _concatenate(DEVICE& device, const Tensor<A_SPEC>& a, const Tensor<B_SPEC>& b, Tensor<OUTPUT_SPEC>& output){
            using TI = typename DEVICE::index_t;
            using A_SHAPE = typename A_SPEC::SHAPE;
            using B_SHAPE = typename B_SPEC::SHAPE;
            using OUT_SHAPE = typename OUTPUT_SPEC::SHAPE;
            constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
            constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
            constexpr TI LAST_DIM_OUT = get_last(OUT_SHAPE{});
            static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
            constexpr TI TOTAL_A = product(A_SHAPE{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            for(TI i = 0; i < LEADING; i++){
                for(TI j = 0; j < LAST_DIM_A; j++){
                    *(data(output) + i * LAST_DIM_OUT + j) = get_flat(device, a, i * LAST_DIM_A + j);
                }
                for(TI j = 0; j < LAST_DIM_B; j++){
                    *(data(output) + i * LAST_DIM_OUT + LAST_DIM_A + j) = get_flat(device, b, i * LAST_DIM_B + j);
                }
            }
        }

        template <typename DEVICE, typename D_OUTPUT_SPEC, typename D_A_SPEC, typename D_B_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _split(DEVICE& device, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_A_SPEC>& d_a, Tensor<D_B_SPEC>& d_b){
            using TI = typename DEVICE::index_t;
            using A_SHAPE = typename D_A_SPEC::SHAPE;
            using B_SHAPE = typename D_B_SPEC::SHAPE;
            using OUT_SHAPE = typename D_OUTPUT_SPEC::SHAPE;
            constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
            constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
            constexpr TI LAST_DIM_OUT = get_last(OUT_SHAPE{});
            static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
            constexpr TI TOTAL_A = product(A_SHAPE{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            for(TI i = 0; i < LEADING; i++){
                for(TI j = 0; j < LAST_DIM_A; j++){
                    *(data(d_a) + i * LAST_DIM_A + j) = get_flat(device, d_output, i * LAST_DIM_OUT + j);
                }
                for(TI j = 0; j < LAST_DIM_B; j++){
                    *(data(d_b) + i * LAST_DIM_B + j) = get_flat(device, d_output, i * LAST_DIM_OUT + LAST_DIM_A + j);
                }
            }
        }

        template <auto NUM_BRANCHES, auto I = 0, auto OFFSET = 0, typename DEVICE, typename INTERMEDIATES, typename OUTPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _concatenate_n(DEVICE& device, const INTERMEDIATES& intermediates, Tensor<OUTPUT_SPEC>& output){
            if constexpr(I < NUM_BRANCHES){
                using TI = typename DEVICE::index_t;
                using OUTPUT_SHAPE = typename OUTPUT_SPEC::SHAPE;
                constexpr TI LAST_DIM_OUT = get_last(OUTPUT_SHAPE{});
                constexpr TI LEADING = product(OUTPUT_SHAPE{}) / LAST_DIM_OUT;
                const auto& src = get<I>(intermediates);
                using SRC_SHAPE = typename utils::typing::remove_reference_t<decltype(src)>::SPEC::SHAPE;
                constexpr TI DIM_I = get_last(SRC_SHAPE{});
                for(TI i = 0; i < LEADING; i++){
                    for(TI j = 0; j < DIM_I; j++){
                        TI src_idx = i * DIM_I + j;
                        TI out_idx = i * LAST_DIM_OUT + (TI)OFFSET + j;
                        *(data(output) + out_idx) = get_flat(device, src, src_idx);
                    }
                }
                _concatenate_n<NUM_BRANCHES, I + 1, OFFSET + DIM_I>(device, intermediates, output);
            }
        }

        // N-way split: extract each branch's gradient from the concatenated gradient
        template <auto NUM_BRANCHES, auto I = 0, auto OFFSET = 0, typename DEVICE, typename D_OUTPUT_SPEC, typename D_OUTPUTS>
        RL_TOOLS_FUNCTION_PLACEMENT void _split_n(DEVICE& device, const Tensor<D_OUTPUT_SPEC>& d_output, D_OUTPUTS& d_outputs){
            if constexpr(I < NUM_BRANCHES){
                using TI = typename DEVICE::index_t;
                using D_OUTPUT_SHAPE = typename D_OUTPUT_SPEC::SHAPE;
                constexpr TI LAST_DIM_OUT = get_last(D_OUTPUT_SHAPE{});
                auto& dst = get<I>(d_outputs);
                using DST_SHAPE = typename utils::typing::remove_reference_t<decltype(dst)>::SPEC::SHAPE;
                constexpr TI DIM_I = get_last(DST_SHAPE{});
                constexpr TI TOTAL_DST = product(DST_SHAPE{});
                constexpr TI LEADING = TOTAL_DST / DIM_I;
                for(TI i = 0; i < LEADING; i++){
                    for(TI j = 0; j < DIM_I; j++){
                        TI out_idx = i * LAST_DIM_OUT + (TI)OFFSET + j;
                        TI dst_idx = i * DIM_I + j;
                        *(data(dst) + dst_idx) = get_flat(device, d_output, out_idx);
                    }
                }
                _split_n<NUM_BRANCHES, I + 1, OFFSET + DIM_I>(device, d_output, d_outputs);
            }
        }

        // Branch iteration helpers
        template <auto I = 0, typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _evaluate_branches(DEVICE& device, const ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                evaluate(device, get<I>(model.pipelines), get<I>(inputs), get<I>(buffer.intermediates), get<I>(buffer.sub_buffers), rng, mode);
                _evaluate_branches<I + 1>(device, model, inputs, buffer, rng, mode);
            }
        }

        template <auto I = 0, auto OFFSET = 0, typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename STATE_SPEC, typename CONCAT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _evaluate_step_branches(DEVICE& device, const ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, ModuleState<STATE_SPEC>& state, Tensor<CONCAT_SPEC>& concatenated, ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                using TI = typename DEVICE::index_t;
                using INTERMEDIATE_SHAPE = typename utils::typing::remove_reference_t<decltype(get<I>(buffer.intermediates))>::SPEC::SHAPE;
                constexpr TI DIM_I = get_last(INTERMEDIATE_SHAPE{});
                auto output_view = view_range(device, concatenated, (TI)OFFSET, tensor::ViewSpec<1, DIM_I>{});
                evaluate_step(device, get<I>(model.pipelines), get<I>(inputs), get<I>(state.states), output_view, get<I>(buffer.sub_buffers), rng, mode);
                _evaluate_step_branches<I + 1, OFFSET + DIM_I>(device, model, inputs, state, concatenated, buffer, rng, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _forward_branches(DEVICE& device, ModuleGradient<SPEC>& model, INPUT_TUPLE& inputs, ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                forward(device, get<I>(model.pipelines), get<I>(inputs), get<I>(buffer.sub_buffers), rng, mode);
                auto output_i = rl_tools::output(device, get<I>(model.pipelines));
                copy(device, device, output_i, get<I>(buffer.intermediates));
                _forward_branches<I + 1>(device, model, inputs, buffer, rng, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _backward_full_branches(DEVICE& device, ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, D_INPUT_TUPLE& d_inputs, ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                backward_full(device, get<I>(model.pipelines), get<I>(inputs), get<I>(buffer.d_outputs), get<I>(d_inputs), get<I>(buffer.sub_buffers), mode);
                _backward_full_branches<I + 1>(device, model, inputs, d_inputs, buffer, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename BUFFER_SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _backward_branches(DEVICE& device, ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                backward(device, get<I>(model.pipelines), get<I>(inputs), get<I>(buffer.d_outputs), get<I>(buffer.sub_buffers), mode);
                _backward_branches<I + 1>(device, model, inputs, buffer, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _backward_input_branches(DEVICE& device, ModuleBackward<SPEC>& model, D_INPUT_TUPLE& d_inputs, ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                backward_input(device, get<I>(model.pipelines), get<I>(buffer.d_outputs), get<I>(d_inputs), get<I>(buffer.sub_buffers), mode);
                _backward_input_branches<I + 1>(device, model, d_inputs, buffer, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _malloc_pipelines(DEVICE& device, ModuleForward<SPEC>& model){
            if constexpr(I < SPEC::NUM_BRANCHES){
                malloc(device, get<I>(model.pipelines));
                _malloc_pipelines<I + 1>(device, model);
            }
        }
        template <auto I = 0, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _free_pipelines(DEVICE& device, ModuleForward<SPEC>& model){
            if constexpr(I < SPEC::NUM_BRANCHES){
                free(device, get<I>(model.pipelines));
                _free_pipelines<I + 1>(device, model);
            }
        }
        template <auto I = 0, typename DEVICE, typename BUFFER_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _malloc_buffer_tuples(DEVICE& device, ModuleBuffer<BUFFER_SPEC>& buffer){
            if constexpr(I < BUFFER_SPEC::SPEC::NUM_BRANCHES){
                malloc(device, get<I>(buffer.sub_buffers));
                malloc(device, get<I>(buffer.intermediates));
                malloc(device, get<I>(buffer.d_outputs));
                _malloc_buffer_tuples<I + 1>(device, buffer);
            }
        }
        template <auto I = 0, typename DEVICE, typename BUFFER_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _free_buffer_tuples(DEVICE& device, ModuleBuffer<BUFFER_SPEC>& buffer){
            if constexpr(I < BUFFER_SPEC::SPEC::NUM_BRANCHES){
                free(device, get<I>(buffer.sub_buffers));
                free(device, get<I>(buffer.intermediates));
                free(device, get<I>(buffer.d_outputs));
                _free_buffer_tuples<I + 1>(device, buffer);
            }
        }
        template <auto I = 0, typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _malloc_states(DEVICE& device, ModuleState<STATE_SPEC>& state){
            if constexpr(I < STATE_SPEC::SPEC::NUM_BRANCHES){
                malloc(device, get<I>(state.states));
                _malloc_states<I + 1>(device, state);
            }
        }
        template <auto I = 0, typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _free_states(DEVICE& device, ModuleState<STATE_SPEC>& state){
            if constexpr(I < STATE_SPEC::SPEC::NUM_BRANCHES){
                free(device, get<I>(state.states));
                _free_states<I + 1>(device, state);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _init_weights_branches(DEVICE& device, ModuleForward<SPEC>& model, RNG& rng){
            if constexpr(I < SPEC::NUM_BRANCHES){
                init_weights(device, get<I>(model.pipelines), rng);
                _init_weights_branches<I + 1>(device, model, rng);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void _reset_branches(DEVICE& device, const ModuleForward<SPEC>& model, ModuleState<STATE_SPEC>& state, RNG& rng, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                reset(device, get<I>(model.pipelines), get<I>(state.states), rng, mode);
                _reset_branches<I + 1>(device, model, state, rng, mode);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _zero_gradient_branches(DEVICE& device, ModuleGradient<SPEC>& model){
            if constexpr(I < SPEC::NUM_BRANCHES){
                zero_gradient(device, get<I>(model.pipelines));
                _zero_gradient_branches<I + 1>(device, model);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename OPTIMIZER>
        RL_TOOLS_FUNCTION_PLACEMENT void _update_branches(DEVICE& device, ModuleGradient<SPEC>& model, OPTIMIZER& optimizer){
            if constexpr(I < SPEC::NUM_BRANCHES){
                update(device, get<I>(model.pipelines), optimizer);
                _update_branches<I + 1>(device, model, optimizer);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename OPTIMIZER>
        RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state_branches(DEVICE& device, ModuleGradient<SPEC>& model, OPTIMIZER& optimizer){
            if constexpr(I < SPEC::NUM_BRANCHES){
                _reset_optimizer_state(device, get<I>(model.pipelines), optimizer);
                _reset_optimizer_state_branches<I + 1>(device, model, optimizer);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _reset_forward_state_branches(DEVICE& device, ModuleForward<SPEC>& model){
            if constexpr(I < SPEC::NUM_BRANCHES){
                reset_forward_state(device, get<I>(model.pipelines));
                _reset_forward_state_branches<I + 1>(device, model);
            }
        }

        template <auto I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _copy_branches(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const ModuleForward<SOURCE_SPEC>& source, ModuleForward<TARGET_SPEC>& target){
            if constexpr(I < SOURCE_SPEC::NUM_BRANCHES){
                copy(source_device, target_device, get<I>(source.pipelines), get<I>(target.pipelines));
                _copy_branches<I + 1>(source_device, target_device, source, target);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC_A, typename SPEC_B>
        RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::TYPE_POLICY::DEFAULT _abs_diff_branches(DEVICE& device, ModuleForward<SPEC_A>& a, const ModuleForward<SPEC_B>& b){
            using T = typename SPEC_A::TYPE_POLICY::DEFAULT;
            if constexpr(I < SPEC_A::NUM_BRANCHES){
                return static_cast<T>(abs_diff(device, get<I>(a.pipelines), get<I>(b.pipelines))) + _abs_diff_branches<I + 1>(device, a, b);
            }
            else{
                return static_cast<T>(0);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT bool _is_nan_branches(DEVICE& device, ModuleForward<SPEC>& model, const Mode<MODE>& mode){
            if constexpr(I < SPEC::NUM_BRANCHES){
                return is_nan(device, get<I>(model.pipelines), mode) || _is_nan_branches<I + 1>(device, model, mode);
            }
            else{
                return false;
            }
        }
    }

    // ======================== malloc / free ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& module){
        nn_models::parallel::_malloc_pipelines(device, module);
        if constexpr(SPEC::HAS_HEAD){
            malloc(device, module.head);
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& module){
        nn_models::parallel::_free_pipelines(device, module);
        if constexpr(SPEC::HAS_HEAD){
            free(device, module.head);
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::parallel::ModuleBackward<SPEC>& module){
        malloc(device, static_cast<nn_models::parallel::ModuleForward<SPEC>&>(module));
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::parallel::ModuleBackward<SPEC>& module){
        free(device, static_cast<nn_models::parallel::ModuleForward<SPEC>&>(module));
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& module){
        malloc(device, static_cast<nn_models::parallel::ModuleBackward<SPEC>&>(module));
        malloc(device, module.output);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& module){
        free(device, static_cast<nn_models::parallel::ModuleBackward<SPEC>&>(module));
        free(device, module.output);
    }

    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer){
        using SPEC = typename BUFFER_SPEC::SPEC;
        nn_models::parallel::_malloc_buffer_tuples(device, buffer);
        malloc(device, buffer.concatenated);
        malloc(device, buffer.d_concatenated);
        if constexpr(SPEC::HAS_HEAD){
            malloc(device, buffer.head_buffer);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer){
        using SPEC = typename BUFFER_SPEC::SPEC;
        nn_models::parallel::_free_buffer_tuples(device, buffer);
        free(device, buffer.concatenated);
        free(device, buffer.d_concatenated);
        if constexpr(SPEC::HAS_HEAD){
            free(device, buffer.head_buffer);
        }
    }

    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::parallel::ModuleState<STATE_SPEC>& state){
        using SPEC = typename STATE_SPEC::SPEC;
        nn_models::parallel::_malloc_states(device, state);
        if constexpr(SPEC::HAS_HEAD){
            malloc(device, state.head_state);
        }
    }
    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::parallel::ModuleState<STATE_SPEC>& state){
        using SPEC = typename STATE_SPEC::SPEC;
        nn_models::parallel::_free_states(device, state);
        if constexpr(SPEC::HAS_HEAD){
            free(device, state.head_state);
        }
    }

    // ======================== init_weights ========================
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& module, RNG& rng){
        nn_models::parallel::_init_weights_branches(device, module, rng);
        if constexpr(SPEC::HAS_HEAD){
            init_weights(device, module.head, rng);
        }
    }

    // ======================== reset ========================
    template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn_models::parallel::ModuleForward<SPEC>& model, nn_models::parallel::ModuleState<STATE_SPEC>& state, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::parallel::_reset_branches(device, model, state, rng, mode);
        if constexpr(SPEC::HAS_HEAD){
            reset(device, model.head, state.head_state, rng, mode);
        }
    }

    // ======================== evaluate_step ========================
    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename STATE_SPEC, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, nn_models::parallel::ModuleState<STATE_SPEC>& state, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename SPEC::TI;
        using OUTPUT_SHAPE = typename OUTPUT::SPEC::SHAPE;
        using CONCAT_BUFFER_SHAPE = typename utils::typing::remove_reference_t<decltype(buffer.concatenated)>::SPEC::SHAPE;
        constexpr TI CONCAT_LAST_DIM = get_last(CONCAT_BUFFER_SHAPE{});
        using CONCAT_VIEW_SHAPE = tensor::Replace<OUTPUT_SHAPE, CONCAT_LAST_DIM, length(OUTPUT_SHAPE{}) - 1>;
        auto concat_view = view_memory<CONCAT_VIEW_SHAPE>(device, buffer.concatenated);
        nn_models::parallel::_evaluate_step_branches(device, model, inputs, state, concat_view, buffer, rng, mode);
        if constexpr(SPEC::HAS_HEAD){
            evaluate_step(device, model.head, concat_view, state.head_state, output, buffer.head_buffer, rng, mode);
        }
        else{
            copy(device, device, concat_view, output);
        }
    }

    // ======================== evaluate ========================
    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::parallel::_evaluate_branches(device, model, inputs, buffer, rng, mode);
        nn_models::parallel::_concatenate_n<SPEC::NUM_BRANCHES>(device, buffer.intermediates, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            evaluate(device, model.head, buffer.concatenated, output, buffer.head_buffer, rng, mode);
        }
        else{
            copy(device, device, buffer.concatenated, output);
        }
    }

    // ======================== forward ========================
    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_TUPLE& inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::parallel::_forward_branches(device, model, inputs, buffer, rng, mode);
        nn_models::parallel::_concatenate_n<SPEC::NUM_BRANCHES>(device, buffer.intermediates, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            forward(device, model.head, buffer.concatenated, buffer.head_buffer, rng, mode);
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        }
        else{
            copy(device, device, buffer.concatenated, model.output);
        }
    }

    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_TUPLE& inputs, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model, inputs, buffer, rng, mode);
        copy(device, device, model.output, output);
    }

    // ======================== backward_full ========================
    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename D_OUTPUT, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, D_OUTPUT& d_output, D_INPUT_TUPLE& d_inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_full_branches(device, model, inputs, d_inputs, buffer, mode);
    }

    // ======================== backward (gradients only) ========================
    template <typename DEVICE, typename SPEC, typename INPUT_TUPLE, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename utils::typing::enable_if_t<nn_models::parallel::detail::is_input_tuple<utils::typing::remove_reference_t<INPUT_TUPLE>>::value>* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, D_OUTPUT& d_output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_branches(device, model, inputs, buffer, mode);
    }

    // ======================== backward_input ========================
    template <typename DEVICE, typename SPEC, typename D_OUTPUT, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, nn_models::parallel::ModuleBackward<SPEC>& model, D_OUTPUT& d_output, D_INPUT_TUPLE& d_inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_input(device, model.head, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            nn_models::parallel::_split_n<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_input_branches(device, model, d_inputs, buffer, mode);
    }

    // ======================== zero_gradient ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& module){
        nn_models::parallel::_zero_gradient_branches(device, module);
        if constexpr(SPEC::HAS_HEAD){
            zero_gradient(device, module.head);
        }
    }

    // ======================== update ========================
    template <typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer){
        nn_models::parallel::_update_branches(device, module, optimizer);
        if constexpr(SPEC::HAS_HEAD){
            update(device, module.head, optimizer);
        }
    }

    // ======================== _reset_optimizer_state ========================
    template <typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer){
        nn_models::parallel::_reset_optimizer_state_branches(device, module, optimizer);
        if constexpr(SPEC::HAS_HEAD){
            _reset_optimizer_state(device, module.head, optimizer);
        }
    }

    // ======================== reset_forward_state ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& module){
        nn_models::parallel::_reset_forward_state_branches(device, module);
        if constexpr(SPEC::HAS_HEAD){
            reset_forward_state(device, module.head);
        }
    }

    // ======================== copy ========================
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn_models::parallel::ModuleForward<SOURCE_SPEC>& source, nn_models::parallel::ModuleForward<TARGET_SPEC>& target){
        nn_models::parallel::_copy_branches(source_device, target_device, source, target);
        if constexpr(SOURCE_SPEC::HAS_HEAD){
            static_assert(TARGET_SPEC::HAS_HEAD, "Source has HEAD but target does not");
            copy(source_device, target_device, source.head, target.head);
        }
    }

    // ======================== abs_diff ========================
    template <typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, nn_models::parallel::ModuleForward<SPEC_A>& a, const nn_models::parallel::ModuleForward<SPEC_B>& b){
        using T = typename SPEC_A::TYPE_POLICY::DEFAULT;
        T result = nn_models::parallel::_abs_diff_branches(device, a, b);
        if constexpr(SPEC_A::HAS_HEAD){
            result += static_cast<T>(abs_diff(device, a.head, b.head));
        }
        return result;
    }

    // ======================== is_nan ========================
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        bool result = nn_models::parallel::_is_nan_branches(device, model, mode);
        if constexpr(SPEC::HAS_HEAD){
            result = result || is_nan(device, model.head, mode);
        }
        return result;
    }

    // ======================== output ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, model.output);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, const nn_models::parallel::ModuleGradient<SPEC>& model){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, model.output);
    }

    // ======================== get_last_layer ========================
    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto& get_last_layer(nn_models::parallel::ModuleForward<SPEC>& model){
        static_assert(SPEC::HAS_HEAD, "get_last_layer on parallel model requires a HEAD module");
        return model.head;
    }
    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto& get_last_layer(const nn_models::parallel::ModuleForward<SPEC>& model){
        static_assert(SPEC::HAS_HEAD, "get_last_layer on parallel model requires a HEAD module");
        return model.head;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
