#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_GENERIC_H

#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::gru::mode{

        template <typename SPEC, typename MODE, typename MODE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT bool reset_full_batch(const Mode<nn::layers::gru::ResetMode<MODE, MODE_SPEC>>& mode){
            return false;
        }
        template <typename SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT bool reset_full_batch(const Mode<MODE>& mode){
            return false;
        }

        template <typename SPEC, typename MODE, typename MODE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr bool can_reset_sample(const Mode<nn::layers::gru::ResetMode<MODE, MODE_SPEC>>){
            return true;
        }
        template <typename SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr bool can_reset_sample(const Mode<MODE>){
            return false;
        }

        template <typename SPEC, typename DEVICE, typename MODE, typename MODE_SPEC, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool reset_sample(DEVICE& device, const Mode<nn::layers::gru::ResetMode<MODE, MODE_SPEC>>& mode, TI step_i, TI batch_sample_i){
            return get(device, mode.reset_container, (typename MODE_SPEC::TI)step_i, (typename MODE_SPEC::TI)batch_sample_i, (typename MODE_SPEC::TI)0);
        }
        template <typename SPEC, typename DEVICE, typename MODE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool reset_sample(DEVICE& device, const Mode<MODE>& mode, TI step_i, TI batch_sample_i){
            return false;
        }

        template <typename DEVICE, typename MODE, typename MODE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT typename DEVICE::index_t num_resets(DEVICE& device, const Mode<nn::layers::gru::ResetMode<MODE, MODE_SPEC>>& mode){
            return cast_reduce_sum<typename MODE_SPEC::TI>(device, mode.reset_container);
        }
        template <typename DEVICE, typename MODE, bool SHOULD_NEVER=false>
        RL_TOOLS_FUNCTION_PLACEMENT typename DEVICE::index_t num_resets(DEVICE& device, const Mode<MODE>& mode){
            static_assert(SHOULD_NEVER, "This should not be called");
            return 1;
        }
    }
namespace rl_tools::nn::layers::gru::helper{
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    RL_TOOLS_FUNCTION_PLACEMENT void matrix_multiply_transpose_bias(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_transpose_bias");
#endif
        // Y = WX
        // Y^T = X^T W^T
        // W = t1, X^T = t2, Y^T = result
        // Y^T = result = t2 t1^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{})); // INPUT_DIM
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI i=0; i < get<0>(typename SPEC_OUT::SHAPE{}); i++){
            for(TI j=0; j < get<1>(typename SPEC_OUT::SHAPE{}); j++){
                T bias_value = get(device, bias, j);
                set(device, result, bias_value, i, j);
            }
        }
        auto t1_transpose = permute(device, t1, tensor::PermutationSpec<1, 0>{});
        matrix_multiply_accumulate(device, t2, t1_transpose, result);
//        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
//            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
//                T acc = get(device, bias, i);
//                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
//                    acc += get(device, t1, i, k) * get(device, t2, j, k);
//                }
//                set(device, result, acc, j, i);
//            }
//        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    RL_TOOLS_FUNCTION_PLACEMENT void matrix_multiply_transpose_bias_accumulate(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_transpose_bias_accumulate");
#endif
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{})); // INPUT_DIM
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI i=0; i < get<0>(typename SPEC_OUT::SHAPE{}); i++){
            for(TI j=0; j < get<1>(typename SPEC_OUT::SHAPE{}); j++){
                T value = get(device, result, i, j) + get(device, bias, j);
                set(device, result, value, i, j);
            }
        }
        auto t1_transpose = permute(device, t1, tensor::PermutationSpec<1, 0>{});
        matrix_multiply_accumulate(device, t2, t1_transpose, result);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    RL_TOOLS_FUNCTION_PLACEMENT void matrix_multiply_broadcast_transpose_bias(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        // t2_expand = expand_along_cols(t2)
        // result^T = t1 t2_expand + bias
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_broadcast_transpose_bias");
#endif
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 1);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{})); // INPUT_DIM
//        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI i=0; i < get<1>(typename SPEC_OUT::SHAPE{}); ++i){
            for(TI j=0; j < get<0>(typename SPEC_OUT::SHAPE{}); ++j){
                T acc = get(device, bias, i);
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, i, k) * get(device, t2, k);
                }
                set(device, result, acc, j, i);
            }
        }
    }

    namespace unary_operations{
        template <typename DEVICE, typename PARAMETER, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T d_sigmoid(DEVICE& device, const PARAMETER& parameter, T a){
            T s = sigmoid(device, parameter, a);
            return s * (1 - s);
        }
        template <typename DEVICE, typename PARAMETER, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T d_sigmoid_post_activation(DEVICE& device, const PARAMETER& parameter, T s){
            return s * (1 - s);
        }
        template <typename DEVICE, typename PARAMETER, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T d_tanh(DEVICE& device, const PARAMETER& parameter, T a){
            T t = tanh(device, parameter, a);// todo: make version that can take advantage of a stored value so that tanh does not have to be calculated again
            return 1 - t * t;
        }
        template <typename DEVICE, typename PARAMETER, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T d_tanh_post_activation(DEVICE& device, const PARAMETER& parameter, T t){
            return 1 - t * t;
        }
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT void d_sigmoid(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::OperationLegacy<unary_operations::d_sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT void d_sigmoid_post_activation(DEVICE& device, Tensor<SPEC>& t){
        // this function takes sigmoid(x) as an input instead of x (to reuse it in the calculation)
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::OperationLegacy<unary_operations::d_sigmoid_post_activation<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT void d_sigmoid_post_activation(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        // this function takes sigmoid(x) as an input instead of x (to reuse it in the calculation)
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::OperationLegacy<unary_operations::d_sigmoid_post_activation<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT void d_tanh(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::OperationLegacy<unary_operations::d_tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT void d_tanh_post_activation(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        // this function takes tanh(x) as an input instead of x (to reuse it in the calculation)
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::OperationLegacy<unary_operations::d_tanh_post_activation<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }

    template<typename LAYER_SPEC, typename DEVICE, typename INITIAL_HIDDEN_STATE_SPEC, typename OUTPUT_PREVIOUS_STEP_SPEC, typename PREVIOUS_OUTPUT_SCRATCH_SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void select_previous_output(DEVICE& device, const Tensor<INITIAL_HIDDEN_STATE_SPEC>& initial_hidden_state, const Tensor<OUTPUT_PREVIOUS_STEP_SPEC>& output_previous_step, Tensor<PREVIOUS_OUTPUT_SCRATCH_SPEC>& previous_output_scratch, typename DEVICE::index_t step_i, const Mode<MODE>& mode){
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = get<0>(typename PREVIOUS_OUTPUT_SCRATCH_SPEC::SHAPE{});
        for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
            auto target = view(device, previous_output_scratch, sample_i);
            if(nn::layers::gru::mode::reset_sample<LAYER_SPEC>(device, mode, step_i, sample_i)){
                copy(device, device, initial_hidden_state, target);
            }
            else{
                auto source = view(device, output_previous_step, sample_i);
                copy(device, device, source, target);
            }
        }
    }
    template<typename LAYER_SPEC, bool ACCUMULATE_INITIAL_HIDDEN_STATE_GRADIENT, typename DEVICE, typename D_OUTPUT_STEP_SPEC, typename Z_POST_ACTIVATION_SPEC, typename D_OUTPUT_PREVIOUS_STEP_SPEC, typename INITIAL_HIDDEN_STATE_GRADIENT_SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void multiply_accumulate_d_output_previous_step(DEVICE& device, const Tensor<D_OUTPUT_STEP_SPEC>& d_output_step, const Tensor<Z_POST_ACTIVATION_SPEC>& z_post_activation, Tensor<D_OUTPUT_PREVIOUS_STEP_SPEC>& d_output_previous_step, Tensor<INITIAL_HIDDEN_STATE_GRADIENT_SPEC>& initial_hidden_state_gradient, typename DEVICE::index_t step_i, const Mode<MODE>& mode){
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = get<0>(typename D_OUTPUT_STEP_SPEC::SHAPE{});
        for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
            auto d_output_step_sample = view(device, d_output_step, sample_i);
            auto z_post_activation_sample = view(device, z_post_activation, sample_i);
            if(nn::layers::gru::mode::reset_sample<LAYER_SPEC>(device, mode, step_i, sample_i)){
                if constexpr(ACCUMULATE_INITIAL_HIDDEN_STATE_GRADIENT){
                    multiply_accumulate(device, d_output_step_sample, z_post_activation_sample, initial_hidden_state_gradient);
                }
            }
            else{
                auto d_output_previous_step_sample = view(device, d_output_previous_step, sample_i);
                multiply_accumulate(device, d_output_step_sample, z_post_activation_sample, d_output_previous_step_sample);
            }
        }
    }
    namespace detail{
        template<typename LAYER_SPEC, bool RESET_SAMPLES, typename DEVICE, typename SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void zero_samples(DEVICE& device, Tensor<SPEC>& tensor, typename DEVICE::index_t step_i, const Mode<MODE>& mode){
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = get<0>(typename SPEC::SHAPE{});
            for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
                if(nn::layers::gru::mode::reset_sample<LAYER_SPEC>(device, mode, step_i, sample_i) == RESET_SAMPLES){
                    auto sample = view(device, tensor, sample_i);
                    set_all(device, sample, 0);
                }
            }
        }
    }
    template<typename LAYER_SPEC, typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_reset_samples(DEVICE& device, Tensor<SPEC>& tensor, typename DEVICE::index_t step_i, const Mode<MODE>& mode){
        detail::zero_samples<LAYER_SPEC, true>(device, tensor, step_i, mode);
    }
    template<typename LAYER_SPEC, typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_continuing_samples(DEVICE& device, Tensor<SPEC>& tensor, typename DEVICE::index_t step_i, const Mode<MODE>& mode){
        detail::zero_samples<LAYER_SPEC, false>(device, tensor, step_i, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
