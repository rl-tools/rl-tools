#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_CUDA_H

#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::gru::helper{
    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    void matrix_multiply_transpose_bias(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
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
    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    void matrix_multiply_transpose_bias_accumulate(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
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
//        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
//            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
//                T acc = get(device, result, j, i) + get(device, bias, i);
//                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
//                    acc += get(device, t1, i, k) * get(device, t2, j, k);
//                }
//                set(device, result, acc, j, i);
//            }
//        }
    }

    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    void matrix_multiply_broadcast_transpose_bias(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
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
        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
            for(TI j=0; j < get<0>(typename SPEC_OUT::SHAPE{}); ++j){
                T acc = get(device, bias, i);
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, i, k) * get(device, t2, k);
                }
                set(device, result, acc, j, i);
            }
        }
    }
}
#endif
