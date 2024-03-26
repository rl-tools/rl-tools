#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_GENERIC_H

#include "tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, Tensor<SPEC>& tensor){
        data_reference(tensor) = (typename SPEC::T*) new typename SPEC::T[SPEC::SIZE];
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, Tensor<SPEC>& tensor){
        delete[] data_reference(tensor);
    }

    template <typename DEVICE, typename SPEC, typename TI, auto DIM = 0>
    auto view(DEVICE& device, Tensor<SPEC>& tensor, TI index, tensor::ViewSpec<DIM> = {}){
        using NEW_SHAPE = tensor::Remove<typename SPEC::SHAPE, DIM>;
        using NEW_STRIDE = tensor::Remove<typename SPEC::STRIDE, DIM>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, NEW_SHAPE, NEW_STRIDE>;
        Tensor<NEW_SPEC> view;
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
        data_reference(view) = data(tensor) + offset;
        return view;
    }

    template<typename DEVICE, typename SPEC, typename TII>
    typename DEVICE::index_t index(DEVICE& device, Tensor<SPEC>& tensor, TII index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        return index * get<0>(typename SPEC::STRIDE{});
    }

    template<typename DEVICE, typename SPEC, typename TII, typename... INDICES>
    auto index(DEVICE& device, Tensor<SPEC>& tensor, const TII index, const INDICES... indices){
        using TI = typename DEVICE::index_t;
        auto v = view(device, tensor, index);
        TI current = get<0>(typename SPEC::STRIDE{}) * index;
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            return current + index(device, v);
        }
        else{
            return current + index(device, v, indices...);
        }
    }

    template<typename DEVICE, typename SPEC, typename TII>
    typename SPEC::T get(DEVICE& device, Tensor<SPEC>& tensor, TII local_index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        return *(data(tensor) + index(device, tensor, local_index));
    }

    template<typename DEVICE, typename SPEC, typename TII, typename... INDICES>
    typename SPEC::T get(DEVICE& device, Tensor<SPEC>& tensor, const TII index, const INDICES... indices){
        auto v = view(device, tensor, index);
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            return get(device, v);
        }
        else{
            return get(device, v, indices...);
        }
    }

    template<typename DEVICE, typename SPEC, typename TII> //SFINAE actually not required: typename utils::typing::enable_if_t<length(typename SPEC::SHAPE{})==1>* = nullptr>
    void set(DEVICE& device, Tensor<SPEC>& tensor, typename SPEC::T value, TII current_index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        *(data(tensor) + index(device, tensor, current_index)) = value;
    }

    template<typename DEVICE, typename SPEC, typename TII, typename... INDICES> //, typename utils::typing::enable_if_t<tensor::RANK_LARGER_THAN<typename SPEC::SHAPE, 1>>* = nullptr>
    void set(DEVICE& device, Tensor<SPEC>& tensor, typename SPEC::T value, const TII index, const INDICES... indices){
        auto v = view(device, tensor, index);
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            set(device, v, value);
        }
        else{
            set(device, v, value, indices...);
        }
    }

    template<typename FROM_DEVICE, typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, TO_DEVICE& to_device, Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        using TI = typename FROM_DEVICE::index_t;
        static_assert(tensor::same_dimensions<FROM_SPEC, TO_SPEC>());
        if constexpr(length(typename FROM_SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename FROM_SPEC::SHAPE{}); ++i){
                auto next_from = view(from_device, from, i);
                auto next_to = view(to_device, to, i);
                copy(from_device, to_device, next_from, next_to);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename FROM_SPEC::SHAPE{}); i++){
                set(to_device, to, get(from_device, from, i), i);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void randn(DEVICE& device, Tensor<SPEC>& t, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next = view(device, t, i);
                randn(device, next, rng);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T value = random::normal_distribution::sample(device.random, (T)0, (T)1, rng);
                set(device, t, value, i);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    typename SPEC::T sum(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            T acc = 0;
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next = view(device, t, i);
                acc += sum(device, next);
            }
            return acc;
        }
        else{
            T acc = 0;
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                acc += get(device, t, i);
            }
            return acc;
        }
    }
    namespace tensor{
        struct OperationEmptyParameter{};
        template <auto T_OPERATION, typename PARAMETER>
        struct Operation{
            static constexpr auto OPERATION = T_OPERATION;
            PARAMETER parameter;
        };
        namespace binary_operations{
            template <typename T>
            T add(T a, T b){
                return a + b;
            }
            template <typename T>
            T subtract(T a, T b){
                return a - b;
            }
            template <typename T>
            T multiply(T a, T b){
                return a * b;
            }
            template <typename T>
            T divide(T a, T b){
                return a / b;
            }
        }
        namespace unary_operations{
            template <typename PARAMETER, typename T>
            T negate(const PARAMETER& parameter, T a){
                return -a;
            }
            template <typename MATH_DEVICE, typename PARAMETER, typename T>
            T abs(const PARAMETER& parameter, T a){
                return math::abs(MATH_DEVICE{}, a);
            }
            template <typename PARAMETER, typename T>
            T constant(const PARAMETER& parameter, T a){
                return parameter;
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT, auto BINARY_OPERATION, typename OPERATION_PARAMETER>
    void binary_operation(DEVICE& device, const tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER>, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        using BOP = tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER>;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_OUT>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                auto next_result = view(device, result, i);
                binary_operation(device, BOP{}, next_t1, next_t2, next_result);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T result_value = BINARY_OPERATION(t1_value, t2_value);
                set(device, result, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void subtract(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        binary_operation(device, tensor::Operation<tensor::binary_operations::subtract<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2, result);
    }

    template<typename DEVICE, typename SPEC, auto UNARY_OPERATION, typename OPERATION_PARAMETER>
    void unary_operation(DEVICE& device, const tensor::Operation<UNARY_OPERATION, OPERATION_PARAMETER>& op, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_t = view(device, t, i);
                unary_operation(device, op, next_t);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T t_value = get(device, t, i);
                T result_value = UNARY_OPERATION(op.parameter, t_value);
                set(device, t, result_value, i);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    void abs(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        using PARAMETER = tensor::OperationEmptyParameter;
        tensor::Operation<tensor::unary_operations::abs<typename DEVICE::SPEC::MATH, PARAMETER, T>, PARAMETER> op;
        unary_operation(device, op, t);
    }

    template<typename DEVICE, typename SPEC>
    void set_all(DEVICE& device, Tensor<SPEC>& t, typename SPEC::T value){
        using T = typename SPEC::T;
        using PARAMETER = T;
        tensor::Operation<tensor::unary_operations::constant<PARAMETER, T>, PARAMETER> op;
        op.parameter = value;
        unary_operation(device, op, t);
    }



}

#endif
