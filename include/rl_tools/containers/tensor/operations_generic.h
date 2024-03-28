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
        delete data(tensor);
    }

    template <typename DEVICE, typename SPEC, typename TI, auto DIM=0, auto SIZE=0>
    auto view_range(DEVICE& device, Tensor<SPEC>& tensor, TI index, tensor::ViewSpec<DIM, SIZE> = {}){
        static_assert(SIZE > 0);
//        using NEW_SHAPE = tensor::Replace<typename SPEC::SHAPE, SIZE, DIM>;
//        using NEW_STRIDE = typename SPEC::STRIDE;
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
//        using NEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, NEW_SHAPE, NEW_STRIDE>;
        Tensor<tensor::spec::view::range::Specification<SPEC, tensor::ViewSpec<DIM, SIZE>>> view;
        data_reference(view) = data(tensor) + offset;
        return view;
    }

    template <typename DEVICE, typename SPEC, typename TI, auto DIM=0>
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

    namespace tensor{
        struct OperationEmptyParameter{};
        template <auto T_OPERATION, typename PARAMETER>
        struct Operation{
            static constexpr auto OPERATION = T_OPERATION;
            PARAMETER parameter;
        };
        template <typename PARAMETER, typename T_ACCUMULATOR_TYPE, typename T_CURRENT_TYPE, auto T_OPERATION>
        struct ReduceOperation{
            using ACCUMULATOR_TYPE = T_ACCUMULATOR_TYPE;
            using CURRENT_TYPE = T_CURRENT_TYPE;
            static constexpr auto OPERATION = T_OPERATION;
            PARAMETER parameter;
            ACCUMULATOR_TYPE initial_value;
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
        namespace unary_reduce_operations{
            namespace impl{
                template <typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE>
                ACCUMULATOR_TYPE sum(const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE current){
                    return accumulator + current;
                }
            }
            template <typename T>
            using Sum = ReduceOperation<OperationEmptyParameter, T, T, impl::sum<OperationEmptyParameter, T, T>>;
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT, auto BINARY_OPERATION, typename OPERATION_PARAMETER>
    inline void binary_operation(DEVICE& device, const tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER>, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
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

    template<typename DEVICE, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE unary_associative_reduce(DEVICE& device, const tensor::ReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION>& op, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            ACCUMULATOR_TYPE accumulator = op.initial_value;
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_t = view(device, t, i);
                accumulator = UNARY_REDUCE_OPERATION(op.parameter, accumulator, unary_associative_reduce(device, op, next_t));
            }
            return accumulator;
        }
        else{
            ACCUMULATOR_TYPE accumulator = op.initial_value;
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T t_value = get(device, t, i);
                accumulator = UNARY_REDUCE_OPERATION(op.parameter, accumulator, t_value);
            }
            return accumulator;
        }
    }

    template<typename DEVICE, typename SPEC>
    typename SPEC::T sum(DEVICE& device, Tensor<SPEC>& t){
        tensor::unary_reduce_operations::Sum<typename SPEC::T> op;
        op.initial_value = 0;
        return unary_associative_reduce(device, op, t);
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

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{}));
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{}));
        static_assert(get<1>(typename SPEC_2::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI row_i=0; row_i < get<0>(typename SPEC_1::SHAPE{}); ++row_i){
            for(TI col_j=0; col_j < get<1>(typename SPEC_2::SHAPE{}); ++col_j){
                T acc = 0;
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, row_i, k) * get(device, t2, k, col_j);
                }
                set(device, result, acc, row_i, col_j);
            }
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void multiply_transpose(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{}));
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{}));
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
                T acc = 0;
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, i, k) * get(device, t2, j, k);
                }
                set(device, result, acc, j, i);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
