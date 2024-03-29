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
        delete[] data(tensor);
    }

    template <typename DEVICE, typename SPEC, auto DIM=0, auto SIZE=0>
    auto view_range(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t index, const tensor::ViewSpec<DIM, SIZE>){
        static_assert(SIZE > 0);
        static_assert(get<DIM>(typename SPEC::SHAPE{}) >= SIZE);
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, offset < SPEC::SIZE, "Index out of bounds");
        utils::assert_exit(device, offset + SIZE <= SPEC::SIZE, "Index out of bounds");
#endif
        Tensor<tensor::spec::view::range::Specification<SPEC, tensor::ViewSpec<DIM, SIZE>>> view;
        data_reference(view) = data(tensor) + offset;
        return view;
    }

    template <typename DEVICE, typename SPEC, auto DIM=0>
    auto view(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t index, const tensor::ViewSpec<DIM> = {}){
        using NEW_SHAPE = tensor::Remove<typename SPEC::SHAPE, DIM>;
        using NEW_STRIDE = tensor::Remove<typename SPEC::STRIDE, DIM>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, NEW_SHAPE, NEW_STRIDE>;
        Tensor<NEW_SPEC> view;
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
        data_reference(view) = data(tensor) + offset;
        return view;
    }

    template<typename DEVICE, typename SPEC, typename TII>
    typename DEVICE::index_t index(DEVICE& device, const Tensor<SPEC>& tensor, TII index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        return index * get<0>(typename SPEC::STRIDE{});
    }

    template<typename DEVICE, typename SPEC, typename... INDICES>
    auto index(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t index, const INDICES... indices){
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

    template<typename DEVICE, typename SPEC>
    typename SPEC::T get(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t local_index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        auto idx = index(device, tensor, local_index);
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, idx < SPEC::SIZE, "Index out of bounds");
#endif
        return *(data(tensor) + idx);
    }

    template<typename DEVICE, typename SPEC, typename... INDICES>
    typename SPEC::T get(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t index, const INDICES... indices){
        auto v = view(device, tensor, index);
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            return get(device, v, index);
        }
        else{
            return get(device, v, indices...);
        }
    }

    template<typename DEVICE, typename SPEC, typename TII> //SFINAE actually not required: typename utils::typing::enable_if_t<length(typename SPEC::SHAPE{})==1>* = nullptr>
    void set(DEVICE& device, Tensor<SPEC>& tensor, typename SPEC::T value, TII current_index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        auto idx = index(device, tensor, current_index);
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, idx < SPEC::SIZE, "Index out of bounds");
#endif
        *(data(tensor) + idx) = value;
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
            template <typename DEVICE, typename PARAMETER, typename T>
            T negate(DEVICE& device, const PARAMETER& parameter, T a){
                return -a;
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T abs(DEVICE& device, const PARAMETER& parameter, T a){
                return math::abs(device.math, a);
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T constant(DEVICE& device, const PARAMETER& parameter, T a){
                return parameter;
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T sigmoid(DEVICE& device, const PARAMETER& parameter, T a){
                return 1 / (1 + math::exp(device.math, a));
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T tanh(DEVICE& device, const PARAMETER& parameter, T a){
                return math::tanh(device.math, a);
            }
        }
        template <typename PARAMETER, typename T_ACCUMULATOR_TYPE, typename T_CURRENT_TYPE, auto T_OPERATION>
        struct UnaryReduceOperation{
            using ACCUMULATOR_TYPE = T_ACCUMULATOR_TYPE;
            using CURRENT_TYPE = T_CURRENT_TYPE;
            static constexpr auto OPERATION = T_OPERATION;
            PARAMETER parameter;
            ACCUMULATOR_TYPE initial_value;
        };
        namespace unary_reduce_operations{
            namespace impl{
                template <typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE>
                ACCUMULATOR_TYPE sum(const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE current){
                    return accumulator + current;
                }
            }
            template <typename T>
            using Sum = UnaryReduceOperation<OperationEmptyParameter, T, T, impl::sum<OperationEmptyParameter, T, T>>;
        }
        template <typename PARAMETER, typename T_ACCUMULATOR_TYPE, typename T_CURRENT_TYPE1, typename T_CURRENT_TYPE2, auto T_OPERATION, auto T_REDUCE_OPERATION>
        struct BinaryReduceOperation{
            using ACCUMULATOR_TYPE = T_ACCUMULATOR_TYPE;
            using CURRENT_TYPE1 = T_CURRENT_TYPE1;
            using CURRENT_TYPE2 = T_CURRENT_TYPE2;
            static constexpr auto OPERATION = T_OPERATION;
            static constexpr auto REDUCE_OPERATION = T_REDUCE_OPERATION; // the associative part
            PARAMETER parameter;
            ACCUMULATOR_TYPE initial_value;
        };
        namespace binary_reduce_operations{
            namespace impl{
                template <typename DEVICE, typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE1, typename CURRENT_TYPE2>
                ACCUMULATOR_TYPE absolute_difference(DEVICE& device, const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE1 current1, CURRENT_TYPE2 current2){
                    return accumulator + math::abs(device.math, current1 - current2);
                }
                template <typename DEVICE, typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE>
                ACCUMULATOR_TYPE sum(DEVICE& device, const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE current){
                    return accumulator + current;
                }
            }
            template <typename DEVICE, typename T1, typename T2>
            using AbsoluteDifference = BinaryReduceOperation<OperationEmptyParameter, T1, T1, T2, impl::absolute_difference<DEVICE, OperationEmptyParameter, T1, T1, T2>, impl::sum<DEVICE, OperationEmptyParameter, T1, T1>>;
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
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_OPERATION, typename OPERATION_PARAMETER>
    inline void binary_operation(DEVICE& device, const tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER>, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        using BOP = tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER>;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                binary_operation(device, BOP{}, next_t1, next_t2);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T result_value = BINARY_OPERATION(t1_value, t2_value);
                set(device, t1, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        binary_operation(device, tensor::Operation<tensor::binary_operations::subtract<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void subtract(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        binary_operation(device, tensor::Operation<tensor::binary_operations::subtract<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2, result);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        binary_operation(device, tensor::Operation<tensor::binary_operations::multiply<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2);
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
                T result_value = UNARY_OPERATION(device, op.parameter, t_value);
                set(device, t, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    void sigmoid(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC>
    void tanh(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }

    template<typename DEVICE, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE unary_associative_reduce(DEVICE& device, const tensor::UnaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION>& op, Tensor<SPEC>& t){
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

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_REDUCE_OPERATION, auto BINARY_ASSOCIATIVE_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE1, typename CURRENT_TYPE2, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE binary_associative_reduce(DEVICE& device, const tensor::BinaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE1, CURRENT_TYPE2, BINARY_REDUCE_OPERATION, BINARY_ASSOCIATIVE_REDUCE_OPERATION>& op, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            ACCUMULATOR_TYPE accumulator = op.initial_value;
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                accumulator = BINARY_ASSOCIATIVE_REDUCE_OPERATION(device, op.parameter, accumulator, binary_associative_reduce(device, op, next_t1, next_t2));
            }
            return accumulator;
        }
        else{
            ACCUMULATOR_TYPE accumulator = op.initial_value;
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                accumulator = BINARY_REDUCE_OPERATION(device, op.parameter, accumulator, t1_value, t2_value);
            }
            return accumulator;
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T absolute_difference(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        tensor::binary_reduce_operations::AbsoluteDifference<DEVICE, typename SPEC_1::T, typename SPEC_2::T> op;
        op.initial_value = 0;
        return binary_associative_reduce(device, op, t1, t2);
    }


    template<typename DEVICE, typename SPEC>
    void abs(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        using PARAMETER = tensor::OperationEmptyParameter;
        tensor::Operation<tensor::unary_operations::abs<DEVICE, PARAMETER, T>, PARAMETER> op;
        unary_operation(device, op, t);
    }

    template<typename DEVICE, typename SPEC>
    void set_all(DEVICE& device, Tensor<SPEC>& t, typename SPEC::T value){
        using T = typename SPEC::T;
        using PARAMETER = T;
        tensor::Operation<tensor::unary_operations::constant<DEVICE, PARAMETER, T>, PARAMETER> op;
        op.parameter = value;
        unary_operation(device, op, t);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void matrix_multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
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

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
    void matrix_multiply_transpose_bias(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
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
        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
                T acc = get(device, bias, i);
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
