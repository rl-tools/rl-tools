#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_GENERIC_H

#include "tensor.h"
#include "../../mode/mode.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename T, typename T_TI, T_TI SIZE>
    void malloc(DEVICE& device, tensor::TensorStatic<T, T_TI, SIZE>& tensor) {
        // no-op
    }
    template<typename DEVICE, typename T, typename T_TI, T_TI SIZE>
    void free(DEVICE& device, tensor::TensorStatic<T, T_TI, SIZE>& tensor) {
        // no-op
    }
    template<typename DEVICE, typename T>
    void malloc(DEVICE& device, tensor::TensorStaticEmpty<T>& tensor) {
        // no-op
    }
    template<typename DEVICE, typename T>
    void free(DEVICE& device, tensor::TensorStaticEmpty<T>& tensor) {
        // no-op
    }

#if !defined(RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS)
    template<typename DEVICE, typename T, typename T_TI, T_TI SIZE, bool CONST>
    void malloc(DEVICE& device, tensor::TensorDynamic<T, T_TI, SIZE, CONST>& tensor){
        T* temp = (T*) new T[SIZE];
        *data_pointer(tensor) = temp;
#if RL_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename DEVICE::index_t i=0; i < SIZE; i++){
            data(tensor)[i] = math::nan<T>(device.math);
        }
#endif
    }
    template <typename DEVICE, typename T, typename T_TI, T_TI SIZE, bool CONST>
    void free(DEVICE& device, tensor::TensorDynamic<T, T_TI, SIZE, CONST>& tensor){
        delete[] data(tensor);
    }
#endif

    template <typename SHAPE, typename DEVICE, typename SPEC>
    auto view_memory(DEVICE& device, const Tensor<SPEC>& tensor){
        static_assert(product(SHAPE{}) <= SPEC::SIZE);
        static_assert(tensor::dense_row_major_layout<SPEC, true>());
        using VIEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, SHAPE, true, typename SPEC::STRIDE, true>; // note the last boolean signals constness and needs to be flipped for the non-const version of this function
        using VIEW_TYPE = Tensor<VIEW_SPEC>;
        const VIEW_TYPE view{data(tensor)};
        return view;
    }

    template <typename SHAPE, typename DEVICE, typename SPEC>
    auto view_memory(DEVICE& device, Tensor<SPEC>& tensor){
        static_assert(product(SHAPE{}) <= SPEC::SIZE);
        static_assert(tensor::dense_row_major_layout<SPEC, true>());
        using DENSE_STRIDE = tensor::RowMajorStride<SHAPE>;
        using STRIDE = tensor::Append<tensor::PopBack<DENSE_STRIDE>, get<length(typename SPEC::STRIDE{}) - 1>(typename SPEC::STRIDE{})>; // the RELAX_MAJOR in dense_row_major_layout allows for a stride in the last element which is accounted for here;
        using VIEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, SHAPE, true, STRIDE, false>;
        using VIEW_TYPE = Tensor<VIEW_SPEC>;
        VIEW_TYPE view{data(tensor)};
        return view;
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
        using VIEW_TYPE = Tensor<tensor::spec::view::range::Specification<SPEC, tensor::ViewSpec<DIM, SIZE>, true>>;
        using VIEW_TYPE_CV = const VIEW_TYPE;

        VIEW_TYPE_CV view{data(tensor) + offset};
        return view;
    }

    template <typename DEVICE, typename SPEC, auto DIM=0, auto SIZE=0>
    auto view_range(DEVICE& device, Tensor<SPEC>& tensor, typename DEVICE::index_t index, const tensor::ViewSpec<DIM, SIZE>){
        static_assert(SIZE > 0);
        static_assert(get<DIM>(typename SPEC::SHAPE{}) >= SIZE);
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, offset < SPEC::SIZE, "Index out of bounds");
        utils::assert_exit(device, offset + SIZE <= SPEC::SIZE, "Index out of bounds");
#endif
        using VIEW_TYPE = Tensor<tensor::spec::view::range::Specification<SPEC, tensor::ViewSpec<DIM, SIZE>, false>>;
        VIEW_TYPE view{data(tensor) + offset};
        return view;
    }
    template <typename DEVICE, typename SPEC, auto DIM=0, auto SIZE=0>
    auto view_range(DEVICE& device, const Tensor<SPEC>& tensor, const tensor::ViewSpec<DIM, SIZE>){
        return view_range(device, tensor, 0, tensor::ViewSpec<DIM, SIZE>{});
    }

    template <typename DEVICE, typename SPEC, auto DIM=0, auto SIZE=0>
    auto view_range(DEVICE& device, Tensor<SPEC>& tensor, const tensor::ViewSpec<DIM, SIZE>){
        return view_range(device, tensor, 0, tensor::ViewSpec<DIM, SIZE>{});
    }

    template <auto DIM=0, typename DEVICE, typename SPEC>
    auto view(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t index, const tensor::ViewSpec<DIM> = {}){
        using NEW_SHAPE = tensor::Remove<typename SPEC::SHAPE, DIM>;
        using NEW_STRIDE = tensor::Remove<typename SPEC::STRIDE, DIM>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, NEW_SHAPE, true, NEW_STRIDE, true>;
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
//        data_reference(view) = ;
        const Tensor<NEW_SPEC> view{data(tensor) + offset};
        return view;
    }

    template <auto DIM=0, typename DEVICE, typename SPEC>
    auto view(DEVICE& device, Tensor<SPEC>& tensor, typename DEVICE::index_t index, const tensor::ViewSpec<DIM> = {}){
        using NEW_SHAPE = tensor::Remove<typename SPEC::SHAPE, DIM>;
        using NEW_STRIDE = tensor::Remove<typename SPEC::STRIDE, DIM>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, NEW_SHAPE, true, NEW_STRIDE, false>;
        auto offset = index * get<DIM>(typename SPEC::STRIDE{});
//        data_reference(view) = ;
        Tensor<NEW_SPEC> view{data(tensor) + offset};
        return view;
    }

    namespace tensor{
        template <auto T_DIM_1, auto T_DIM_2>
        struct PermutationSpec{
            static constexpr auto DIM_1 = T_DIM_1;
            static constexpr auto DIM_2 = T_DIM_2;
        };
    }

    template <typename DEVICE, typename SPEC, auto DIM_1=0, auto DIM_2=1>
    auto constexpr permute(DEVICE& device, const Tensor<SPEC>& tensor, const tensor::PermutationSpec<DIM_1, DIM_2> spec={}){
        static_assert(length(typename SPEC::SHAPE{}) >= 2);
        static_assert(DIM_1 < length(typename SPEC::SHAPE{}));
        static_assert(DIM_2 < length(typename SPEC::SHAPE{}));
        using SHAPE = typename SPEC::SHAPE;
        using TI = typename SHAPE::TI;
        using NEW_SHAPE_INTERMEDIATE = tensor::Replace<SHAPE, get<DIM_2>(SHAPE{}), DIM_1>;
        using NEW_SHAPE = tensor::Replace<NEW_SHAPE_INTERMEDIATE, get<DIM_1>(SHAPE{}), DIM_2>;
        using STRIDE = typename SPEC::STRIDE;
        using NEW_STRIDE_INTERMEDIATE = tensor::Replace<STRIDE, get<DIM_2>(STRIDE{}), DIM_1>;
        using NEW_STRIDE = tensor::Replace<NEW_STRIDE_INTERMEDIATE, get<DIM_1>(STRIDE{}), DIM_2>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, TI, NEW_SHAPE, true, NEW_STRIDE, true>; // const here
        const Tensor<NEW_SPEC> view{data(tensor)};
        return view;
    }
    template <typename DEVICE, typename SPEC, auto DIM_1=0, auto DIM_2=1>
    auto constexpr permute(DEVICE& device, Tensor<SPEC>& tensor, const tensor::PermutationSpec<DIM_1, DIM_2> spec={}){
        static_assert(length(typename SPEC::SHAPE{}) >= 2);
        static_assert(DIM_1 < length(typename SPEC::SHAPE{}));
        static_assert(DIM_2 < length(typename SPEC::SHAPE{}));
        using SHAPE = typename SPEC::SHAPE;
        using TI = typename SHAPE::TI;
        using NEW_SHAPE_INTERMEDIATE = tensor::Replace<SHAPE, get<DIM_2>(SHAPE{}), DIM_1>;
        using NEW_SHAPE = tensor::Replace<NEW_SHAPE_INTERMEDIATE, get<DIM_1>(SHAPE{}), DIM_2>;
        using STRIDE = typename SPEC::STRIDE;
        using NEW_STRIDE_INTERMEDIATE = tensor::Replace<STRIDE, get<DIM_2>(STRIDE{}), DIM_1>;
        using NEW_STRIDE = tensor::Replace<NEW_STRIDE_INTERMEDIATE, get<DIM_1>(STRIDE{}), DIM_2>;
        using NEW_SPEC = tensor::Specification<typename SPEC::T, TI, NEW_SHAPE, true, NEW_STRIDE, false>; // non-const here
        Tensor<NEW_SPEC> view;
        *data_pointer(view) = data(tensor);
        return view;
    }

    template<typename DEVICE, typename SPEC, typename TII>
    typename DEVICE::index_t index(DEVICE& device, const Tensor<SPEC>& tensor, TII index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        return static_cast<typename DEVICE::index_t>(index) * get<0>(typename SPEC::STRIDE{});
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

    template<typename DEVICE, typename SPEC>
    typename SPEC::T get_flat(DEVICE& device, const Tensor<SPEC>& tensor, typename DEVICE::index_t local_index){
        static_assert(tensor::dense_row_major_layout<SPEC>());
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, local_index < SPEC::SIZE, "Index out of bounds");
#endif
        return *(data(tensor) + local_index);
    }

    template<typename DEVICE, typename SPEC, typename TII> //SFINAE actually not required: typename utils::typing::enable_if_t<length(typename SPEC::SHAPE{})==1>* = nullptr>
    void set(DEVICE& device, Tensor<SPEC>& tensor, typename SPEC::T value, TII current_index){
        static_assert(length(typename SPEC::SHAPE{})==1);
        auto idx = index(device, tensor, static_cast<typename DEVICE::index_t>(current_index));
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, idx < SPEC::SIZE, "Index out of bounds");
#endif
        *(data(tensor) + idx) = value;
    }

    template<typename DEVICE, typename SPEC, typename TII, typename... INDICES> //, typename utils::typing::enable_if_t<tensor::RANK_LARGER_THAN<typename SPEC::SHAPE, 1>>* = nullptr>
    void set(DEVICE& device, Tensor<SPEC>& tensor, typename SPEC::T value, const TII index, const INDICES... indices){
        auto v = view(device, tensor, static_cast<typename DEVICE::index_t>(index));
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            set(device, v, value);
        }
        else{
            set(device, v, value, indices...);
        }
    }

    template<typename FROM_DEVICE, typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
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
    template<typename DEVICE, typename SPEC, typename RNG>
    void rand(DEVICE& device, Tensor<SPEC>& t, RNG& rng, typename SPEC::T min=0, typename SPEC::T max=1){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next = view(device, t, i);
                rand(device, next, rng, min, max);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T value = random::uniform_real_distribution(device.random, (T)min, (T)max, rng);
                set(device, t, value, i);
            }
        }
    }

    namespace tensor{
        struct OperationEmptyParameter{};
        template <auto T_OPERATION, typename PARAMETER=OperationEmptyParameter>
        struct Operation{
            static constexpr auto OPERATION = T_OPERATION;
            PARAMETER parameter;
        };
        namespace binary_operations{
            template <typename T>
            T add(T a, T b, const OperationEmptyParameter){
                return a + b;
            }
            template <typename T>
            T subtract(T a, T b, const OperationEmptyParameter){
                return a - b;
            }
            template <typename T>
            T multiply(T a, T b, const OperationEmptyParameter){
                return a * b;
            }
            template <typename T>
            T divide(T a, T b, const OperationEmptyParameter){
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
                return 1 / (1 + math::exp(device.math, -a));
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T fast_sigmoid(DEVICE& device, const PARAMETER& parameter, T a){
                return math::fast_sigmoid(device.math, a);
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T exp(DEVICE& device, const PARAMETER& parameter, T a){
                return math::exp(device.math, a);
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T tanh(DEVICE& device, const PARAMETER& parameter, T a){
                return math::tanh(device.math, a);
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T fast_tanh(DEVICE& device, const PARAMETER& parameter, T a){
                return math::fast_tanh(device.math, a);
            }
            template <typename DEVICE, typename PARAMETER, typename T>
            T one_minus(DEVICE& device, const PARAMETER& parameter, T a){
                return 1 - a;
            }
            template <typename T>
            struct ScaleOperationParameters{
                T scale;
            };
            template <typename DEVICE, typename T>
            T scale(DEVICE& device, const ScaleOperationParameters<T>& parameter, T a){
                return a * parameter.scale;
            }
        }
        namespace ternary_operations{
            template <typename T>
            T multiply_accumulate(T a, T b, T acc){
                return acc + a * b;
            }
        }
        template <typename PARAMETER, typename T_ACCUMULATOR_TYPE, typename T_CURRENT_TYPE, auto T_UNARY_REDUCE_OPERATION>
        struct UnaryReduceOperation{
            using ACCUMULATOR_TYPE = T_ACCUMULATOR_TYPE;
            using CURRENT_TYPE = T_CURRENT_TYPE;
            static constexpr auto UNARY_REDUCE_OPERATION = T_UNARY_REDUCE_OPERATION;
            PARAMETER parameter;
            ACCUMULATOR_TYPE initial_value;
        };
        namespace unary_reduce_operations{
            namespace impl{
                template <typename DEVICE, typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE>
                ACCUMULATOR_TYPE sum(DEVICE, const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE current){
                    return accumulator + current;
                }
                template <typename DEVICE, typename PARAMETER, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE>
                ACCUMULATOR_TYPE squared_sum(DEVICE, const PARAMETER& parameter, const ACCUMULATOR_TYPE& accumulator, CURRENT_TYPE current){
                    return accumulator + current * current;
                }
                template <typename DEVICE, typename PARAMETER, typename CURRENT_TYPE>
                CURRENT_TYPE sum_reduce(DEVICE, const PARAMETER& parameter, CURRENT_TYPE a, CURRENT_TYPE b){
                    return a + b;
                }
                template <typename DEVICE, typename PARAMETER, typename CURRENT_TYPE>
                bool is_nan(DEVICE& device, const PARAMETER& parameter, const bool& accumulator, CURRENT_TYPE current){
                    return accumulator || math::is_nan(device, current);
                }
                template <typename DEVICE, typename PARAMETER, typename CURRENT_TYPE>
                bool is_finite(DEVICE& device, const PARAMETER& parameter, const bool& accumulator, CURRENT_TYPE current){
                    return accumulator || math::is_finite(device, current);
                }
                template <typename DEVICE, typename PARAMETER>
                bool is_nan_reduce(DEVICE& device, const PARAMETER& parameter, bool a, bool b){
                    return a || b;
                }
            }
            template <typename DEVICE, typename T>
            using Sum = UnaryReduceOperation<OperationEmptyParameter, T, T, impl::sum<DEVICE, OperationEmptyParameter, T, T>>;
            template <typename DEVICE, typename T>
            using SquaredSum = UnaryReduceOperation<OperationEmptyParameter, T, T, impl::squared_sum<DEVICE, OperationEmptyParameter, T, T>>;
            template <typename TARGET_TYPE, typename DEVICE, typename T>
            using CastSum = UnaryReduceOperation<OperationEmptyParameter, T, T, impl::sum<DEVICE, OperationEmptyParameter, TARGET_TYPE, TARGET_TYPE>>;
            template <typename DEVICE, typename T>
            using IsNan = UnaryReduceOperation<OperationEmptyParameter, bool, T, impl::is_nan<DEVICE, OperationEmptyParameter, T>>;
            template <typename DEVICE, typename T>
            using IsFinite = UnaryReduceOperation<OperationEmptyParameter, bool, T, impl::is_finite<DEVICE, OperationEmptyParameter, T>>;
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
    inline void binary_operation(DEVICE& device, const tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER> param, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_OUT>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                auto next_result = view(device, result, i);
                binary_operation(device, param, next_t1, next_t2, next_result);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T result_value = BINARY_OPERATION(t1_value, t2_value, param.parameter);
                set(device, result, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_OPERATION, typename OPERATION_PARAMETER>
    inline void binary_operation(DEVICE& device, const tensor::Operation<BINARY_OPERATION, OPERATION_PARAMETER> params, const Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                binary_operation(device, params, next_t1, next_t2);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T result_value = BINARY_OPERATION(t1_value, t2_value, params.parameter);
                set(device, t2, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        binary_operation(device, tensor::Operation<tensor::binary_operations::add<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void subtract(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        binary_operation(device, tensor::Operation<tensor::binary_operations::subtract<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2, result);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2){
        binary_operation(device, tensor::Operation<tensor::binary_operations::multiply<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2);
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
    void multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
        binary_operation(device, tensor::Operation<tensor::binary_operations::multiply<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2, t_output);
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
    template<typename DEVICE, typename SPEC, auto UNARY_OPERATION, typename OPERATION_PARAMETER, typename SPEC_OUTPUT>
    void unary_operation(DEVICE& device, const tensor::Operation<UNARY_OPERATION, OPERATION_PARAMETER>& op, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_t = view(device, t, i);
                auto next_output = view(device, output, i);
                unary_operation(device, op, next_t, next_output);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T t_value = get(device, t, i);
                T result_value = UNARY_OPERATION(device, op.parameter, t_value);
                set(device, output, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    void exp(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::exp<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void exp(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::exp<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC>
    void sigmoid(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void sigmoid(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC>
    void fast_sigmoid(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::fast_sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void fast_sigmoid(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::fast_sigmoid<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC>
    void tanh(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void tanh(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC>
    void fast_tanh(DEVICE& device, Tensor<SPEC>& t){
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::fast_tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void fast_tanh(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        unary_operation(device, tensor::Operation<tensor::unary_operations::fast_tanh<DEVICE, tensor::OperationEmptyParameter, T>, tensor::OperationEmptyParameter>{}, t, output);
    }
    template<typename DEVICE, typename SPEC>
    void scale(DEVICE& device, Tensor<SPEC>& t, typename SPEC::T scale){
        using T = typename SPEC::T;
        using PARAMETER_TYPE = tensor::unary_operations::ScaleOperationParameters<T>;
        tensor::Operation<tensor::unary_operations::scale<DEVICE, T>, PARAMETER_TYPE> operation;
        operation.parameter.scale = scale;
        unary_operation(device, operation, t);
    }
    template<typename DEVICE, typename SPEC, typename SPEC_OUTPUT>
    void scale(DEVICE& device, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        static_assert(tensor::same_dimensions<SPEC, SPEC_OUTPUT>());
        using T = typename SPEC::T;
        using PARAMETER_TYPE = tensor::unary_operations::ScaleOperationParameters<T>;
        tensor::Operation<tensor::unary_operations::scale<DEVICE, PARAMETER_TYPE, T>, PARAMETER_TYPE> operation;
        operation.parameter.scale = scale;
        unary_operation(device, operation, t, output);
    }

    template<typename DEVICE, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE _unary_associative_reduce(DEVICE& device, const tensor::UnaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION>& op, const Tensor<SPEC>& t, ACCUMULATOR_TYPE accumulator){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_t = view(device, t, i);
                accumulator = _unary_associative_reduce(device, op, next_t, accumulator);
            }
            return accumulator;
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                T t_value = get(device, t, i);
                accumulator = UNARY_REDUCE_OPERATION(device.math, op.parameter, accumulator, static_cast<CURRENT_TYPE>(t_value));
            }
            return accumulator;
        }
    }
    template<typename DEVICE, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE unary_associative_reduce(DEVICE& device, const tensor::UnaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION>& op, const Tensor<SPEC>& t){
        return _unary_associative_reduce(device, op, t, op.initial_value);
    }

    template<typename DEVICE, typename SPEC>
    typename SPEC::T sum(DEVICE& device, Tensor<SPEC>& t){
        static_assert(!utils::typing::is_same_v<typename SPEC::T, bool>, "Sum would work like or for boolean tensors");
        tensor::unary_reduce_operations::Sum<decltype(device.math), typename SPEC::T> op;
        op.initial_value = 0;
        return unary_associative_reduce(device, op, t);
    }
    template<typename DEVICE, typename SPEC>
    typename SPEC::T squared_sum(DEVICE& device, const Tensor<SPEC>& t){
        static_assert(!utils::typing::is_same_v<typename SPEC::T, bool>, "Sum would work like or for boolean tensors");
        tensor::unary_reduce_operations::SquaredSum<decltype(device.math), typename SPEC::T> op;
        op.initial_value = 0;
        return unary_associative_reduce(device, op, t);
    }
    template<typename TARGET_TYPE, typename DEVICE, typename SPEC>
    TARGET_TYPE cast_sum(DEVICE& device, Tensor<SPEC>& t){
        tensor::unary_reduce_operations::CastSum<TARGET_TYPE, decltype(device.math), TARGET_TYPE> op;
        op.initial_value = 0;
        return unary_associative_reduce(device, op, t);
    }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    typename SPEC::T is_nan(DEVICE& device, const Tensor<SPEC>& t, const Mode<MODE>& mode = {}){
        tensor::unary_reduce_operations::IsNan<decltype(device.math), typename SPEC::T> op;
        op.initial_value = false;
        return unary_associative_reduce(device, op, t);
    }
    template<typename DEVICE, typename SPEC>
    typename SPEC::T is_finite(DEVICE& device, const Tensor<SPEC>& t){
        tensor::unary_reduce_operations::IsFinite<decltype(device.math), typename SPEC::T> op;
        op.initial_value = false;
        return unary_associative_reduce(device, op, t);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT, auto TERNARY_OPERATION, typename OPERATION_PARAMETER>
    inline void ternary_operation(DEVICE& device, const tensor::Operation<TERNARY_OPERATION, OPERATION_PARAMETER>, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        using BOP = tensor::Operation<TERNARY_OPERATION, OPERATION_PARAMETER>;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_OUT>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                auto next_result = view(device, result, i);
                ternary_operation(device, BOP{}, next_t1, next_t2, next_result);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T t3_value = get(device, result, i);
                T result_value = TERNARY_OPERATION(t1_value, t2_value, t3_value);
                set(device, result, result_value, i);
            }
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3, typename SPEC_OUT, auto TERNARY_OPERATION, typename OPERATION_PARAMETER>
    inline void ternary_operation(DEVICE& device, const tensor::Operation<TERNARY_OPERATION, OPERATION_PARAMETER>, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_3>& t3, Tensor<SPEC_OUT>& result){
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        using BOP = tensor::Operation<TERNARY_OPERATION, OPERATION_PARAMETER>;
        static_assert(tensor::same_dimensions<SPEC_1, SPEC_2>());
        static_assert(tensor::same_dimensions<SPEC_2, SPEC_3>());
        static_assert(tensor::same_dimensions<SPEC_3, SPEC_OUT>());
        if constexpr(length(typename SPEC_1::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
                auto next_t1 = view(device, t1, i);
                auto next_t2 = view(device, t2, i);
                auto next_t3 = view(device, t3, i);
                auto next_result = view(device, result, i);
                ternary_operation(device, BOP{}, next_t1, next_t2, next_t3, next_result);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); i++){
                T t1_value = get(device, t1, i);
                T t2_value = get(device, t2, i);
                T t3_value = get(device, t3, i);
                T result_value = TERNARY_OPERATION(t1_value, t2_value, t3_value);
                set(device, result, result_value, i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
    void multiply_accumulate(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("tensor::multiply_accumulate");
#endif
        ternary_operation(device, tensor::Operation<tensor::ternary_operations::multiply_accumulate<typename SPEC_1::T>, tensor::OperationEmptyParameter>{}, t1, t2, t_output);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_REDUCE_OPERATION, auto BINARY_ASSOCIATIVE_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE1, typename CURRENT_TYPE2, typename OPERATION_PARAMETER>
    ACCUMULATOR_TYPE binary_associative_reduce(DEVICE& device, const tensor::BinaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE1, CURRENT_TYPE2, BINARY_REDUCE_OPERATION, BINARY_ASSOCIATIVE_REDUCE_OPERATION>& op, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2){
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
    typename SPEC_1::T abs_diff(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2){
        tensor::binary_reduce_operations::AbsoluteDifference<DEVICE, typename SPEC_1::T, typename SPEC_2::T> op{};
        op.initial_value = 0;
        return binary_associative_reduce(device, op, t1, t2);
    }

    template <bool ACCUMULATE, typename DEVICE, typename SPEC, typename OUTPUT_SPEC, auto SIZE=0, auto DIM=length(typename SPEC::SHAPE{})-1>
    auto reduce_sum(DEVICE& device, Tensor<SPEC>& input, Tensor<OUTPUT_SPEC>& output, tensor::ViewSpec<DIM, SIZE> = tensor::ViewSpec<length(typename SPEC::SHAPE{})-1, SIZE>{}){
        // reduces along the last dimension by default
        static_assert(DIM == length(typename SPEC::SHAPE{}) - 1); // only supporting the last dimension for now
        using EXPECTED_OUTPUT_SHAPE = tensor::Remove<typename SPEC::SHAPE, DIM>;
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        static_assert(tensor::same_dimensions_shape<typename OUTPUT_SPEC::SHAPE, EXPECTED_OUTPUT_SHAPE>());
        if constexpr(length(typename SPEC::SHAPE{}) == 2){
            for(TI row_i=0; row_i < get<0>(typename SPEC::SHAPE{}); ++row_i){
                auto input_row = view(device, input, row_i);

                T aggregate = sum(device, input_row);
                if constexpr(ACCUMULATE){
                    aggregate += get(device, output, row_i);
                }
                set(device, output, aggregate, row_i);
            }
        }
        else{
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_input = view(device, input, i);
                auto next_output = view(device, output, i);
                reduce_sum(device, next_input, next_output);
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename OUTPUT_SPEC, auto SIZE=0, auto DIM=length(typename SPEC::SHAPE{})-1>
    auto reduce_sum(DEVICE& device, Tensor<SPEC>& input, Tensor<OUTPUT_SPEC>& output, tensor::ViewSpec<DIM, SIZE> = tensor::ViewSpec<length(typename SPEC::SHAPE{})-1, SIZE>{}){
        reduce_sum<false>(device, input, output);
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


#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
    void matrix_multiply(DEVICE& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
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
    void matrix_multiply_accumulate(DEVICE& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
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
                T acc = get(device, result, row_i, col_j);
                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
                    acc += get(device, t1, row_i, k) * get(device, t2, k, col_j);
                }
                set(device, result, acc, row_i, col_j);
            }
        }
    }
#endif
//    template<typename DEVICE, typename SPEC, typename = utils::typing::enable_if_t<length(typename SPEC::SHAPE{}) == 2, typename SPEC::TI>>
//    auto matrix_view(DEVICE& device, Tensor<SPEC>& t){
//        static_assert(tensor::generalized_row_major<typename SPEC::SHAPE, typename SPEC::STRIDE>());
//        using LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, get<0>(typename SPEC::STRIDE{}), get<1>(typename SPEC::STRIDE{})>;
//        MatrixDynamic<matrix::Specification<typename SPEC::T, typename SPEC::TI, get<0>(typename SPEC::SHAPE{}), get<1>(typename SPEC::SHAPE{})>> view{data(t)};
//        return view;
//    }
    namespace tensor{
            template <auto A, auto B>
            constexpr  bool greater_than(){
                return A > B;
            }
        template <auto A, auto B>
        constexpr bool equal(){
            return A == B;
        }
    }
    template<typename DEVICE, typename SPEC>
    auto _matrix_view_one_dim(DEVICE& device, const Tensor<SPEC>& t){
        using TI = typename SPEC::TI;
        constexpr TI N_DIM = length(typename SPEC::SHAPE{});
        static_assert(tensor::generalized_row_major<typename SPEC::SHAPE, typename SPEC::STRIDE>());
        using ROW_MAJOR_STRIDE = tensor::RowMajorStride<typename SPEC::SHAPE>;
        static_assert(tensor::same_dimensions_shape<ROW_MAJOR_STRIDE, typename SPEC::STRIDE>(), "Stride must be row major for creating a matrix view");
        using LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, 1, get<N_DIM-1>(typename SPEC::STRIDE{})>;
        const Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, 1, get<N_DIM-1>(typename SPEC::SHAPE{}), true, LAYOUT>> view{data(t)};
        return view;
    }
    template<typename DEVICE, typename SPEC>
    auto _matrix_view_one_dim(DEVICE& device, Tensor<SPEC>& t){
        using TI = typename SPEC::TI;
        constexpr TI N_DIM = length(typename SPEC::SHAPE{});
        static_assert(tensor::generalized_row_major<typename SPEC::SHAPE, typename SPEC::STRIDE>());
        using ROW_MAJOR_STRIDE = tensor::RowMajorStride<typename SPEC::SHAPE>;
        static_assert(tensor::same_dimensions_shape<ROW_MAJOR_STRIDE, typename SPEC::STRIDE>(), "Stride must be row major for creating a matrix view");
        using LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, 1, get<N_DIM-1>(typename SPEC::STRIDE{})>;
        const Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, 1, get<N_DIM-1>(typename SPEC::SHAPE{}), true, LAYOUT>> view{data(t)};
        return view;
    }
    template<typename DEVICE, typename SPEC>
    auto matrix_view(DEVICE& device, const Tensor<SPEC>& t){
        // broadcasting over the first N-1 dimensions => M x N x K => (M*N) x K
        using TI = typename SPEC::TI;
        constexpr TI N_DIM = length(typename SPEC::SHAPE{});
        if constexpr (N_DIM == 1){
            return _matrix_view_one_dim(device, t);
        }
        else{
            static_assert(tensor::generalized_row_major<typename SPEC::SHAPE, typename SPEC::STRIDE>());
            static_assert(tensor::dense_row_major_layout<SPEC, true>(), "Stride must be row major for creating a matrix view");
            using PROD = tensor::CumulativeProduct<tensor::PopBack<typename SPEC::SHAPE>>;
            constexpr TI TOTAL_ROWS = get<0>(PROD{});
            using LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, get<N_DIM-2>(typename SPEC::STRIDE{}), get<N_DIM-1>(typename SPEC::STRIDE{})>;
            const Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, TOTAL_ROWS, get<N_DIM-1>(typename SPEC::SHAPE{}), true, LAYOUT, true>> view{{data(t)}};
            return view;
        }
    }
    template<typename DEVICE, typename SPEC>
    auto matrix_view(DEVICE& device, Tensor<SPEC>& t){
        // broadcasting over the first N-1 dimensions => M x N x K => (M*N) x K
        using TI = typename SPEC::TI;
        constexpr TI N_DIM = length(typename SPEC::SHAPE{});
        if constexpr (N_DIM == 1){
            return _matrix_view_one_dim(device, t);
        }
        else{
            static_assert(tensor::generalized_row_major<typename SPEC::SHAPE, typename SPEC::STRIDE>());
            static_assert(tensor::dense_row_major_layout<SPEC, true>(), "Stride must be row major for creating a matrix view");
            using PROD = tensor::CumulativeProduct<tensor::PopBack<typename SPEC::SHAPE>>;
            constexpr TI TOTAL_ROWS = get<0>(PROD{});
            using LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, get<N_DIM-2>(typename SPEC::STRIDE{}), get<N_DIM-1>(typename SPEC::STRIDE{})>;
            const Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, TOTAL_ROWS, get<N_DIM-1>(typename SPEC::SHAPE{}), true, LAYOUT, false>> view{{data(t)}};
            return view;
        }
    }
    template<typename DEVICE, typename SPEC, typename RESHAPE>
    auto reshape_row_major(DEVICE& device, Tensor<SPEC>& t, const RESHAPE&){
        static_assert(tensor::dense_row_major_layout<SPEC, true>());
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        constexpr TI N_ELEMENTS = get<0>(tensor::CumulativeProduct<typename SPEC::SHAPE>{});
        constexpr TI N_NEW_ELEMENTS = get<0>(tensor::CumulativeProduct<RESHAPE>{});
        static_assert(N_ELEMENTS == N_NEW_ELEMENTS, "Tensor reshape: Number of elements must be the same");
        using STRIDE = typename SPEC::STRIDE;
        constexpr TI OLD_LAST_STRIDE = get<length(STRIDE{}) - 1>(STRIDE{});
        using NEW_STRIDE = tensor::PopFront<tensor::CumulativeProduct<tensor::Append<RESHAPE, OLD_LAST_STRIDE>>>;
        using NEW_SPEC = tensor::Specification<T, TI, RESHAPE, true, NEW_STRIDE>;
        return Tensor<NEW_SPEC>{data(t)};
    }
    template<typename DEVICE, typename SPEC, typename RESHAPE>
    auto reshape_row_major(DEVICE& device, const Tensor<SPEC>& t, const RESHAPE&){

        static_assert(tensor::dense_row_major_layout<SPEC, true>());
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        constexpr TI N_ELEMENTS = get<0>(tensor::CumulativeProduct<typename SPEC::SHAPE>{});
        constexpr TI N_NEW_ELEMENTS = get<0>(tensor::CumulativeProduct<RESHAPE>{});
        static_assert(N_ELEMENTS == N_NEW_ELEMENTS, "Tensor reshape: Number of elements must be the same");
        using STRIDE = typename SPEC::STRIDE;
        constexpr TI OLD_LAST_STRIDE = get<length(STRIDE{}) - 1>(STRIDE{});
        using NEW_STRIDE = tensor::PopFront<tensor::CumulativeProduct<tensor::Append<RESHAPE, OLD_LAST_STRIDE>>>;
        using NEW_SPEC = tensor::Specification<T, TI, RESHAPE, true, NEW_STRIDE>;
        return Tensor<NEW_SPEC>{data(t)};
    }
    template <typename DEVICE, typename MATRIX_SPEC>
    auto to_tensor(DEVICE& device, Matrix<MATRIX_SPEC>& m){
        using T = typename MATRIX_SPEC::T;
        using TI = typename MATRIX_SPEC::TI;
        using SHAPE = tensor::Shape<TI, MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        constexpr TI ROW_PITCH = MATRIX_SPEC::LAYOUT::template ROW_PITCH<MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        constexpr TI COL_PITCH = MATRIX_SPEC::LAYOUT::template COL_PITCH<MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        using STRIDE = tensor::Stride<TI, ROW_PITCH, COL_PITCH>;
        using SPEC = tensor::Specification<T, TI, SHAPE, true, STRIDE, false>;
        return Tensor<SPEC>{m._data};
    }
    template <typename DEVICE, typename MATRIX_SPEC>
    auto to_tensor(DEVICE& device, const Matrix<MATRIX_SPEC>& m){
        using T = typename MATRIX_SPEC::T;
        using TI = typename MATRIX_SPEC::TI;
        using SHAPE = tensor::Shape<TI, MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        constexpr TI ROW_PITCH = MATRIX_SPEC::LAYOUT::template ROW_PITCH<MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        constexpr TI COL_PITCH = MATRIX_SPEC::LAYOUT::template COL_PITCH<MATRIX_SPEC::ROWS, MATRIX_SPEC::COLS>;
        using STRIDE = tensor::Stride<TI, ROW_PITCH, COL_PITCH>;
        using SPEC = tensor::Specification<T, TI, SHAPE, true, STRIDE, true>;
        return Tensor<SPEC>{m._data};
    }
    template <typename DEVICE, typename SPEC>
    auto to_tensor(DEVICE& device, const Tensor<SPEC>& t){
        return t;
    }
    template <typename DEVICE, typename SPEC>
    auto to_tensor(DEVICE& device, Tensor<SPEC>& t){
        return t;
    }
    template <typename DEVICE, typename SPEC>
    auto squeeze(DEVICE& device, const Tensor<SPEC>& m){
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        constexpr TI N_DIM = length(typename SPEC::SHAPE{});
        static_assert(N_DIM > 1, "Cannot squeeze a tensor with less than 2 dimensions");
        static_assert(get<0>(typename SPEC::SHAPE{}) == 1, "Cannot squeeze a tensor with a dimension size greater than 1");
        using NEW_SHAPE = tensor::PopFront<typename SPEC::SHAPE>;
        using NEW_STRIDE = tensor::PopFront<typename SPEC::STRIDE>;
        using NEW_SPEC = tensor::Specification<T, TI, NEW_SHAPE, true, NEW_STRIDE>;
        return Tensor<NEW_SPEC>{data(m)};
    }
    template <typename DEVICE, typename SPEC>
    auto unsqueeze(DEVICE& device, const Tensor<SPEC>& m){
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        using NEW_SHAPE = tensor::Insert<typename SPEC::SHAPE, 1, 0>;
        using NEW_STRIDE = tensor::Insert<typename SPEC::STRIDE, get<0>(typename SPEC::STRIDE{}) * get<0>(typename SPEC::SHAPE{}), 0>;
        using NEW_SPEC = tensor::Specification<T, TI, NEW_SHAPE, true, NEW_STRIDE>;
        return Tensor<NEW_SPEC>{data(m)};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
