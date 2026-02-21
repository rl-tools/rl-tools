#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace tensor{
        template <typename T_TI, T_TI... T_VALUES>
        struct Tuple{
            using TI = T_TI;
            static constexpr TI LENGTH = sizeof...(T_VALUES);
            template <TI N>
            static constexpr TI GET = [](){
                static_assert(N < LENGTH, "Index out of bounds in Tuple::GET");
                if constexpr(LENGTH == 0){
                    return static_cast<TI>(0);
                }
                else{
                    constexpr TI values[] = {T_VALUES...};
                    return values[N];
                }
            }();

            static constexpr TI VALUE = [](){
                if constexpr(LENGTH == 0){
                    return static_cast<TI>(0);
                }
                else{
                    return GET<0>;
                }
            }();
            static constexpr TI FIRST = VALUE;
            static constexpr TI LAST = [](){
                if constexpr(LENGTH == 0){
                    return static_cast<TI>(0);
                }
                else{
                    return GET<LENGTH-1>;
                }
            }();
        };

        template <typename TI, TI... T_DIMS>
        struct Shape: Tuple<TI, T_DIMS...> {
        };

        template <typename TI, TI... T_DIMS>
        struct Stride: Tuple<TI, T_DIMS...> {
        };

    }
    template <typename TI, TI... VALUES>
    RL_TOOLS_FUNCTION_PLACEMENT TI constexpr length(tensor::Tuple<TI, VALUES...>, TI current_length=0){
        return static_cast<TI>(sizeof...(VALUES)) + current_length;
    }
    template <typename TI, TI... VALUES>
    RL_TOOLS_FUNCTION_PLACEMENT TI constexpr product(tensor::Tuple<TI, VALUES...>){
        return (static_cast<TI>(1) * ... * VALUES);
    }
    template <auto TARGET_INDEX_INPUT, typename TI, TI... VALUES>
    RL_TOOLS_FUNCTION_PLACEMENT TI constexpr get(tensor::Tuple<TI, VALUES...>){
        constexpr TI TARGET_INDEX = TARGET_INDEX_INPUT;
        static_assert(TARGET_INDEX >= 0, "Index out of bounds");
        static_assert(TARGET_INDEX < static_cast<TI>(sizeof...(VALUES)), "Index out of bounds");
        return tensor::Tuple<TI, VALUES...>::template GET<TARGET_INDEX>;
    }
    template <typename DEVICE, typename TI, TI... VALUES>
    RL_TOOLS_FUNCTION_PLACEMENT TI get(DEVICE& device, const tensor::Tuple<TI, VALUES...>, TI index){
        utils::assert_exit(device, index >= 0 && index < static_cast<TI>(sizeof...(VALUES)), "Index out of bounds");
        if constexpr(sizeof...(VALUES) == 0){
            return static_cast<TI>(0);
        }
        else{
            constexpr TI values[] = {VALUES...};
            return values[index];
        }
    }
    template <typename TI, TI... VALUES>
    RL_TOOLS_FUNCTION_PLACEMENT TI constexpr get_last(tensor::Tuple<TI, VALUES...>){
        static_assert(sizeof...(VALUES) > 0, "Cannot get last element of empty Tuple");
        return get<static_cast<TI>(sizeof...(VALUES) - 1)>(tensor::Tuple<TI, VALUES...>{});
    }
    namespace tensor {
        namespace shape_math {
            using SizeType = decltype(sizeof(0));

            template <typename T, SizeType N>
            struct ConstexprArray {
                T data[N > 0 ? N : 1];
            };

            template <SizeType... Is>
            struct IndexSequence {};

            template <SizeType N, SizeType... Is>
            struct MakeIndexSequenceImpl : MakeIndexSequenceImpl<N - 1, N - 1, Is...> {};

            template <SizeType... Is>
            struct MakeIndexSequenceImpl<0, Is...> {
                using type = IndexSequence<Is...>;
            };

            template <SizeType N>
            using MakeIndexSequence = typename MakeIndexSequenceImpl<N>::type;

            template <typename ELEMENT>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr SizeType rank() {
                return static_cast<SizeType>(length(ELEMENT{}));
            }

            template <typename ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto element_to_array_unpack(IndexSequence<Is...>) {
                using TI = typename ELEMENT::TI;
                return ConstexprArray<TI, sizeof...(Is)>{{get<static_cast<TI>(Is)>(ELEMENT{})...}};
            }

            template <typename ELEMENT>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto element_to_array() {
                return element_to_array_unpack<ELEMENT>(MakeIndexSequence<rank<ELEMENT>()>{});
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_append(ConstexprArray<TI, N> in, TI new_element) {
                ConstexprArray<TI, N + 1> out{};
                for (SizeType i = 0; i < N; ++i) {
                    out.data[i] = in.data[i];
                }
                out.data[N] = new_element;
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_prepend(ConstexprArray<TI, N> in, TI new_element) {
                ConstexprArray<TI, N + 1> out{};
                out.data[0] = new_element;
                for (SizeType i = 0; i < N; ++i) {
                    out.data[i + 1] = in.data[i];
                }
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_pop_front(ConstexprArray<TI, N> in) {
                ConstexprArray<TI, N - 1> out{};
                for (SizeType i = 1; i < N; ++i) {
                    out.data[i - 1] = in.data[i];
                }
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_pop_back(ConstexprArray<TI, N> in) {
                ConstexprArray<TI, N - 1> out{};
                for (SizeType i = 0; i + 1 < N; ++i) {
                    out.data[i] = in.data[i];
                }
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_cumulative_product(ConstexprArray<TI, N> in) {
                ConstexprArray<TI, N> out{};
                if constexpr (N > 0) {
                    out.data[N - 1] = in.data[N - 1];
                    for (SizeType i = N - 1; i > 0; --i) {
                        out.data[i - 1] = out.data[i] * in.data[i - 1];
                    }
                }
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_replace(ConstexprArray<TI, N> in, TI new_element, SizeType offset) {
                in.data[offset] = new_element;
                return in;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_insert(ConstexprArray<TI, N> in, TI new_element, SizeType offset) {
                ConstexprArray<TI, N + 1> out{};
                for (SizeType i = 0; i < offset; ++i) {
                    out.data[i] = in.data[i];
                }
                out.data[offset] = new_element;
                for (SizeType i = offset; i < N; ++i) {
                    out.data[i + 1] = in.data[i];
                }
                return out;
            }

            template <typename TI, SizeType N>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto compute_remove(ConstexprArray<TI, N> in, SizeType offset) {
                ConstexprArray<TI, N - 1> out{};
                for (SizeType i = 0; i < offset; ++i) {
                    out.data[i] = in.data[i];
                }
                for (SizeType i = offset + 1; i < N; ++i) {
                    out.data[i - 1] = in.data[i];
                }
                return out;
            }

            template <typename ELEMENT, auto NEW_ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto append_unpack(IndexSequence<Is...>) {
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_append<TI, rank<ELEMENT>()>(in, static_cast<TI>(NEW_ELEMENT));
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, auto NEW_ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto prepend_unpack(IndexSequence<Is...>) {
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_prepend<TI, rank<ELEMENT>()>(in, static_cast<TI>(NEW_ELEMENT));
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto pop_front_unpack(IndexSequence<Is...>) {
                static_assert(rank<ELEMENT>() > 0, "PopFront requires rank > 0");
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_pop_front<TI, rank<ELEMENT>()>(in);
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto pop_back_unpack(IndexSequence<Is...>) {
                static_assert(rank<ELEMENT>() > 0, "PopBack requires rank > 0");
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_pop_back<TI, rank<ELEMENT>()>(in);
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto cumulative_product_unpack(IndexSequence<Is...>) {
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_cumulative_product<TI, rank<ELEMENT>()>(in);
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto replace_unpack(IndexSequence<Is...>) {
                static_assert(rank<ELEMENT>() > 0, "Replace requires rank > 0");
                static_assert(NEW_ELEMENT_OFFSET < rank<ELEMENT>(), "Replace index out of bounds");
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_replace<TI, rank<ELEMENT>()>(in, static_cast<TI>(NEW_ELEMENT), static_cast<SizeType>(NEW_ELEMENT_OFFSET));
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto insert_unpack(IndexSequence<Is...>) {
                static_assert(NEW_ELEMENT_OFFSET <= rank<ELEMENT>(), "Insert index out of bounds");
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_insert<TI, rank<ELEMENT>()>(in, static_cast<TI>(NEW_ELEMENT), static_cast<SizeType>(NEW_ELEMENT_OFFSET));
                return Shape<TI, out.data[Is]...>{};
            }

            template <typename ELEMENT, auto ELEMENT_OFFSET, SizeType... Is>
            RL_TOOLS_FUNCTION_PLACEMENT constexpr auto remove_unpack(IndexSequence<Is...>) {
                static_assert(rank<ELEMENT>() > 0, "Remove requires rank > 0");
                static_assert(ELEMENT_OFFSET < rank<ELEMENT>(), "Remove index out of bounds");
                using TI = typename ELEMENT::TI;
                constexpr auto in = element_to_array<ELEMENT>();
                constexpr auto out = compute_remove<TI, rank<ELEMENT>()>(in, static_cast<SizeType>(ELEMENT_OFFSET));
                return Shape<TI, out.data[Is]...>{};
            }
        }

        template <typename ELEMENT, auto NEW_ELEMENT>
        using Append = decltype(shape_math::append_unpack<ELEMENT, NEW_ELEMENT>(shape_math::MakeIndexSequence<shape_math::rank<ELEMENT>() + 1>{}));

        template <typename ELEMENT, auto NEW_ELEMENT>
        using Prepend = decltype(shape_math::prepend_unpack<ELEMENT, NEW_ELEMENT>(shape_math::MakeIndexSequence<shape_math::rank<ELEMENT>() + 1>{}));

        template <typename ELEMENT>
        using PopFront = decltype(shape_math::pop_front_unpack<ELEMENT>(shape_math::MakeIndexSequence<(shape_math::rank<ELEMENT>() > 0 ? shape_math::rank<ELEMENT>() - 1 : 0)>{}));

        template <typename ELEMENT>
        using PopBack = decltype(shape_math::pop_back_unpack<ELEMENT>(shape_math::MakeIndexSequence<(shape_math::rank<ELEMENT>() > 0 ? shape_math::rank<ELEMENT>() - 1 : 0)>{}));

        template <typename ELEMENT>
        using CumulativeProduct = decltype(shape_math::cumulative_product_unpack<ELEMENT>(shape_math::MakeIndexSequence<shape_math::rank<ELEMENT>()>{}));

        template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET>
        using Replace = decltype(shape_math::replace_unpack<ELEMENT, NEW_ELEMENT, NEW_ELEMENT_OFFSET>(shape_math::MakeIndexSequence<shape_math::rank<ELEMENT>()>{}));

        template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET>
        using Insert = decltype(shape_math::insert_unpack<ELEMENT, NEW_ELEMENT, NEW_ELEMENT_OFFSET>(shape_math::MakeIndexSequence<shape_math::rank<ELEMENT>() + 1>{}));

        template <typename SHAPE, auto COMPARISON>
        constexpr bool RANK_LARGER_THAN = length(SHAPE{}) > COMPARISON;

        template <typename ELEMENT, auto ELEMENT_OFFSET>
        using Remove = decltype(shape_math::remove_unpack<ELEMENT, ELEMENT_OFFSET>(shape_math::MakeIndexSequence<(shape_math::rank<ELEMENT>() > 0 ? shape_math::rank<ELEMENT>() - 1 : 0)>{}));

        template <typename SHAPE>
        using RowMajorStride = Append<PopFront<CumulativeProduct<SHAPE>>, 1>;

        template <typename SHAPE, typename STRIDE>
        constexpr typename SHAPE::TI max_span(){
            static_assert(length(SHAPE{}) == length(STRIDE{}));
            if constexpr(length(SHAPE{}) == 1){
                return get<0>(SHAPE{}) * get<0>(STRIDE{});
            }
            else{
                using NEXT_SHAPE = PopFront<SHAPE>;
                using NEXT_STRIDE = PopFront<STRIDE>;
                auto previous = max_span<NEXT_SHAPE, NEXT_STRIDE>();
                auto current = get<0>(SHAPE{}) * get<0>(STRIDE{});
                return previous > current ? previous : current;
            }
        }

        template <typename T_T, typename T_TI, typename T_SHAPE, bool T_DYNAMIC_ALLOCATION=true, typename T_STRIDE = RowMajorStride<T_SHAPE>, bool T_CONST=false>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using SHAPE = T_SHAPE;
            using STRIDE = T_STRIDE;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
            static constexpr bool CONST = T_CONST;
            static constexpr TI SIZE = max_span<SHAPE, STRIDE>();
            static constexpr TI SIZE_BYTES = SIZE * sizeof(T);

        };
        template<auto T_DIM, auto T_SIZE=0>
        struct ViewSpec{
            static constexpr auto DIM = T_DIM;
            static constexpr auto SIZE = T_SIZE;
        };
        template <typename SHAPE, typename STRIDE>
        RL_TOOLS_FUNCTION_PLACEMENT bool constexpr generalized_row_major(){
            static_assert(length(SHAPE{}) == length(STRIDE{}));
            if constexpr(length(SHAPE{}) == 1){
                return true;
            }
            else{
                constexpr auto back_value_shape = get<length(SHAPE{})-1>(SHAPE{});
                constexpr auto back_value_stride = get<length(STRIDE{})-1>(STRIDE{});
                using NEXT_SHAPE = PopBack<SHAPE>;
                using NEXT_STRIDE = PopBack<STRIDE>;
                return back_value_shape * back_value_stride <= get<length(NEXT_STRIDE{})-1>(NEXT_STRIDE{}) && generalized_row_major<NEXT_SHAPE, NEXT_STRIDE>();
            }
        }
        template <typename A, typename B>
       RL_TOOLS_FUNCTION_PLACEMENT  bool constexpr same_dimensions_shape(){
            if constexpr(length(A{}) != length(B{})){
                return false;
            }
            if constexpr(length(A{}) == 0){
                return true;
            }
            else{
                using NEXT_A = PopFront<A>;
                using NEXT_B = PopFront<B>;
                return (A::VALUE == B::VALUE) && same_dimensions_shape<NEXT_A, NEXT_B>();
            }
        }
        template <typename SPEC_A, typename SPEC_B>
        RL_TOOLS_FUNCTION_PLACEMENT bool constexpr same_dimensions(){
            return same_dimensions_shape<typename SPEC_A::SHAPE, typename SPEC_B::SHAPE>();
        }


        template <typename SHAPE, typename STRIDE, bool RELAX_MAJOR=false>
        RL_TOOLS_FUNCTION_PLACEMENT bool constexpr _dense_row_major_layout_shape(){
            static_assert(length(SHAPE{}) > 0);
            if(length(STRIDE{}) != length(SHAPE{})){
                return false;
            }
            if constexpr(length(STRIDE{}) == 1){
                return RELAX_MAJOR || STRIDE::FIRST == 1;
            }
            else{
                if constexpr(RELAX_MAJOR && STRIDE::LENGTH == 2){
                    return STRIDE::FIRST >= STRIDE::template GET<1> * SHAPE::template GET<1>;
                }
                else{
                    using NEXT_SHAPE = PopFront<SHAPE>;
                    using NEXT_STRIDE = PopFront<STRIDE>;
                    return (STRIDE::VALUE == NEXT_STRIDE::FIRST * NEXT_SHAPE::FIRST || ((SHAPE::FIRST == 1) && (STRIDE::VALUE >= NEXT_STRIDE::FIRST * NEXT_SHAPE::FIRST))) && _dense_row_major_layout_shape<NEXT_SHAPE, NEXT_STRIDE, RELAX_MAJOR>();
                }
            }
        }
        template <typename SPEC, bool RELAX_MAJOR=false>
        RL_TOOLS_FUNCTION_PLACEMENT bool constexpr dense_row_major_layout(){
            return _dense_row_major_layout_shape<typename SPEC::SHAPE, typename SPEC::STRIDE, RELAX_MAJOR>();
        }
        namespace spec::view{
            namespace range{
                template <typename SHAPE, typename VIEW_SPEC>
                using Shape = tensor::Replace<SHAPE, VIEW_SPEC::SIZE, VIEW_SPEC::DIM>;
                template <typename STRIDE, typename VIEW_SPEC>
                using Stride = STRIDE;
                template <typename SPEC, typename VIEW_SPEC, bool T_CONST>
                using Specification = tensor::Specification<typename SPEC::T, typename SPEC::TI, Shape<typename SPEC::SHAPE, VIEW_SPEC>, true, Stride<typename SPEC::STRIDE, VIEW_SPEC>, T_CONST>;
            }
            namespace point{
                template <typename SHAPE, typename VIEW_SPEC>
                using Shape = tensor::Remove<SHAPE, VIEW_SPEC::DIM>;
                template <typename STRIDE, typename VIEW_SPEC>
                using Stride = tensor::Remove<STRIDE, VIEW_SPEC::DIM>;
                template <typename SPEC, typename VIEW_SPEC, bool T_CONST>
                using Specification = tensor::Specification<typename SPEC::T, typename SPEC::TI, Shape<typename SPEC::SHAPE, VIEW_SPEC>, true, Stride<typename SPEC::STRIDE, VIEW_SPEC>, T_CONST>;
            }
        }
    }

    namespace tensor{
        template <typename T, typename TI, TI SIZE>
        struct TensorStatic{
            static constexpr bool DYNAMIC_ALLOCATION = false;
            static_assert(SIZE > 0, "MSVC does not allow SIZE=0");
            T _data[SIZE];
        };
        template <typename T>
        struct TensorStaticEmpty{
            static constexpr bool DYNAMIC_ALLOCATION = false;
            T* _data = nullptr;
        };
        template <typename T, typename TI, TI SIZE, bool CONST = false>
        struct TensorDynamic{
            static constexpr bool DYNAMIC_ALLOCATION = true;
            T* _data = nullptr;
        };
        template <typename T, typename TI, TI SIZE>
        struct TensorDynamic<T, TI, SIZE, true>{
            static constexpr bool DYNAMIC_ALLOCATION = true;
            const T* _data;
        };
    }

    template <typename T_SPEC>
    struct Tensor: utils::typing::conditional_t<T_SPEC::DYNAMIC_ALLOCATION, tensor::TensorDynamic<typename T_SPEC::T, typename T_SPEC::TI, T_SPEC::SIZE, T_SPEC::CONST>, utils::typing::conditional_t<(T_SPEC::SIZE > 0), tensor::TensorStatic<typename T_SPEC::T, typename T_SPEC::TI, T_SPEC::SIZE>, tensor::TensorStaticEmpty<typename T_SPEC::T>>>{
        using SPEC = T_SPEC;
        using SHAPE = typename SPEC::SHAPE;
        using T = typename SPEC::T;
        template <typename VIEW_SPEC>
        using VIEW_POINT = Tensor<tensor::spec::view::point::Specification<SPEC, VIEW_SPEC, SPEC::CONST>>;
        template <typename VIEW_SPEC>
        using VIEW_RANGE = Tensor<tensor::spec::view::range::Specification<SPEC, VIEW_SPEC, SPEC::CONST>>;
        // Tensor() = default;
//        Tensor(DATA_TYPE data): _data(data){};
    };

    template <typename T, typename TI, TI SIZE>
    RL_TOOLS_FUNCTION_PLACEMENT auto data(tensor::TensorStatic<T, TI, SIZE>& tensor){
        return tensor._data;
    }
    template <typename T, typename TI, TI SIZE>
    RL_TOOLS_FUNCTION_PLACEMENT auto data(const tensor::TensorStatic<T, TI, SIZE>& tensor){
        return &tensor._data[0];
    }

    template <typename T, typename TI, TI SIZE, bool CONST>
    RL_TOOLS_FUNCTION_PLACEMENT auto data(tensor::TensorDynamic<T, TI, SIZE, CONST>& tensor){
        return tensor._data;
    }
    template <typename T, typename TI, TI SIZE, bool CONST>
    RL_TOOLS_FUNCTION_PLACEMENT auto data(const tensor::TensorDynamic<T, TI, SIZE, CONST>& tensor){
        return &tensor._data[0];
    }

    template <typename T, typename TI, TI SIZE, bool CONST>
    RL_TOOLS_FUNCTION_PLACEMENT T** data_pointer(tensor::TensorDynamic<T, TI, SIZE, CONST>& tensor){
        return &tensor._data;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
