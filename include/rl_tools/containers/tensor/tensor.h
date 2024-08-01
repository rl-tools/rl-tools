#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace tensor{
        struct FinalElement{};
        template <typename T_TI, T_TI T_VALUE, typename T_NEXT_ELEMENT>
        struct Element{
            using TI = T_TI;
            static constexpr TI VALUE = T_VALUE;
    //            static constexpr bool FINAL_ELEMENT = utils::typing::is_same_v<T_NEXT_ELEMENT, FinalElement>;
            using NEXT_ELEMENT = T_NEXT_ELEMENT;
        };


        template <typename T_TI, T_TI... T_VALUES>
        struct Tuple: Element<T_TI, 0, FinalElement>{
        };

        template <typename T_TI, T_TI T_VALUE, T_TI... T_VALUES>
        struct Tuple<T_TI, T_VALUE, T_VALUES...>: Element<T_TI, T_VALUE, Tuple<T_TI, T_VALUES...>>{
            using TI = T_TI;
            static constexpr TI VALUE = T_VALUE;
        };

        template <typename TI, TI... T_DIMS>
        struct Shape: Tuple<TI, T_DIMS...> {
        };

        template <typename TI, TI... T_DIMS>
        struct Stride: Tuple<TI, T_DIMS...> {
        };

    }
    template <typename TI, TI VALUE, typename NEXT_ELEMENT>
    TI constexpr length(tensor::Element<TI, VALUE, NEXT_ELEMENT>, TI current_length=0){
        if constexpr(utils::typing::is_same_v<NEXT_ELEMENT, tensor::FinalElement>){
            return current_length;
        }
        else{
            return length(NEXT_ELEMENT{}, current_length+1);
        }
    }
    template <typename TI, TI VALUE, typename NEXT_ELEMENT>
    TI constexpr product(tensor::Element<TI, VALUE, NEXT_ELEMENT>){
        if constexpr(utils::typing::is_same_v<NEXT_ELEMENT, tensor::FinalElement>){
            return 1;
        }
        else{
            return VALUE * product(NEXT_ELEMENT{});
        }
    }
    template <auto TARGET_INDEX_INPUT, typename TI, TI VALUE, typename NEXT_ELEMENT>
    TI constexpr get(tensor::Element<TI, VALUE, NEXT_ELEMENT>){
        constexpr TI TARGET_INDEX = TARGET_INDEX_INPUT;
    //        constexpr bool LAST_ELEMENT = utils::typing::is_same_v<NEXT_ELEMENT, tensor::FinalElement>;
        static_assert(TARGET_INDEX <= length(NEXT_ELEMENT{}), "Index out of bounds");
        if constexpr(TARGET_INDEX == 0){
            return VALUE;
        }
        else{
            return get<TARGET_INDEX_INPUT-1>(NEXT_ELEMENT{});
        }
    }
    namespace tensor {
        template<typename ELEMENT, auto NEW_ELEMENT> // since the last Element is a dummy element containing 0, we need to insert the new element once the NEXT_ELEMENT is the FinalElement
        struct Append: Element<
                typename ELEMENT::TI,
                !utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement> ? ELEMENT::VALUE : NEW_ELEMENT,
        utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
        Append<typename ELEMENT::NEXT_ELEMENT, NEW_ELEMENT>,
        Element<typename ELEMENT::TI, 0, FinalElement>
        >>{};
        template<typename ELEMENT, auto NEW_ELEMENT> // since the last Element is a dummy element containing 0, we need to insert the new element once the NEXT_ELEMENT is the FinalElement
        struct Prepend: Element<typename ELEMENT::TI, NEW_ELEMENT, ELEMENT>{};

        template<typename ELEMENT>
        struct PopFront: ELEMENT::NEXT_ELEMENT{
            static_assert(length(ELEMENT{}) > 0);
        };

        template<typename ELEMENT>
        struct PopBack: utils::typing::conditional_t<
                utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT::NEXT_ELEMENT, FinalElement>,
                typename ELEMENT::NEXT_ELEMENT,
                Element<typename ELEMENT::TI, ELEMENT::VALUE, PopBack<typename ELEMENT::NEXT_ELEMENT>>
        >{
            static_assert(length(ELEMENT{}) > 0);
        };

        template <typename ELEMENT>
        struct CumulativeProduct: Element< // e.g. (2, 3, 4) -> (24, 12, 4)
                typename ELEMENT::TI,
                product(ELEMENT{}),
                utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
                        CumulativeProduct<typename ELEMENT::NEXT_ELEMENT>,
                        FinalElement
        >>{};
        template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET>
        struct Replace: Element<
                typename ELEMENT::TI,
                NEW_ELEMENT_OFFSET == 0 ? NEW_ELEMENT : ELEMENT::VALUE,
                utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
                        Replace<typename ELEMENT::NEXT_ELEMENT, NEW_ELEMENT, NEW_ELEMENT_OFFSET-1>,
                        FinalElement
                >>{
        };

        template <typename ELEMENT, auto NEW_ELEMENT, auto NEW_ELEMENT_OFFSET>
        struct Insert: utils::typing::conditional_t<NEW_ELEMENT_OFFSET == 0,
                Element<typename ELEMENT::TI, NEW_ELEMENT, ELEMENT>,//Element<typename ELEMENT::TI, ELEMENT::VALUE, typename ELEMENT::NEXT_ELEMENT>>,
                Element<typename ELEMENT::TI, ELEMENT::VALUE,
                utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
                Insert<typename ELEMENT::NEXT_ELEMENT, NEW_ELEMENT, NEW_ELEMENT_OFFSET-1>,
                FinalElement
                >>>{
        };

        template <typename SHAPE, auto COMPARISON>
        constexpr bool RANK_LARGER_THAN = length(SHAPE{}) > COMPARISON;

        template <typename ELEMENT, auto ELEMENT_OFFSET> //, typename utils::typing::enable_if_t<tensor::RANK_LARGER_THAN<ELEMENT, ELEMENT_OFFSET>, void>* = nullptr>
        struct Remove: utils::typing::conditional_t<ELEMENT_OFFSET == 0,
                typename ELEMENT::NEXT_ELEMENT,
                utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
                        Element<typename ELEMENT::TI, ELEMENT::VALUE, Remove<typename ELEMENT::NEXT_ELEMENT, ELEMENT_OFFSET-1>>,
                        FinalElement
                >>{
            static_assert(length(ELEMENT{}) > ELEMENT_OFFSET);
        };

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

        template <typename T_T, typename T_TI, typename T_SHAPE, typename T_STRIDE = RowMajorStride<T_SHAPE>, bool T_STATIC=false, bool T_CONST=false>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using SHAPE = T_SHAPE;
            using STRIDE = T_STRIDE;
            static constexpr bool STATIC = T_STATIC;
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
        bool constexpr generalized_row_major(){
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
        bool constexpr same_dimensions_shape(){
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
        bool constexpr same_dimensions(){
            return same_dimensions_shape<typename SPEC_A::SHAPE, typename SPEC_B::SHAPE>();
        }


        template <typename SHAPE, typename STRIDE, bool RELAX_MAJOR=false>
        bool constexpr _dense_row_major_layout_shape(){
            static_assert(length(SHAPE{}) > 0);
            if(length(STRIDE{}) != length(SHAPE{})){
                return false;
            }
            if constexpr(length(STRIDE{}) == 1){
                return RELAX_MAJOR || get<0>(STRIDE{}) == 1;
            }
            else{
                using NEXT_SHAPE = PopFront<SHAPE>;
                using NEXT_STRIDE = PopFront<STRIDE>;
                return (STRIDE::VALUE == get<0>(NEXT_STRIDE{}) * get<0>(NEXT_SHAPE{})) && _dense_row_major_layout_shape<NEXT_SHAPE, NEXT_STRIDE, RELAX_MAJOR>();
            }
        }
        template <typename SPEC, bool RELAX_MAJOR=false>
        bool constexpr dense_row_major_layout(){
            return _dense_row_major_layout_shape<typename SPEC::SHAPE, typename SPEC::STRIDE, RELAX_MAJOR>();
        }
        namespace spec::view{
            namespace range{
                template <typename SHAPE, typename VIEW_SPEC>
                using Shape = tensor::Replace<SHAPE, VIEW_SPEC::SIZE, VIEW_SPEC::DIM>;
                template <typename STRIDE, typename VIEW_SPEC>
                using Stride = STRIDE;
                template <typename SPEC, typename VIEW_SPEC, bool T_CONST>
                using Specification = tensor::Specification<typename SPEC::T, typename SPEC::TI, Shape<typename SPEC::SHAPE, VIEW_SPEC>, Stride<typename SPEC::STRIDE, VIEW_SPEC>, false, T_CONST>;
            }
            namespace point{
                template <typename SHAPE, typename VIEW_SPEC>
                using Shape = tensor::Remove<SHAPE, VIEW_SPEC::DIM>;
                template <typename STRIDE, typename VIEW_SPEC>
                using Stride = tensor::Remove<STRIDE, VIEW_SPEC::DIM>;
                template <typename SPEC, typename VIEW_SPEC, bool T_CONST>
                using Specification = tensor::Specification<typename SPEC::T, typename SPEC::TI, Shape<typename SPEC::SHAPE, VIEW_SPEC>, Stride<typename SPEC::STRIDE, VIEW_SPEC>, false, T_CONST>;
            }
        }
    }

    template <typename T_SPEC>
    struct Tensor{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        template <typename VIEW_SPEC>
        using VIEW_POINT = Tensor<tensor::spec::view::point::Specification<SPEC, VIEW_SPEC, SPEC::CONST>>;
        template <typename VIEW_SPEC>
        using VIEW_RANGE = Tensor<tensor::spec::view::range::Specification<SPEC, VIEW_SPEC, SPEC::CONST>>;
        using T_CV = utils::typing::conditional_t<SPEC::CONST, const T, T>;
        using DATA_TYPE = utils::typing::conditional_t<SPEC::STATIC, T_CV[SPEC::SIZE], T_CV*>;
        DATA_TYPE _data;
    };

    template <typename SPEC>
    constexpr auto data(Tensor<SPEC>& tensor){
        return tensor._data;
    }

    template <typename SPEC>
    constexpr auto data(const Tensor<SPEC>& tensor){
        return &tensor._data[0];
    }
    template <typename SPEC>
    constexpr typename SPEC::T*& data_reference(Tensor<SPEC>& tensor){
        return tensor._data;
    }
    struct TensorDynamicTag{
        template<typename SPEC>
        using type = Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::SHAPE, typename SPEC::STRIDE, false>>;
    };
    struct TensorStaticTag{
        template<typename SPEC>
        using type = Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::SHAPE, typename SPEC::STRIDE, true>>;
    };
}

#endif