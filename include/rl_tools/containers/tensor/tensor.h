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

        template <typename T_T, typename T_TI, typename T_SHAPE, typename T_STRIDE>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using SHAPE = T_SHAPE;
            using STRIDE = T_STRIDE;
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
        static_assert(length(ELEMENT{}) > 1);
    };

    template<typename ELEMENT>
    struct PopBack: Element<
            typename ELEMENT::TI,
            ELEMENT::VALUE,
            utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT::NEXT_ELEMENT::NEXT_ELEMENT, FinalElement>,
                    PopBack<typename ELEMENT::NEXT_ELEMENT>,
                    Element<typename ELEMENT::TI, 0, FinalElement>
    >>{
    static_assert(length(ELEMENT{}) > 1);
    };

    template <typename ELEMENT>
    struct Product: Element<
            typename ELEMENT::TI,
            product(ELEMENT{}),
            utils::typing::conditional_t<!utils::typing::is_same_v<typename ELEMENT::NEXT_ELEMENT, FinalElement>,
                    Product<typename ELEMENT::NEXT_ELEMENT>,
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

    template <typename SHAPE>
    using RowMajorStride = Append<PopFront<Product<SHAPE>>, 1>;
    }

    template <typename T_SPEC>
    struct Tensor{
        using SPEC = T_SPEC;
    };
}

#endif