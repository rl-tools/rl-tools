#include "version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace matrix{
        namespace layouts{

            template <typename TI, TI ALIGNMENT = 1>
            struct RowMajorAlignment{
                template <TI ROWS, TI COLS>
                static constexpr TI ROW_PITCH = ((COLS + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
                template <TI ROWS, TI COLS>
                static constexpr TI COL_PITCH = 1;
            };

            template <typename TI, TI ALIGNMENT = 1>
            using RowMajorAlignmentOptimized = RowMajorAlignment<TI, ALIGNMENT>;

            template <typename TI, TI T_ROW_PITCH, TI T_COL_PITCH>
            struct Fixed{
                template <TI ROWS, TI COLS>
                static constexpr TI ROW_PITCH = T_ROW_PITCH;
                template <TI ROWS, TI COLS>
                static constexpr TI COL_PITCH = T_COL_PITCH;
            };

            template <typename TI>
            using DEFAULT = layouts::RowMajorAlignmentOptimized<TI>; // row-major by default

        }
        template <typename T_T, typename T_TI, T_TI T_ROWS, T_TI T_COLS, typename LAYOUT_FACTORY = matrix::layouts::DEFAULT<T_TI>, bool T_IS_VIEW = false>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr TI ROWS = T_ROWS;
            static constexpr TI COLS = T_COLS;
            static constexpr bool IS_VIEW = T_IS_VIEW;
            static constexpr TI ROW_PITCH = LAYOUT_FACTORY::template ROW_PITCH<ROWS, COLS>;
            static constexpr TI COL_PITCH = LAYOUT_FACTORY::template COL_PITCH<ROWS, COLS>;
            using LAYOUT = layouts::Fixed<TI, ROW_PITCH, COL_PITCH>;

            static_assert(COL_PITCH * T_COLS <= ROW_PITCH || ROW_PITCH * T_ROWS <= COL_PITCH, "Pitches of the matrix dimensions are not compatible");
            static constexpr bool ROW_MAJOR = ROWS * ROW_PITCH >= COL_PITCH * COLS;
            static constexpr TI SIZE = ROW_MAJOR ? ROWS * ROW_PITCH : COLS * COL_PITCH;
            static constexpr TI SIZE_BYTES = SIZE * sizeof(T); // todo: discount size for views
        };
        template <auto T_ROWS, auto T_COLS> // todo: replace old view() calling convention by new ViewSpec based one
        struct ViewSpec{
            static constexpr auto ROWS = T_ROWS;
            static constexpr auto COLS = T_COLS;
        };
    }
    template<typename T_SPEC>
    struct Matrix{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI ROWS = SPEC::ROWS;
        static constexpr TI COLS = SPEC::COLS;
        static constexpr bool IS_VIEW = SPEC::IS_VIEW;
        static constexpr TI ROW_PITCH = SPEC::ROW_PITCH;
        static constexpr TI COL_PITCH = SPEC::COL_PITCH;

        using VIEW_LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
//        // pure virtual function to make this class abstract (should be instantiated by either the MatrixStatic or MatrixDynamic subclasses class)
//        virtual void _abstract_tag() = 0;
        T* _data = nullptr;
        Matrix() = default;
        Matrix(T* data): _data(data){};
    };
    template<typename T_SPEC>
    struct MatrixDynamic: Matrix<T_SPEC>{
        using T = typename MatrixDynamic::T;
        using TI = typename MatrixDynamic::TI;
        using SPEC = T_SPEC;

        template<typename SPEC::TI ROWS, typename SPEC::TI COLS>
        using VIEW = MatrixDynamic<matrix::Specification<T, TI, ROWS, COLS, typename MatrixDynamic::VIEW_LAYOUT, true>>;
//        virtual void _abstract_tag(){};
    };
    struct MatrixDynamicTag{
        template<typename T_SPEC>
        using type = MatrixDynamic<T_SPEC>;
    };
    template<typename T_SPEC>
    struct MatrixStatic: Matrix<T_SPEC>{
        using T = typename MatrixStatic::T;
        using TI = typename MatrixStatic::TI;
        using SPEC = T_SPEC;

        template<typename SPEC::TI ROWS, typename SPEC::TI COLS>
        using VIEW = MatrixDynamic<matrix::Specification<T, TI, ROWS, COLS, typename MatrixStatic::VIEW_LAYOUT, true>>;
//        virtual void _abstract_tag(){};

        alignas(T) unsigned char _data_memory[SPEC::SIZE_BYTES];
        MatrixStatic(): Matrix<T_SPEC>((T*)_data_memory){};
    };
    struct MatrixStaticTag{
        template<typename T_SPEC>
        using type = MatrixStatic<T_SPEC>;
    };
    namespace matrix{
        template <typename T, typename TI>
        using Empty = Matrix<matrix::Specification<T, TI, 0, 0>>;
    }


    namespace containers{
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_structure = SPEC_1::ROWS == SPEC_2::ROWS && SPEC_1::COLS == SPEC_2::COLS;
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_memory_layout = check_structure<SPEC_1, SPEC_2> && SPEC_1::ROW_PITCH == SPEC_2::ROW_PITCH && SPEC_1::COL_PITCH == SPEC_2::COL_PITCH && SPEC_1::IS_VIEW == false && SPEC_2::IS_VIEW == false && utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T> && SPEC_1::SIZE_BYTES == SPEC_2::SIZE_BYTES; // check if e.g. direct memory copy is possible
    }

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
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif