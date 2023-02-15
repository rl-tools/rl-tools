#ifndef LAYER_IN_C_CONTAINERS_H
#define LAYER_IN_C_CONTAINERS_H

namespace layer_in_c{
    namespace matrix{
        namespace layouts{

            template <typename TI, TI ALIGNMENT = 1>
            struct RowMajorAlignment{
                template <TI ROWS, TI COLS>
                static constexpr TI ROW_PITCH = ((COLS + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
                template <TI ROWS, TI COLS>
                static constexpr TI COL_PITCH = 1;
            };

            template <typename TI, TI ALIGNMENT = 128>
            using RowMajorAlignmentOptimized = RowMajorAlignment<TI, ALIGNMENT>;

            template <typename TI, TI T_ROW_PITCH, TI T_COL_PITCH>
            struct Fixed{
                template <TI ROWS, TI COLS>
                static constexpr TI ROW_PITCH = T_ROW_PITCH;
                template <TI ROWS, TI COLS>
                static constexpr TI COL_PITCH = T_COL_PITCH;
            };

        }
        template <typename T_T, typename T_TI, T_TI T_ROWS, T_TI T_COLS, typename LAYOUT_FACTORY = layouts::RowMajorAlignmentOptimized<T_TI>, bool T_IS_VIEW = false> // row-major by default
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
            static constexpr TI SIZE_BYTES = SIZE * sizeof(T);
        };




    }
    template<typename T_SPEC>
    struct Matrix{
        Matrix(){};
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI ROWS = SPEC::ROWS;
        static constexpr TI COLS = SPEC::COLS;
        static constexpr bool IS_VIEW = SPEC::IS_VIEW;
        static constexpr TI ROW_PITCH = SPEC::ROW_PITCH;
        static constexpr TI COL_PITCH = SPEC::COL_PITCH;

        using VIEW_LAYOUT = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
        template<typename SPEC::TI ROWS, typename SPEC::TI COLS>
        using VIEW = Matrix<matrix::Specification<T, TI, ROWS, COLS, VIEW_LAYOUT, true>>;

        T* _data = nullptr;
    };

    namespace containers{
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_structure = SPEC_1::ROWS == SPEC_2::ROWS && SPEC_1::COLS == SPEC_2::COLS;
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_memory_layout = check_structure<SPEC_1, SPEC_2> && SPEC_1::ROW_PITCH == SPEC_2::ROW_PITCH && SPEC_1::COL_PITCH == SPEC_2::COL_PITCH;
    }
}

#endif