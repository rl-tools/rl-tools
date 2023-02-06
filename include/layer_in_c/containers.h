#ifndef LAYER_IN_C_CONTAINERS_H
#define LAYER_IN_C_CONTAINERS_H

namespace layer_in_c{
    template<typename T_T, typename T_TI, T_TI DIM>
    struct Vector{
        T_T* data = nullptr;
    };

    enum class Majority{
        RowMajor,
        ColMajor,
    };
    template <typename TI, TI ROWS, TI COLS, TI ALIGNMENT = 1>
    struct RowMajorAligned{
        static constexpr TI ROW_PITCH = (COLS + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        static constexpr TI COL_PITCH = 1;
    };
    template <typename T_T, typename T_TI, T_TI T_ROWS, T_TI T_COLS, typename T_LAYOUT = RowMajorAligned<T_TI, T_ROWS, T_COLS>> // row-major by default
    struct MatrixSpecification{
        using T = T_T;
        using TI = T_TI;
        using LAYOUT = T_LAYOUT;
        static constexpr TI ROW_PITCH = LAYOUT::ROW_PITCH;
        static constexpr TI COL_PITCH = LAYOUT::COL_PITCH;
        static_assert(COL_PITCH * T_COLS <= ROW_PITCH || ROW_PITCH * T_ROWS <= COL_PITCH, "Pitches of the matrix dimensions are not compatible");
        static constexpr TI SIZE = std::max(COL_PITCH * T_COLS, ROW_PITCH * T_ROWS);
        static constexpr TI SIZE_BYTES = SIZE * sizeof(T);
        static constexpr TI ROWS = T_ROWS;
        static constexpr TI COLS = T_COLS;
    };
    template<typename T_SPEC>
    struct Matrix{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI ROWS = SPEC::ROWS;
        static constexpr TI COLS = SPEC::COLS;
        static constexpr TI ROW_PITCH = SPEC::ROW_PITCH;
        static constexpr TI COL_PITCH = SPEC::COL_PITCH;

        T* data = nullptr;
    };
    namespace containers{
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_structure = SPEC_1::ROWS == SPEC_2::ROWS && SPEC_1::COLS == SPEC_2::COLS;
    }
}

#endif