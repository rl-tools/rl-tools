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
    template <typename T_T, typename T_TI, T_TI T_ROWS, T_TI T_COLS, T_TI T_ROW_PITCH=T_COLS, T_TI T_COL_PITCH=1> // row-major by default
    struct MatrixSpecification{
        using T = T_T;
        using TI = T_TI;
        static_assert(T_COL_PITCH * T_COLS <= T_ROW_PITCH || T_ROW_PITCH * T_ROWS <= T_COL_PITCH, "Pitches of the matrix dimensions are not compatible");
        static constexpr TI SIZE = std::max(T_COL_PITCH * T_COLS, T_ROW_PITCH * T_ROWS);
        static constexpr TI SIZE_BYTES = SIZE * sizeof(T);
        static constexpr TI ROWS = T_ROWS;
        static constexpr TI COLS = T_COLS;
        static constexpr TI ROW_PITCH = T_ROW_PITCH;
        static constexpr TI COL_PITCH = T_COL_PITCH;
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