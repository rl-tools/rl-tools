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
    constexpr auto RowMajor = Majority::RowMajor;
    constexpr auto ColMajor = Majority::ColMajor;
    template <typename T_T, typename T_TI, T_TI T_ROWS, T_TI T_COLS, Majority T_LAYOUT = RowMajor>
    struct MatrixSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI ROWS = T_ROWS;
        static constexpr TI COLS = T_COLS;
        static constexpr Majority LAYOUT = T_LAYOUT;
    };
    template<typename T_SPEC>
    struct Matrix{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI ROWS = SPEC::ROWS;
        static constexpr TI COLS = SPEC::COLS;
        static constexpr Majority LAYOUT = SPEC::LAYOUT;

        T* data = nullptr;
    };
    namespace containers{
        template<typename SPEC_1, typename SPEC_2>
        constexpr bool check_structure = SPEC_1::ROWS == SPEC_2::ROWS && SPEC_1::COLS == SPEC_2::COLS;
    }
}

#endif