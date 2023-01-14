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
    template<typename T_T, typename T_TI, T_TI ROWS, T_TI COLS, Majority LAYOUT = RowMajor>
    struct Matrix{
        T_T* data = nullptr;
    };
}

#endif