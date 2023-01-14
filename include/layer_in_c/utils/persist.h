#ifndef LAYER_IN_C_UTILS_PERSIST_H
#define LAYER_IN_C_UTILS_PERSIST_H
#include <vector>

namespace layer_in_c::utils::persist::array_conversion{
    template <typename DEVICE, typename T, typename TI, TI ROWS, TI COLS>
    auto matrix_to_std_vector(DEVICE& device, Matrix<T, TI, ROWS, COLS, RowMajor> M){
        if constexpr(COLS == 1){
            return std::vector<T>(M.data, M.data + ROWS);
        }
        else{
            std::vector<std::vector<T>> data(ROWS);
            for(typename DEVICE::index_t i=0; i < ROWS; i++){
                data[i] = std::vector<T>(&M.data[i * COLS], &M.data[i * COLS] + COLS);
            }
            return data;
        }
    }
    template <typename DEVICE, typename T, auto ROWS>
    void std_vector_to_vector(DEVICE& device, T target[ROWS], std::vector<T> source){
        for (typename DEVICE::index_t i=0; i < ROWS; i++){
            target[i] = source[i];
        }
    }
    template <typename DEVICE, typename T, auto ROWS, auto COLS>
    void std_vector_to_matrix(Matrix<T, typename DEVICE::index_t, ROWS, COLS, RowMajor> target, std::vector<std::vector<T>> source){
        for(typename DEVICE::index_t i=0; i < ROWS; i++){
            for(typename DEVICE::index_t j=0; j < COLS; j++){
                target->data[i * COLS + j] = source[i][j];
            }
        }
    }
}

#endif