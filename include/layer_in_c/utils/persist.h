#ifndef LAYER_IN_C_UTILS_PERSIST_H
#define LAYER_IN_C_UTILS_PERSIST_H
#include <vector>

namespace layer_in_c::utils::persist::array_conversion{
    template <typename DEVICE, typename T, auto ROWS>
    std::vector<T> vector_to_std_vector(T M[ROWS]){
        return std::vector<T>(M, M + ROWS);
    }
    template <typename DEVICE, typename T, auto ROWS, auto COLS>
    std::vector<std::vector<T>> matrix_to_std_vector(Matrix<T, typename DEVICE::index_t, ROWS, COLS, RowMajor> M){
        std::vector<std::vector<T>> data(ROWS);
        for(typename DEVICE::index_t i=0; i < ROWS; i++){
            data[i] = std::vector<T>(M[i], M[i] + COLS);
        }
        return data;
    }
    template <typename DEVICE, typename T, auto ROWS>
    void std_vector_to_vector(T target[ROWS], std::vector<T> source){
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