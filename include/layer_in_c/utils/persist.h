#ifndef LAYER_IN_C_UTILS_PERSIST_H
#define LAYER_IN_C_UTILS_PERSIST_H
#include <vector>

namespace layer_in_c::utils::persist::array_conversion{
    template <typename T, index_t ROWS>
    std::vector<T> vector_to_std_vector(T M[ROWS]){
        return std::vector<T>(M, M + ROWS);
    }
    template <typename T, index_t ROWS, index_t COLS>
    std::vector<std::vector<T>> matrix_to_std_vector(T M[ROWS][COLS]){
        std::vector<std::vector<T>> data(ROWS);
        for(index_t i=0; i < ROWS; i++){
            data[i] = std::vector<T>(M[i], M[i] + COLS);
        }
        return data;
    }
    template <typename T, index_t ROWS>
    void std_vector_to_vector(T target[ROWS], std::vector<T> source){
        for (index_t i=0; i < ROWS; i++){
            target[i] = source[i];
        }
    }
    template <typename T, index_t ROWS, index_t COLS>
    void std_vector_to_matrix(T target[ROWS][COLS], std::vector<std::vector<T>> source){
        for(index_t i=0; i < ROWS; i++){
            for(index_t j=0; j < COLS; j++){
                target[i][j] = source[i][j];
            }
        }
    }
}

#endif