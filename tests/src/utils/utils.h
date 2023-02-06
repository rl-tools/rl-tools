#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <vector>

template <typename T, int DIM>
void standardise(const T input[DIM], const T mean[DIM], const T std[DIM], T output[DIM]){
    for (int i = 0; i < DIM; i++){
        output[i] = (input[i] - mean[i]) / std[i];
    }
}
template <typename T, int DIM>
T abs_diff(const T A[DIM], const T B[DIM]){
    T acc = 0;
    for (int i = 0; i < DIM; i++){
        acc += std::abs(A[i] - B[i]);
    }
    return acc;
}

template <typename T, typename SPEC>
T abs_diff_matrix(const layer_in_c::Matrix<SPEC> A, const std::vector<std::vector<T>>& B) {
    T acc = 0;
    for (int i = 0; i < SPEC::ROWS; i++){
        for (int j = 0; j < SPEC::COLS; j++){
            auto index = layer_in_c::index(A, i, j);
            std::cout << "row pitch: " << row_pitch(A) << std::endl;
            acc += std::abs(A.data[index] - B[i][j]);
        }
    }
    return acc;
}

template <typename T, typename SPEC>
T abs_diff_matrix(layer_in_c::Matrix<SPEC> A, const T* B) {
    T acc = 0;
    for (int i = 0; i < SPEC::ROWS; i++){
        for (int j = 0; j < SPEC::COLS; j++){
            acc += std::abs(A[index(A, i, j)] - B[index(B, i, j)]);
        }
    }
    return acc;
}

template <typename T, int N_ROWS>
T abs_diff_vector(const T A[N_ROWS], const T B[N_ROWS]) {
    T acc = 0;
    for (int i = 0; i < N_ROWS; i++){
        acc += std::abs(A[i] - B[i]);
    }
    return acc;
}

template <typename T, typename SPEC>
void assign(layer_in_c::Matrix<SPEC> A, const std::vector<std::vector<T>>& B) {
    for (int i = 0; i < SPEC::ROWS; i++){
        for (int j = 0; j < SPEC::COLS; j++){
            A.data[index(A, i, j)] = B[i][j];
        }
    }
}

#endif