#ifndef TEST_UTILS_H
#define TEST_UTILS_H

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

template <typename T, int N_ROWS, int N_COLS>
T abs_diff_matrix(const T A[N_ROWS][N_COLS], const std::vector<std::vector<T>>& B) {
    T acc = 0;
    for (int i = 0; i < N_ROWS; i++){
        for (int j = 0; j < N_COLS; j++){
            acc += std::abs(A[i][j] - B[i][j]);
        }
    }
    return acc;
}

template <typename T, int N_ROWS, int N_COLS>
T abs_diff_matrix(const T A[N_ROWS][N_COLS], const T B[N_ROWS][N_COLS]) {
    T acc = 0;
    for (int i = 0; i < N_ROWS; i++){
        for (int j = 0; j < N_COLS; j++){
            acc += std::abs(A[i][j] - B[i][j]);
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

template <typename T, int N_ROWS, int N_COLS>
void assign(T A[N_ROWS][N_COLS], const std::vector<std::vector<T>>& B) {
    for (int i = 0; i < N_ROWS; i++){
        for (int j = 0; j < N_COLS; j++){
            A[i][j] = B[i][j];
        }
    }
}

#endif