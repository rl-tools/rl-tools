template<typename T, int N_ROWS, int N_COLS>
void polyak_update_matrix(T target[N_ROWS][N_COLS], const T source[N_ROWS][N_COLS], const T polyak) {
    for(int i = 0; i < N_ROWS; i++) {
        for(int j = 0; j < N_COLS; j++) {
            target[i][j] = polyak * target[i][j] + (1 - polyak) * source[i][j];
        }
    }
}
template<typename T, int DIM>
void polyak_update(T target[DIM], const T source[DIM], const T polyak) {
    for(int i = 0; i < DIM; i++) {
        target[i] = polyak * target[i] + (1 - polyak) * source[i];
    }
}

template<typename T, int N_ROWS, int N_COLS>
void polyak_update_squared_matrix(T target[N_ROWS][N_COLS], const T source[N_ROWS][N_COLS], const T polyak) {
    for(int i = 0; i < N_ROWS; i++) {
        for(int j = 0; j < N_COLS; j++) {
            target[i][j] = polyak * target[i][j] + (1 - polyak) * source[i][j] * source[i][j];
        }
    }
}
template<typename T, int DIM>
void polyak_update_squared(T target[DIM], const T source[DIM], const T polyak) {
    for(int i = 0; i < DIM; i++) {
        target[i] = polyak * target[i] + (1 - polyak) * source[i] * source[i];
    }
}
