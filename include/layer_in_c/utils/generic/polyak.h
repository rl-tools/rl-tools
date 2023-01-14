#ifndef LAYER_IN_C_NN_UTILS_POLYAK
#define LAYER_IN_C_NN_UTILS_POLYAK


namespace layer_in_c::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    template<typename DEVICE, typename T, typename DEVICE::index_t N_ROWS, typename DEVICE::index_t N_COLS>
    void update(DEVICE& dev, Matrix<T, typename DEVICE::index_t, N_ROWS, N_COLS, RowMajor>& target, const Matrix<T, typename DEVICE::index_t, N_ROWS, N_COLS, RowMajor>& source, const T polyak) {
        for(typename DEVICE::index_t i = 0; i < N_ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < N_COLS; j++) {
                target.data[i * N_COLS + j] = polyak * target.data[i * N_COLS + j] + (1 - polyak) * source.data[i * N_COLS + j];
            }
        }
    }

    template<typename DEVICE, typename T, typename DEVICE::index_t N_ROWS, typename DEVICE::index_t N_COLS>
    void update_squared(DEVICE& dev, Matrix<T, typename DEVICE::index_t, N_ROWS, N_COLS, RowMajor>& target, const Matrix<T, typename DEVICE::index_t, N_ROWS, N_COLS, RowMajor>& source, const T polyak) {
        for(typename DEVICE::index_t i = 0; i < N_ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < N_COLS; j++) {
                T s = source.data[i * N_COLS + j];
                target.data[i * N_COLS + j] = polyak * target.data[i * N_COLS + j] + (1 - polyak) * s * s;
            }
        }
    }
}


#endif