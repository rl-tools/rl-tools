#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>

namespace layer_in_c{
    template<typename DEVICE, typename T_T, typename T_TI, T_TI ROWS, T_TI COLS>
    void malloc(DEVICE& device, Matrix<T_T, T_TI, ROWS, COLS>& matrix){
        matrix.data = new T_T[ROWS * COLS];
    }
    template<typename DEVICE, typename T_T, typename T_TI, T_TI ROWS, T_TI COLS>
    void free(DEVICE& device, Matrix<T_T, T_TI, ROWS, COLS>& matrix){
        delete matrix.data;
    }

    template<typename DEVICE, typename T_T, typename T_TI, T_TI ROWS, T_TI COLS>
    void transpose(DEVICE& device, Matrix<T_T, T_TI, ROWS, COLS>& target, Matrix<T_T, T_TI, COLS, ROWS>& source){
        for(T_TI i = 0; i < ROWS; i++){
            for(T_TI j = 0; j < COLS; j++){
                target.data[i * COLS + j] = source.data[j * ROWS + i];
            }
        }
    }

    template<typename DEVICE, typename T_T, typename T_TI, T_TI ROWS, T_TI COLS>
    T_T abs_diff(DEVICE& device, const Matrix<T_T, T_TI, ROWS, COLS>& m1, const Matrix<T_T, T_TI, ROWS, COLS>& m2){
        T_T acc = 0;
        for(T_TI i = 0; i < ROWS * COLS; i++){
            acc += math::abs(m1.data[i] - m2.data[i]);
        }
        return acc;
    }



}
#endif
