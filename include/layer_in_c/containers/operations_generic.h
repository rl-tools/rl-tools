#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, Matrix<SPEC>& matrix){
        matrix.data = new typename SPEC::T[SPEC::ROWS * SPEC::COLS];
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, Matrix<SPEC>& matrix){
        delete matrix.data;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void transpose(DEVICE& device, Matrix<SPEC_1>& target, Matrix<SPEC_2>& source){
        static_assert(SPEC_1::ROWS == SPEC_2::COLS);
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                target.data[i * SPEC::COLS + j] = source.data[j * SPEC::ROWS + i];
            }
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const Matrix<SPEC_1>& m1, const Matrix<SPEC_2>& m2){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        typename SPEC::T acc = 0;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS * SPEC::COLS; i++){
            acc += math::abs(m1.data[i] - m2.data[i]);
        }
        return acc;
    }



}
#endif
