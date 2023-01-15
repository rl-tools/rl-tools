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
    template<typename DEVICE, typename SPEC>
    Matrix<MatrixSpecification<typename SPEC::T, typename SPEC::TI, SPEC::COLS, SPEC::ROWS>> transpose(DEVICE& device, Matrix<SPEC>& target){
        static_assert(SPEC::LAYOUT == RowMajor);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = i + 1; j < SPEC::COLS; j++){
                T temp = target.data[i * SPEC::COLS + j];
                target.data[i * SPEC::COLS + j] = target.data[j * SPEC::ROWS + i];
                target.data[j * SPEC::ROWS + i] = temp;
            }
        }
        auto data = target.data;
        target.data = nullptr;
        return {data};
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

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void copy(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS * SPEC::COLS; i++){
            target.data[i] = source.data[i];
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS * SPEC::COLS; i++){
            target.data[i] += source.data[i];
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add_broadcast(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(SPEC_1::COLS == SPEC_2::COLS);
        static_assert(SPEC_2::ROWS == 1);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                target.data[i * SPEC::COLS + j] += source.data[j];
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void set_broadcast(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(SPEC_1::COLS == SPEC_2::COLS);
        static_assert(SPEC_2::ROWS == 1);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                target.data[i * SPEC::COLS + j] = source.data[j];
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename VALUE_T>
    void set(DEVICE& device, const Matrix<SPEC>& m, VALUE_T value){
        for(typename SPEC::TI i = 0; i < SPEC::ROWS * SPEC::COLS; i++){
            m.data[i] = value;
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3>
    void mul(DEVICE& device, const Matrix<SPEC_1>& A, const Matrix<SPEC_2>& B, const Matrix<SPEC_3>& C){
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        static_assert(SPEC_1::ROWS == SPEC_3::ROWS);
        static_assert(SPEC_2::COLS == SPEC_3::COLS);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        static_assert(SPEC_3::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = 0; j < SPEC::COLS; j++){
                T acc = 0;
                for(TI k = 0; k < SPEC::COLS; k++){
                    acc += A.data[i * SPEC::COLS + k] * B.data[k * SPEC::ROWS + j];
                }
                C.data[i * SPEC::COLS + j] = acc;
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3>
    void hcat(DEVICE& device, const Matrix<SPEC_1>& A, const Matrix<SPEC_2>& B, const Matrix<SPEC_3>& C){
        static_assert(SPEC_1::ROWS == SPEC_2::ROWS);
        static_assert(SPEC_3::ROWS == SPEC_1::ROWS);
        static_assert(SPEC_1::COLS + SPEC_2::COLS == SPEC_3::COLS);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        static_assert(SPEC_3::LAYOUT == RowMajor);
        // concatenate horizontally
        using TI = typename SPEC_1::TI;
        using T = typename SPEC_1::T;
        for(TI i = 0; i < SPEC_1::ROWS; i++){
            for(TI j = 0; j < SPEC_1::COLS; j++){
                C.data[i * SPEC_3::COLS + j] = A.data[i * SPEC_1::COLS + j];
            }
            for(TI j = 0; j < SPEC_2::COLS; j++){
                C.data[i * SPEC_3::COLS + j + SPEC_1::COLS] = B.data[i * SPEC_2::COLS + j];
            }
        }
    }
    // vcat
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3>
    void vcat(DEVICE& device, const Matrix<SPEC_1>& A, const Matrix<SPEC_2>& B, const Matrix<SPEC_3>& C){
        static_assert(SPEC_1::COLS == SPEC_2::COLS);
        static_assert(SPEC_3::COLS == SPEC_1::COLS);
        static_assert(SPEC_1::ROWS + SPEC_2::ROWS == SPEC_3::ROWS);
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        static_assert(SPEC_3::LAYOUT == RowMajor);
        // concatenate horizontally
        using TI = typename SPEC_1::TI;
        using T = typename SPEC_1::T;
        for(TI i = 0; i < SPEC_1::ROWS; i++){
            for(TI j = 0; j < SPEC_1::COLS; j++){
                C.data[i * SPEC_3::COLS + j] = A.data[i * SPEC_1::COLS + j];
            }
        }
        for(TI i = 0; i < SPEC_2::ROWS; i++){
            for(TI j = 0; j < SPEC_2::COLS; j++){
                C.data[(i + SPEC_1::ROWS) * SPEC_3::COLS + j] = B.data[i * SPEC_2::COLS + j];
            }
        }
    }


}
#endif
