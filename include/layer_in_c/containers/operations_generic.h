#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, Matrix<SPEC>& matrix){
        utils::assert_exit(device, matrix.data == nullptr, "Matrix is already allocated");
        matrix.data = (typename SPEC::T*)std::malloc(SPEC::SIZE_BYTES);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, Matrix<SPEC>& matrix){
        utils::assert_exit(device, matrix.data != nullptr, "Matrix has not been allocated");
        std::free(matrix.data);
        matrix.data = nullptr;
    }

    template<typename SPEC>
    inline typename SPEC::TI row_pitch(const Matrix<SPEC>& m){
        return SPEC::ROW_PITCH;
    }
    template<typename SPEC>
    inline typename SPEC::TI col_pitch(const Matrix<SPEC>& m){
        return SPEC::COL_PITCH;
    }

    template<typename SPEC>
    inline typename SPEC::TI index(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        return SPEC::ROW_PITCH * row + SPEC::COL_PITCH * col;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void transpose(DEVICE& device, Matrix<SPEC_1>& target, Matrix<SPEC_2>& source){
        static_assert(SPEC_1::ROWS == SPEC_2::COLS);
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        for(typename SPEC_1::TI i = 0; i < SPEC_1::ROWS; i++){
            for(typename SPEC_1::TI j = 0; j < SPEC_1::COLS; j++){
                target.data[index(target, i, j)] = source.data[index(source, j, i)];
            }
        }
    }
    namespace containers::vectorization::operators{
        template<typename T>
        T copy(T a, T b){
            return b;
        }
        template<typename T>
        T add(T a, T b){
            return a+b;
        }
    }
    template<typename DEVICE, typename SPEC>
    Matrix<MatrixSpecification<typename SPEC::T, typename SPEC::TI, SPEC::COLS, SPEC::ROWS>> transpose(DEVICE& device, Matrix<SPEC>& target){
        static_assert(SPEC::ROWS == SPEC::COLS);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = i + 1; j < SPEC::COLS; j++){
                T temp = target.data[index(target, i, j)];
                target.data[index(target, i, j)] = target.data[index(target, j, i)];
                target.data[index(target, j, i)] = temp;
            }
        }
        auto data = target.data;
        target.data = nullptr;
        return {data};
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const Matrix<SPEC_1>& m1, const Matrix<SPEC_2>& m2){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        typename SPEC::T acc = 0;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                acc += math::abs(m1.data[index(m1, i, j)] - m2.data[index(m2, i, j)]);
            }
        }
        return acc;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_OPERATOR>
    void vectorize_binary(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                target.data[index(target, i, j)] = BINARY_OPERATOR(target.data[index(target, i, j)], source.data[index(source, i, j)]);
            }
        }
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename SPEC_1, typename SPEC_2>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
//        static_assert(utils::typing::is_same<TARGET_DEVICE, SOURCE_DEVICE>); // todo: implement
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_binary<TARGET_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename SPEC::T>>(target_device, target, source);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_binary<DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::add<typename SPEC::T>>(device, target, source);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add_broadcast(DEVICE& device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(SPEC_1::COLS == SPEC_2::COLS);
        static_assert(SPEC_2::ROWS == 1);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                target.data[index(target, i, j)] += source.data[index(source, 0, j)];
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
                target.data[index(target, i, j)] = source.data[index(source, 0, j)];
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename VALUE_T>
    void set(DEVICE& device, const Matrix<SPEC>& m, VALUE_T value){
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                m.data[index(m, i, j)] = value;
            }
        }
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3>
    void mul(DEVICE& device, const Matrix<SPEC_1>& A, const Matrix<SPEC_2>& B, const Matrix<SPEC_3>& C){
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        static_assert(SPEC_1::ROWS == SPEC_3::ROWS);
        static_assert(SPEC_2::COLS == SPEC_3::COLS);
        using SPEC = SPEC_1;
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = 0; j < SPEC::COLS; j++){
                T acc = 0;
                for(TI k = 0; k < SPEC::COLS; k++){
                    acc += A.data[index(A, i, k)] * B.data[index(B, k, j)];
                }
                C.data[index(C, i, j)] = acc;
            }
        }
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_C>
    void hcat(DEVICE& device, const Matrix<SPEC_A>& A, const Matrix<SPEC_B>& B, const Matrix<SPEC_C>& C){
        static_assert(SPEC_A::ROWS == SPEC_B::ROWS);
        static_assert(SPEC_C::ROWS == SPEC_A::ROWS);
        static_assert(SPEC_A::COLS + SPEC_B::COLS == SPEC_C::COLS);
        // concatenate horizontally
        using TI = typename SPEC_A::TI;
        using T = typename SPEC_A::T;
        for(TI i = 0; i < SPEC_A::ROWS; i++){
            for(TI j = 0; j < SPEC_A::COLS; j++){
                C.data[index(C, i, j)] = A.data[index(A, i, j)];
            }
            for(TI j = 0; j < SPEC_B::COLS; j++){
                C.data[index(C, i, (j + SPEC_A::COLS))] = B.data[index(B, i, j)];
            }
        }
    }
    // vcat
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_C>
    void vcat(DEVICE& device, const Matrix<SPEC_A>& A, const Matrix<SPEC_B>& B, const Matrix<SPEC_C>& C){
        static_assert(SPEC_A::COLS == SPEC_B::COLS);
        static_assert(SPEC_C::COLS == SPEC_A::COLS);
        static_assert(SPEC_A::ROWS + SPEC_B::ROWS == SPEC_C::ROWS);
        // concatenate horizontally
        using TI = typename SPEC_A::TI;
        using T = typename SPEC_A::T;
        for(TI i = 0; i < SPEC_A::ROWS; i++){
            for(TI j = 0; j < SPEC_A::COLS; j++){
                C.data[index(C, i, j)] = A.data[index(A, i, j)];
            }
        }
        for(TI i = 0; i < SPEC_B::ROWS; i++){
            for(TI j = 0; j < SPEC_B::COLS; j++){
                C.data[index(C, i + SPEC_A::ROWS, j)] = B.data[index(B, i, j)];
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void slice(DEVICE& device, const Matrix<SPEC_1>& target, Matrix<SPEC_2>& source, typename SPEC_1::TI row, typename SPEC_1::TI col){
        static_assert(SPEC_1::ROWS <= SPEC_2::ROWS);
        static_assert(SPEC_1::COLS <= SPEC_2::COLS);
        using TI = typename SPEC_1::TI;
        using T = typename SPEC_1::T;
        for(TI i = 0; i < SPEC_1::ROWS; i++){
            for(TI j = 0; j < SPEC_1::COLS; j++){
                target.data[index(target, i, j)] = source.data[index(source, i + row, col + j)];
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    typename SPEC::T sum(DEVICE& device, const Matrix<SPEC>& m){
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        T acc = 0;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = 0; j < SPEC::COLS; j++){
                acc += m.data[index(m, i, j)];
            }
        }
        return acc;
    }

}
#endif
