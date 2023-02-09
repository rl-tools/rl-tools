#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>
#ifndef LAYER_IN_C_FUNCTION_PLACEMENT
    #define LAYER_IN_C_FUNCTION_PLACEMENT
#endif

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, Matrix<SPEC>& matrix){
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
        matrix._data = (typename SPEC::T*)new char[SPEC::SIZE_BYTES];
        // for debugging, initializing to NaN
//        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
//            matrix._data[i] = 0.0/0.0;
//        }
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, Matrix<SPEC>& matrix){
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
        delete matrix._data;
        matrix._data = nullptr;
    }

    template<typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT typename SPEC::TI row_pitch(const Matrix<SPEC>& m){
        return SPEC::ROW_PITCH;
    }
    template<typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT typename SPEC::TI col_pitch(const Matrix<SPEC>& m){
        return SPEC::COL_PITCH;
    }

    template<typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT inline typename SPEC::TI index(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        typename SPEC::TI index = row * row_pitch(m) + col * col_pitch(m);
        // bounds checking for debugging
//        if(row >= SPEC::ROWS || col >= SPEC::COLS){
//            std::cout << "index: " << row << "(" << SPEC::ROWS << "):" << col << "(" << SPEC::COLS << ") out of bounds" << std::endl;
//        }
        return index;
    }
    template<typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT inline typename SPEC::T get(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        return m._data[index(m, row, col)];
    }
    template<typename SPEC, typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT inline void set(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] = value;
    }
    template<typename SPEC, typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT inline void increment(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] += value;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void transpose(DEVICE& device, Matrix<SPEC_1>& target, Matrix<SPEC_2>& source){
        static_assert(SPEC_1::ROWS == SPEC_2::COLS);
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        for(typename SPEC_1::TI i = 0; i < SPEC_1::ROWS; i++){
            for(typename SPEC_1::TI j = 0; j < SPEC_1::COLS; j++){
                get(target,  i,  j) = get(source,  j,  i);
            }
        }
    }
    namespace containers::vectorization::operators{
        template<typename T>
        inline T copy(T b){
            return b;
        }
        template<typename T>
        inline T add(T a, T b){
            return a+b;
        }
    }
    template<typename DEVICE, typename SPEC>
    auto transpose(DEVICE& device, Matrix<SPEC>& target){
        static_assert(SPEC::ROWS == SPEC::COLS);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = i + 1; j < SPEC::COLS; j++){
                T temp = get(target, i,  j);
                set(target,  i,  j, get(target,  j,  i));
                set(target,  j,  i, temp);
            }
        }
        auto data = target._data;
        target._data = nullptr;
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::COLS, SPEC::ROWS>> out;
        out._data = data;
        return out;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const Matrix<SPEC_1>& m1, const Matrix<SPEC_2>& m2){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        typename SPEC::T acc = 0;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                acc += math::abs(get(m1, i, j) - get(m2, i, j));
            }
        }
        return acc;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto BINARY_OPERATOR>
    void vectorize_binary(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(target, i, j, BINARY_OPERATOR(get(target, i, j), get(source, i, j)));
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto UNARY_OPERATOR>
    void vectorize_unary(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(target, i, j, UNARY_OPERATOR(get(source, i, j)));
            }
        }
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename SPEC_1, typename SPEC_2>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
//        static_assert(utils::typing::is_same<TARGET_DEVICE, SOURCE_DEVICE>); // todo: implement
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<TARGET_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename SPEC::T>>(target_device, target, source);
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
                get(target, i, j) += get(source, 0, j);
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void set_broadcast(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(SPEC_1::COLS == SPEC_2::COLS);
        static_assert(SPEC_2::ROWS == 1);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(target, i, j, get(source, 0, j));
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename VALUE_T>
    void set_all(DEVICE& device, Matrix<SPEC>& m, VALUE_T value){
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(m, i, j, value);
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
                    acc += get(A, i, k) * get(B, k, j);
                }
                get(C, i, j) = acc;
            }
        }
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_C>
    void hcat(DEVICE& device, const Matrix<SPEC_A>& A, const Matrix<SPEC_B>& B, Matrix<SPEC_C>& C){
        static_assert(SPEC_A::ROWS == SPEC_B::ROWS);
        static_assert(SPEC_C::ROWS == SPEC_A::ROWS);
        static_assert(SPEC_A::COLS + SPEC_B::COLS == SPEC_C::COLS);
        // concatenate horizontally
        using TI = typename SPEC_A::TI;
        using T = typename SPEC_A::T;
        for(TI i = 0; i < SPEC_A::ROWS; i++){
            for(TI j = 0; j < SPEC_A::COLS; j++){
                set(C, i, j, get(A, i, j));
            }
            for(TI j = 0; j < SPEC_B::COLS; j++){
                set(C, i, (j + SPEC_A::COLS), get(B, i, j));
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
                set(C, i, j, get(A, i, j));
            }
        }
        for(TI i = 0; i < SPEC_B::ROWS; i++){
            for(TI j = 0; j < SPEC_B::COLS; j++){
                set(C, i + SPEC_A::ROWS, j, get(B, i, j));
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void slice(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source, typename SPEC_1::TI row, typename SPEC_1::TI col, typename SPEC_1::TI rows = SPEC_1::ROWS, typename SPEC_1::TI cols = SPEC_1::COLS, typename SPEC_1::TI target_row=0, typename SPEC_1::TI target_col=0){
//        static_assert(SPEC_1::ROWS <= SPEC_2::ROWS);
//        static_assert(SPEC_1::COLS <= SPEC_2::COLS);
        using TI = typename SPEC_1::TI;
        using T = typename SPEC_1::T;
        utils::assert_exit(device, row + rows <= SPEC_2::ROWS, "row + rows <= SPEC_2::ROWS");
        utils::assert_exit(device, col + cols <= SPEC_2::COLS, "col + cols <= SPEC_2::COLS");
        utils::assert_exit(device, target_row + rows <= SPEC_1::ROWS, "target_row + rows <= SPEC_1::ROWS");
        utils::assert_exit(device, target_col + cols <= SPEC_1::COLS, "target_col + cols <= SPEC_1::COLS");
        for(TI i = 0; i < rows; i++){
            for(TI j = 0; j < cols; j++){
                set(target, target_row + i, target_col + j, get(source, row + i, col + j));
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
                acc += get(m, i, j);
            }
        }
        return acc;
    }
    template<typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const Matrix<SPEC>& m){
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        for(TI i = 0; i < SPEC::ROWS; i++){
            for(TI j = 0; j < SPEC::COLS; j++){
                if(math::is_nan(typename DEVICE::SPEC::MATH(), get(m, i, j))){
                    return true;
                }
            }
        }
        return false;
    }
    template<typename TARGET_DEVICE, typename SPEC, typename T>
    void assign(TARGET_DEVICE& target_device, Matrix<SPEC>& target, const T* source, typename SPEC::TI row = 0, typename SPEC::TI col = 0, typename SPEC::TI rows = SPEC::ROWS, typename SPEC::TI cols = SPEC::COLS, typename SPEC::TI row_pitch = SPEC::COLS, typename SPEC::TI col_pitch = 1){
        using TI = typename SPEC::TI;
        utils::assert_exit(target_device, row + rows <= SPEC::ROWS, "row + rows <= SPEC::ROWS");
        utils::assert_exit(target_device, col + cols <= SPEC::COLS, "col + cols <= SPEC::COLS");
        for(TI i = 0; i < rows; i++){
            for(TI j = 0; j < cols; j++){
                set(target, row + i, col+j, source[i * row_pitch + j * col_pitch]);
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename SPEC::TI ROWS, typename SPEC::TI COLS>
    auto view(DEVICE& device, Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        static_assert(SPEC::ROWS >= ROWS);
        static_assert(SPEC::COLS >= COLS);
        using ViewLayout = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, ROWS, COLS, ViewLayout>> out;
        out._data = m._data + row * row_pitch(m) + col * col_pitch(m);
        return out;
    }


}
#endif
