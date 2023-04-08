#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>
#ifndef LAYER_IN_C_FUNCTION_PLACEMENT
    #define LAYER_IN_C_FUNCTION_PLACEMENT
#endif

#if defined(LAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS) || defined(LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC)
#include <iostream>
#include <sstream>
#endif


namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, MatrixStatic<SPEC>& matrix){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)&matrix._data_memory[0];
#ifdef LAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = 0.0/0.0;
            }
        }
#endif
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, MatrixStatic<SPEC>& matrix){
        // free is a no-op for statically allocated matrices
    }

#ifndef LAYER_IN_C_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, MatrixDynamic<SPEC>& matrix){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)new char[SPEC::SIZE_BYTES];
        count_malloc(device, SPEC::SIZE_BYTES);

#ifdef LAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = 0.0/0.0;
            }
        }
#endif
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, MatrixDynamic<SPEC>& matrix){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        delete matrix._data;
        matrix._data = nullptr;
    }
#endif

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
#if defined(LAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS)
        if(row >= SPEC::ROWS || col >= SPEC::COLS){
#if !defined(__CUDA_ARCH__)
            std::stringstream ss;
            ss << "index: " << row << "(" << SPEC::ROWS << "):" << col << "(" << SPEC::COLS << ") out of bounds";
            throw std::runtime_error(ss.str());
#else
            printf("index: %d(%d):%d(%d) out of bounds", row, SPEC::ROWS, col, SPEC::COLS);
#endif
        }
#endif
        return index;
    }
    template<typename SPEC>
    // todo: return reference
    LAYER_IN_C_FUNCTION_PLACEMENT inline typename SPEC::T& get(const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
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
    template<typename SPEC, typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT inline void multiply(Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col, T value){
        m._data[index(m, row, col)] *= value;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void transpose(DEVICE& device, Matrix<SPEC_1>& target, Matrix<SPEC_2>& source){
        static_assert(SPEC_1::ROWS == SPEC_2::COLS);
        static_assert(SPEC_1::COLS == SPEC_2::ROWS);
        for(typename SPEC_1::TI i = 0; i < SPEC_1::ROWS; i++){
            for(typename SPEC_1::TI j = 0; j < SPEC_1::COLS; j++){
                set(target, i, j, get(source, j, i));
            }
        }
    }
    namespace containers::vectorization::operators{
        template<typename DEVICE, typename T>
        inline T copy(DEVICE dev, T b){
            return b;
        }
        template<typename DEVICE, typename T>
        inline T add(DEVICE dev, T a, T b){
            return a+b;
        }
        template<typename DEVICE, typename T>
        inline T sub(DEVICE dev, T a, T b){
            return a-b;
        }
        template<typename DEVICE, typename T>
        inline bool is_nan(DEVICE dev, bool a, T c){
            return a || math::is_nan(dev, c);
        }
        template<typename DEVICE, typename T>
        inline bool is_finite(DEVICE dev, bool a, T c){
            return a && math::is_finite(dev, c);
        }
    }
    template<typename DEVICE, typename SPEC>
    auto view_transpose(DEVICE& device, Matrix<SPEC>& target){
//        static_assert(SPEC::ROWS == SPEC::COLS);
        using TI = typename SPEC::TI;
//        for(TI i = 0; i < SPEC::ROWS; i++){
//            for(TI j = i + 1; j < SPEC::COLS; j++){
//                T temp = get(target, i,  j);
//                set(target,  i,  j, get(target,  j,  i));
//                set(target,  j,  i, temp);
//            }
//        }
        using LayOut = matrix::layouts::Fixed<TI, SPEC::COL_PITCH, SPEC::ROW_PITCH>;
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::COLS, SPEC::ROWS, LayOut>> out;
        out._data = target._data;
        return out;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, const Matrix<SPEC_1>& m1, const Matrix<SPEC_2>& m2){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        using T = typename SPEC::T;
        typename SPEC::T acc = 0;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                T v1 = get(m1, i, j);
                T v2 = get(m2, i, j);
                acc += math::abs(typename DEVICE::SPEC::MATH(), v1 - v2);
            }
        }
        return acc;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3, auto BINARY_OPERATOR>
    void vectorize_binary(DEVICE& device, const Matrix<SPEC_1>& a, const Matrix<SPEC_2>& b, Matrix<SPEC_3>& c){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(containers::check_structure<SPEC_2, SPEC_3>);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(c, i, j, BINARY_OPERATOR(get(a, i, j), get(b, i, j)));
            }
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, auto UNARY_OPERATOR>
    LAYER_IN_C_FUNCTION_PLACEMENT void vectorize_unary(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                set(target, i, j, UNARY_OPERATOR(typename DEVICE::SPEC::MATH(), get(source, i, j)));
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename RETURN_TYPE, auto BINARY_OPERATOR>
    LAYER_IN_C_FUNCTION_PLACEMENT RETURN_TYPE reduce_unary(DEVICE device, const Matrix<SPEC>& source, const RETURN_TYPE& init){
        using TI = typename SPEC::TI;
        RETURN_TYPE acc = init;
        for(TI row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(TI col_i = 0; col_i < SPEC::COLS; col_i++){
                acc = BINARY_OPERATOR(typename DEVICE::SPEC::MATH(), acc, get(source, row_i, col_i));
            }
        }
        return acc;
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename RETURN_TYPE, auto TERTIARY_OPERATOR>
    LAYER_IN_C_FUNCTION_PLACEMENT RETURN_TYPE reduce_binary(DEVICE& device, Matrix<SPEC_1>& source_1, Matrix<SPEC_2>& source_2, const RETURN_TYPE& init){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using TI = typename SPEC_1::TI;
        RETURN_TYPE acc = init;
        for(TI row_i = 0; row_i < SPEC_1::ROWS; row_i++){
            for(TI col_i = 0; col_i < SPEC_1::COLS; col_i++){
                acc = TERTIARY_OPERATOR(typename DEVICE::SPEC::MATH(), acc, get(source_1, row_i, col_i), get(source_2, row_i, col_i));
            }
        }
        return acc;
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    void add(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_binary<DEVICE, SPEC_1, SPEC_2, SPEC_1, containers::vectorization::operators::add<typename SPEC::T>>(device, target, source, target);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2, typename SPEC_3>
    void sub(DEVICE& device, const Matrix<SPEC_1>& a, const Matrix<SPEC_2>& b, Matrix<SPEC_3>& c){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(containers::check_structure<SPEC_2, SPEC_3>);
        using SPEC = SPEC_1;
        vectorize_binary<DEVICE, SPEC_1, SPEC_2, SPEC_3, containers::vectorization::operators::sub<typename SPEC::T>>(device, a, b, c);
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
    template<typename DEVICE, typename SPEC, typename VALUE_T>
    void increment_all(DEVICE& device, Matrix<SPEC>& m, VALUE_T value){
        for(typename SPEC::TI i = 0; i < SPEC::ROWS; i++){
            for(typename SPEC::TI j = 0; j < SPEC::COLS; j++){
                increment(m, i, j, value);
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
    template<typename DEVICE, typename SPEC_1, typename SPEC_2, bool BOUNDS_CHECKING=true>
    void slice(DEVICE& device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source, typename SPEC_1::TI row, typename SPEC_1::TI col, typename SPEC_1::TI rows = SPEC_1::ROWS, typename SPEC_1::TI cols = SPEC_1::COLS, typename SPEC_1::TI target_row=0, typename SPEC_1::TI target_col=0){
//        static_assert(SPEC_1::ROWS <= SPEC_2::ROWS);
//        static_assert(SPEC_1::COLS <= SPEC_2::COLS);
        using TI = typename SPEC_1::TI;
        if constexpr(BOUNDS_CHECKING){
            utils::assert_exit(device, row + rows <= SPEC_2::ROWS, "row + rows <= SPEC_2::ROWS");
            utils::assert_exit(device, col + cols <= SPEC_2::COLS, "col + cols <= SPEC_2::COLS");
            utils::assert_exit(device, target_row + rows <= SPEC_1::ROWS, "target_row + rows <= SPEC_1::ROWS");
            utils::assert_exit(device, target_col + cols <= SPEC_1::COLS, "target_col + cols <= SPEC_1::COLS");
        }
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
        return reduce_unary<DEVICE, SPEC, bool, containers::vectorization::operators::is_nan<typename DEVICE::SPEC::MATH, typename SPEC::T>>(device, m, false);
    }
    template<typename DEVICE, typename SPEC>
    bool is_finite(DEVICE& device, const Matrix<SPEC>& m){
        return reduce_unary<DEVICE, SPEC, bool, containers::vectorization::operators::is_finite<typename DEVICE::SPEC::MATH, typename SPEC::T>>(device, m, true);
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

    template<typename TARGET_DEVICE, typename SPEC, typename T>
    void assign(TARGET_DEVICE& source_device, T* target, Matrix<SPEC>& source, typename SPEC::TI row = 0, typename SPEC::TI col = 0, typename SPEC::TI rows = SPEC::ROWS, typename SPEC::TI cols = SPEC::COLS, typename SPEC::TI row_pitch = SPEC::COLS, typename SPEC::TI col_pitch = 1){
        using TI = typename SPEC::TI;
        utils::assert_exit(source_device, row + rows <= SPEC::ROWS, "row + rows <= SPEC::ROWS");
        utils::assert_exit(source_device, col + cols <= SPEC::COLS, "col + cols <= SPEC::COLS");
        for(TI i = 0; i < rows; i++){
            for(TI j = 0; j < cols; j++){
//                set(target, row + i, col+j, source[i * row_pitch + j * col_pitch]);
                target[i * cols + j] = get(source, row + i, col+j);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename SPEC::TI ROWS, typename SPEC::TI COLS>
    LAYER_IN_C_FUNCTION_PLACEMENT auto view(DEVICE& device, const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        static_assert(SPEC::ROWS >= ROWS);
        static_assert(SPEC::COLS >= COLS);
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, (row + ROWS) <= SPEC::ROWS, "row + ROWS <= SPEC::ROWS");
        utils::assert_exit(device, (col + COLS) <= SPEC::COLS, "col + COLS <= SPEC::COLS");
#endif
        using ViewLayout = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
        MatrixDynamic<matrix::Specification<typename SPEC::T, typename SPEC::TI, ROWS, COLS, ViewLayout, true>> out;
        out._data = m._data + row * row_pitch(m) + col * col_pitch(m);
        return out;
    }
    template<typename DEVICE, typename SPEC, typename ViewSpec>
    LAYER_IN_C_FUNCTION_PLACEMENT auto view(DEVICE& device, const Matrix<SPEC>& m, const ViewSpec& vs, typename SPEC::TI row, typename SPEC::TI col){
        return view<DEVICE, SPEC, ViewSpec::ROWS, ViewSpec::COLS>(device, m, row, col);
    }

    template<typename DEVICE, typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT auto row(DEVICE& device, const Matrix<SPEC>& m, typename SPEC::TI row){
        using ViewLayout = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, 1, SPEC::COLS, ViewLayout, true>> out;
        out._data = m._data + row * row_pitch(m);
        return out;
    }

    template<typename DEVICE, typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT auto col(DEVICE& device, const Matrix<SPEC>& m, typename SPEC::TI col){
        using ViewLayout = matrix::layouts::Fixed<typename SPEC::TI, SPEC::ROW_PITCH, SPEC::COL_PITCH>;
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ROWS, 1, ViewLayout, true>> out;
        out._data = m._data + col * col_pitch(m);
        return out;
    }

    template <typename DEVICE, typename T, typename INPUT_SPEC, typename MEAN_SPEC, typename STD_SPEC, typename OUTPUT_SPEC>
    void standardise(DEVICE& device, const layer_in_c::Matrix<INPUT_SPEC>& input, const layer_in_c::Matrix<MEAN_SPEC> mean, const layer_in_c::Matrix<STD_SPEC> std, layer_in_c::Matrix<OUTPUT_SPEC> output){
        static_assert(layer_in_c::containers::check_structure<INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(layer_in_c::containers::check_structure<MEAN_SPEC, STD_SPEC>);
        static_assert(INPUT_SPEC::COLS == MEAN_SPEC::COLS);
        static_assert(MEAN_SPEC::ROWS == 1);
        for(typename DEVICE::index_t row_i = 0; row_i < INPUT_SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < INPUT_SPEC::COLS; col_i++){
                set(output, row_i, col_i, (get(input, row_i, col_i) - get(mean, 0, col_i)) / get(std, 0, col_i));
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void randn(DEVICE& device, layer_in_c::Matrix<SPEC>& m, RNG& rng){
        using T = typename SPEC::T;
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                set(m, row_i, col_i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, (T)1, rng));
            }
        }
    }
    template <typename DEVICE, typename SPEC>
    typename SPEC::T mean(DEVICE& device, layer_in_c::Matrix<SPEC>& m){
        using T = typename SPEC::T;
        T acc = 0;
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                acc += get(m, row_i, col_i);
            }
        }
        return acc/(SPEC::ROWS * SPEC::COLS);
    }
    template <typename DEVICE, typename SPEC>
    typename SPEC::T std(DEVICE& device, layer_in_c::Matrix<SPEC>& m){
        using T = typename SPEC::T;
        T acc = 0;
        T avg = mean(device, m);
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                T diff = get(m, row_i, col_i) - avg;
                acc += diff * diff;
            }
        }
        return math::sqrt(typename DEVICE::SPEC::MATH(), acc/(SPEC::ROWS * SPEC::COLS));
    }

    template <typename DEVICE, typename T, typename DEVICE::index_t DIM>
    Matrix<matrix::Specification<T, typename DEVICE::index_t, 1, DIM, matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>> wrap(DEVICE& dev, T* data){
        return {data};
    }

    template <typename DEVICE, typename SPEC>
    void clamp(DEVICE& device, layer_in_c::Matrix<SPEC>& m, typename SPEC::T lower, typename SPEC::T upper){
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                set(m, row_i, col_i, math::clamp<typename SPEC::T>(typename DEVICE::SPEC::MATH(), get(m, row_i, col_i), lower, upper));
            }
        }
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    void swap(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b){
        using T = typename SPEC_A::T;
        using TI = typename DEVICE::index_t;
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++){
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++){
                T tmp = get(a, row_i, col_i);
                set(a, row_i, col_i, get(b, row_i, col_i));
                set(b, row_i, col_i, tmp);
            }
        }
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    void swap(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, typename DEVICE::index_t row_a, typename DEVICE::index_t col_a, typename DEVICE::index_t row_b, typename DEVICE::index_t col_b){
        using T = typename SPEC_A::T;
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        T tmp = get(a, row_a, col_a);
        set(a, row_a, col_a, get(b, row_b, col_b));
        set(b, row_b, col_b, tmp);
    }

    template <typename DEVICE, typename MEAN_SPEC, typename STD_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void normalize(DEVICE& device, Matrix<MEAN_SPEC>& mean, Matrix<STD_SPEC>& std, Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        static_assert(containers::check_structure<MEAN_SPEC, STD_SPEC>);
        static_assert(containers::check_structure<INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(MEAN_SPEC::ROWS == 1);
        static_assert(MEAN_SPEC::COLS == INPUT_SPEC::COLS);

        using T = typename INPUT_SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI DATA_SIZE = INPUT_SPEC::ROWS;
        constexpr TI DIM = INPUT_SPEC::COLS;
        for(TI row_i = 0; row_i < DATA_SIZE; row_i++){
            for(TI col_i = 0; col_i < DIM; col_i++){
                T x = get(input, row_i, col_i);
                T mu = get(mean, 0, col_i);
                T sigma = get(std, 0, col_i);
                T normalized_x = (x - mu) / sigma;
                set(output, row_i, col_i, normalized_x);
            }
        }
    }
}
#endif
