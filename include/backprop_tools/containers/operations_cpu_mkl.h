#ifndef BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_MKL_H
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_MKL_H

#include <backprop_tools/containers.h>
#include "operations_cpu.h"
#include <backprop_tools/devices/cpu_mkl.h>

#include <mkl.h>

namespace backprop_tools{
#ifndef BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)mkl_malloc(SPEC::SIZE_BYTES, 64);
        count_malloc(device, SPEC::SIZE_BYTES);
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = math::nan<typename SPEC::T>(typename DEV_SPEC::MATH());
            }
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        mkl_free(matrix._data);
    }
#endif
    template<typename DEV_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(devices::CPU_MKL<DEV_SPEC>& device, const Matrix<INPUT_SPEC_A>& A, const Matrix<INPUT_SPEC_B>& B, Matrix<OUTPUT_SPEC>& output) {
        static_assert(INPUT_SPEC_A::ROWS == OUTPUT_SPEC::ROWS);
        static_assert(INPUT_SPEC_A::COLS == INPUT_SPEC_B::ROWS);
        static_assert(INPUT_SPEC_B::COLS == OUTPUT_SPEC::COLS);
        static_assert(INPUT_SPEC_A::COL_PITCH == 1); // dense row-major
        static_assert(INPUT_SPEC_B::COL_PITCH == 1); // dense row-major

        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEV_SPEC::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 0;
        constexpr auto m = OUTPUT_SPEC::ROWS;
        constexpr auto k = INPUT_SPEC_A::COLS;
        constexpr auto n = OUTPUT_SPEC::COLS;

        if constexpr(utils::typing::is_same_v<T, float>){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)A._data, row_pitch(A), (T*)B._data, row_pitch(B), beta, (T*)output._data, row_pitch(output));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)A._data, row_pitch(A), (T*)B._data, row_pitch(B), beta, (T*)output._data, row_pitch(output));
        }
    }
}
#endif
