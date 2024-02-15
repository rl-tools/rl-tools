#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_OPERATIONS_CPU_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_OPERATIONS_CPU_BLAS_H

#include "operations_cpu.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(devices::CPU_BLAS<DEV_SPEC>& device, const Matrix<INPUT_SPEC_A>& A, const Matrix<INPUT_SPEC_B>& B, Matrix<OUTPUT_SPEC>& output) {
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A._data, row_pitch(A), B._data, row_pitch(B), beta, output._data, row_pitch(output));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A._data, row_pitch(A), B._data, row_pitch(B), beta, output._data, row_pitch(output));
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif