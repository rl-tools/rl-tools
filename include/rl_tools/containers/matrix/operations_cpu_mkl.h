#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_OPERATIONS_CPU_MKL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_OPERATIONS_CPU_MKL_H

#include "../containers.h"
#include "operations_cpu_blas.h"
#include "../devices/cpu_mkl.h"

#include <mkl.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
#ifndef RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)mkl_malloc(SPEC::SIZE_BYTES, 64);
        count_malloc(device, SPEC::SIZE_BYTES);
#ifdef RL_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(utils::typing::is_same_v<typename SPEC::T, float> || utils::typing::is_same_v<typename SPEC::T, double>){
                matrix._data[i] = math::nan<typename SPEC::T>(typename DEV_SPEC::MATH());
            }
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        mkl_free(matrix._data);
    }
#endif
    template<typename DEV_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(devices::CPU_MKL<DEV_SPEC>& device, const Matrix<INPUT_SPEC_A>& A, const Matrix<INPUT_SPEC_B>& B, Matrix<OUTPUT_SPEC>& output) {
        multiply((devices::CPU_BLAS<DEV_SPEC>&)device, A, B, output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

#include "operations_cpu_blas.h"