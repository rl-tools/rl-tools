#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_MKL_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_MKL_H

#include <layer_in_c/containers.h>
#include "operations_cpu.h"
#include <layer_in_c/devices/cpu_mkl.h>

#include <mkl.h>

namespace layer_in_c{
#ifndef LAYER_IN_C_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        matrix._data = (typename SPEC::T*)mkl_malloc(SPEC::SIZE_BYTES, 64);
        count_malloc(device, SPEC::SIZE_BYTES);
#ifdef LAYER_IN_C_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(std::is_convertible<typename SPEC::T, float>::value){
                matrix._data[i] = 0.0/0.0;
            }
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CPU_MKL<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        mkl_free(matrix._data);
    }
#endif
}
#endif
