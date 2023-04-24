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
                matrix._data[i] = 0.0/0.0;
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
}
#endif
