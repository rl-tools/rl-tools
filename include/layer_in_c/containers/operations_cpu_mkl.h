#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_MKL_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_MKL_H

#include <layer_in_c/containers.h>
#include "operations_cpu.h"
#include <layer_in_c/devices/cpu_mkl.h>

#include <mkl.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CPU_MKL<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        matrix.data = (typename SPEC::T*)mkl_malloc(SPEC::SIZE_BYTES, 64);
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CPU_MKL<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        mkl_free(matrix.data);
    }
}
#endif
