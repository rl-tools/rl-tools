#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CUBLAS_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CUBLAS_H

#include <layer_in_c/containers.h>
#include "operations_cpu.h"
#include <layer_in_c/devices/cublas.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUBLAS<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        cudaMalloc(&matrix.data, SPEC::SIZE_BYTES);
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CUBLAS<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        cudaFree(matrix.data);
    }
}
#endif
