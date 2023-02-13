#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H

#include <layer_in_c/containers.h>
#include <layer_in_c/devices/cuda.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        auto result = cudaMalloc(&matrix._data, SPEC::SIZE_BYTES);

#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate container: " << cudaGetErrorString(result) << std::endl;
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        cudaFree(matrix._data);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_memory_layout<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        using SPEC = SPEC_1;
        cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyHostToDevice);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_memory_layout<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        using SPEC = SPEC_1;
        cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
    }

    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_memory_layout<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        using SPEC = SPEC_1;
        cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice);
    }
    // todo: implement copy kernels for matrices with different memory layouts
}

#endif