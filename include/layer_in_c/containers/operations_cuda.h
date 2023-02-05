#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H

#include <layer_in_c/containers.h>
#include <layer_in_c/devices/cuda.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        auto result = cudaMalloc(&matrix.data, SPEC::ROWS * SPEC::COLS *sizeof(typename SPEC::T));

        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate container: " << cudaGetErrorString(result) << std::endl;
        }
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        cudaFree(matrix.data);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        // todo: implement second copy that allows for different types
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        cudaMemcpy(target.data, source.data, SPEC::ROWS * SPEC::COLS * sizeof(typename SPEC::T), cudaMemcpyHostToDevice);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        // todo: implement second copy that allows for different types
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        cudaMemcpy(target.data, source.data, SPEC::ROWS * SPEC::COLS * sizeof(typename SPEC::T), cudaMemcpyDeviceToHost);
    }

    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, const Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        static_assert(utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>);
        // todo: implement second copy that allows for different types
        static_assert(SPEC_1::LAYOUT == RowMajor);
        static_assert(SPEC_2::LAYOUT == RowMajor);
        using SPEC = SPEC_1;
        cudaMemcpy(target.data, source.data, SPEC::ROWS * SPEC::COLS * sizeof(typename SPEC::T), cudaMemcpyDeviceToDevice);
    }
}

#endif