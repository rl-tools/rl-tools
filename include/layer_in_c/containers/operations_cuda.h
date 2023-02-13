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

    template<typename DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    __global__ void
    copy_structure_mismatch_kernel(devices::CUDA<DEV_SPEC>& device, Matrix<TARGET_SPEC> target, Matrix<SOURCE_SPEC> source) {
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        using T = typename TARGET_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        TI col = blockIdx.x * blockDim.x + threadIdx.x;
        if(col < TARGET_SPEC::COLS){
            for(TI row = 0; row < TARGET_SPEC::ROWS; row++){
                set(target, row, col, get(source, row, col));
            }
        }
    }

    template<typename TARGET_DEV, typename SOURCE_DEV, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy_structure_mismatch(TARGET_DEV& target_device, SOURCE_DEV& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        using SPEC = TARGET_SPEC;
        using T = typename TARGET_SPEC::T;
        using TI = typename TARGET_SPEC::TI;
        using TARGET_DEV_CUDA = devices::CUDA<typename TARGET_DEV::SPEC>;
        using TARGET_DEV_CPU = devices::CPU<typename TARGET_DEV::SPEC>;
        using SOURCE_DEV_CUDA = devices::CUDA<typename SOURCE_DEV::SPEC>;
        using SOURCE_DEV_CPU = devices::CPU<typename SOURCE_DEV::SPEC>;

        if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CPU>){
            Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename TARGET_SPEC::LAYOUT>> temp;
            malloc(source_device, temp);
            copy(source_device, source_device, temp, source);
            copy(target_device, source_device, target, temp);
            free(source_device, temp);
        }
        else{
            if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CPU> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>){
                Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename SOURCE_SPEC::LAYOUT>> temp;
                malloc(target_device, temp);
                copy(target_device, source_device, temp, source);
                copy(target_device, target_device, target, temp);
                free(target_device, temp);
            }
            else{
                if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>){
                    constexpr TI BLOCKSIZE_COLS = 32;
                    constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
                    dim3 bias_grid(N_BLOCKS_COLS);
                    dim3 bias_block(BLOCKSIZE_COLS);
                    copy_structure_mismatch_kernel<<<bias_grid, bias_block>>>(target_device, target, source);
                }
            }
        }

    }
}

#endif