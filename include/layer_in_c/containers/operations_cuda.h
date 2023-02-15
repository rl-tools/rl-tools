#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H

#include <layer_in_c/containers.h>
#include <layer_in_c/devices/cuda.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace layer_in_c{
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        /* for checking the pitch
        {
            size_t pitch;
            cudaMallocPitch(&matrix._data, &pitch, SPEC::COLS * sizeof(typename SPEC::T), SPEC::ROWS);

        }
        */
        auto result = cudaMalloc(&matrix._data, SPEC::SIZE_BYTES);
        check_status(device);

#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate container: " << cudaGetErrorString(result) << std::endl;
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        cudaFree(matrix._data);
        check_status(device);
    }
    namespace containers::cuda::internal {
        template<typename DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
        __global__ void
        copy(devices::CUDA<DEV_SPEC>& device, Matrix<TARGET_SPEC> target, Matrix<SOURCE_SPEC> source) {
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
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
//        static_assert(!TARGET_SPEC::IS_VIEW);
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(!TARGET_SPEC::IS_VIEW){
            Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename TARGET_SPEC::LAYOUT, false>> temp;
            malloc(source_device, temp);
            copy(source_device, source_device, temp, source);
            cudaMemcpy(target._data, temp._data, SPEC::SIZE_BYTES, cudaMemcpyHostToDevice);
            check_status(target_device);
            free(source_device, temp);
        }
        else{
            Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename SOURCE_SPEC::LAYOUT, false>> temp;
            malloc(target_device, temp);
            copy(target_device, source_device, temp, source);
            {
                constexpr TI BLOCKSIZE_COLS = 32;
                constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
                dim3 grid(N_BLOCKS_COLS);
                dim3 block(BLOCKSIZE_COLS);
                containers::cuda::internal::copy<<<grid, block>>>(target_device, target, temp);
                check_status(target_device);
            }
        }
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS>> temp_gpu, temp_cpu;
        malloc(source_device, temp_gpu);
        {
            constexpr TI BLOCKSIZE_COLS = 32;
            constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
            dim3 grid(N_BLOCKS_COLS);
            dim3 block(BLOCKSIZE_COLS);
            containers::cuda::internal::copy<<<grid, block>>>(source_device, temp_gpu, source);
            check_status(source_device);
        }
        malloc(target_device, temp_cpu);
        cudaMemcpy(temp_cpu._data, temp_gpu._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
        check_status(source_device);
        free(source_device, temp_gpu);
        copy(target_device, target_device, target, temp_cpu);
        free(target_device, temp_cpu);
    }

    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
//        static_assert(!TARGET_SPEC::IS_VIEW);
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
//        cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice);
//        check_status(source_device);
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        containers::cuda::internal::copy<<<grid, block>>>(target_device, target, source);
        check_status(target_device);
    }


    template<typename TARGET_DEV, typename SOURCE_DEV, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy_structure_mismatch(TARGET_DEV& target_device, SOURCE_DEV& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
//        static_assert(!TARGET_SPEC::IS_VIEW);
        using SPEC = TARGET_SPEC;
        using T = typename TARGET_SPEC::T;
        using TI = typename TARGET_SPEC::TI;
        using TARGET_DEV_CUDA = devices::CUDA<typename TARGET_DEV::SPEC>;
        using TARGET_DEV_CPU = devices::CPU<typename TARGET_DEV::SPEC>;
        using SOURCE_DEV_CUDA = devices::CUDA<typename SOURCE_DEV::SPEC>;
        using SOURCE_DEV_CPU = devices::CPU<typename SOURCE_DEV::SPEC>;
        static_assert(
                (utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CPU >) ||
                (utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CPU > && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>) ||
                (utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>)
        );
        constexpr bool TARGET_IS_VIEW = TARGET_SPEC::IS_VIEW;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);

        if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CPU>){
            if constexpr(!TARGET_IS_VIEW){
                Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename TARGET_SPEC::LAYOUT, false>> temp;
                malloc(source_device, temp);
                copy(source_device, source_device, temp, source);
                copy(target_device, source_device, target, temp);
                free(source_device, temp);
            }
            else{
                Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename SOURCE_SPEC::LAYOUT, false>> temp;
                malloc(target_device, temp);
                copy(target_device, source_device, temp, source);
                containers::cuda::internal::copy<<<bias_grid, bias_block>>>(target_device, target, temp);
                check_status(target_device);
            }
        }
        else{
            if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CPU> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>){
                // GPU (possible view) -> GPU (dense) -> CPU (dense) -> CPU (possible view)
                Matrix<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS>> temp_gpu, temp_cpu;
                malloc(source_device, temp_gpu);
                containers::cuda::internal::copy<<<bias_grid, bias_block>>>(source_device, temp_gpu, source);
                check_status(source_device);
                malloc(target_device, temp_cpu);
                copy(target_device, source_device, temp_cpu, temp_gpu);
                free(source_device, temp_gpu);
                copy(target_device, target_device, target, temp_cpu);
                free(target_device, temp_cpu);
            }
            else{
                if constexpr(utils::typing::is_same_v<TARGET_DEV, TARGET_DEV_CUDA> && utils::typing::is_same_v<SOURCE_DEV, SOURCE_DEV_CUDA>){
                    containers::cuda::internal::copy<<<bias_grid, bias_block>>>(target_device, target, source);
                    check_status(target_device);
                }
            }
        }

    }
}

#endif