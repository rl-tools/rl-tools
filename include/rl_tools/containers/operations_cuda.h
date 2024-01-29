#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_OPERATIONS_CUDA_H

#include "../containers.h"
#include "../devices/cuda.h"

#include <cuda_runtime.h>
#include <cuda.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
#ifndef RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
        /* for checking the pitch
        {
            size_t pitch;
            cudaMallocPitch(&matrix._data, &pitch, SPEC::COLS * sizeof(typename SPEC::T), SPEC::ROWS);

        }
        */
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        auto result = cudaMalloc(&matrix._data, SPEC::SIZE_BYTES);
        check_status(device);
        count_malloc(device, SPEC::SIZE_BYTES);

#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate container: " << cudaGetErrorString(result) << std::endl;
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CUDA<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
        cudaFree(matrix._data);
        check_status(device);
    }
#endif
    namespace containers::cuda::kernels {
        template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        __global__ void
        copy(const Matrix<SOURCE_SPEC> source, Matrix<TARGET_SPEC> target) {
            static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
            using T = typename TARGET_SPEC::T;
            using TI = typename DEVICE::index_t;

            TI col = blockIdx.x * blockDim.x + threadIdx.x;
            if(col < TARGET_SPEC::COLS){
                for(TI row = 0; row < TARGET_SPEC::ROWS; row++){
                    set(target, row, col, get(source, row, col));
                }
            }
        }
        template<typename DEVICE, typename SPEC, typename VALUE_T>
        __global__
        void set_all(Matrix<SPEC> m, VALUE_T value){
            using TI = typename DEVICE::index_t;
            TI col = blockIdx.x * blockDim.x + threadIdx.x;
            if(col < SPEC::COLS){
                for(typename SPEC::TI row = 0; row < SPEC::ROWS; row++){
                    set(m, row, col, value);
                }
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename RNG>
        __global__
        void randn(devices::CUDA<DEV_SPEC> device, Matrix<SPEC> m, typename SPEC::T mean, typename SPEC::T std, RNG rng){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            TI col = blockIdx.x * blockDim.x + threadIdx.x;
            curandState rng_state;
            curand_init(rng, col, 0, &rng_state);
            if(col < SPEC::COLS){
                for(TI row = 0; row < SPEC::ROWS; row++){
                    T sample = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, mean, std, rng_state);
                    set(m, row, col, sample);
                }
            }
        }
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy_layout_mismatch(devices::CUDA<SOURCE_DEV_SPEC>& source_device, devices::CUDA<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
        using DEVICE = devices::CUDA<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
//        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        containers::cuda::kernels::copy<DEVICE, SOURCE_SPEC, TARGET_SPEC><<<grid, block>>>(source, target);
        check_status(target_device);
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(devices::CUDA<SOURCE_DEV_SPEC>& source_device, devices::CUDA<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice);
            check_status(source_device);
        }
        else{
            copy_layout_mismatch(source_device, target_device, source, target);
        }
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy_layout_mismatch(devices::CPU<SOURCE_DEV_SPEC>& source_device, devices::CUDA<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
//        static_assert(!TARGET_SPEC::IS_VIEW);
        using DEVICE_CUDA = devices::CUDA<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
//        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(!TARGET_SPEC::IS_VIEW){
            // make a temporary copy of the source matrix (with the same layout as the target) and then copy it directly
            MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename TARGET_SPEC::LAYOUT, false>> temp;
            using TEMP_SPEC = typename decltype(temp)::SPEC;
            static_assert(TEMP_SPEC::SIZE_BYTES == TARGET_SPEC::SIZE_BYTES);
            malloc(source_device, temp);
            copy(source_device, source_device, source, temp);
            auto temp_size = TEMP_SPEC::SIZE_BYTES;
            cudaMemcpy(target._data, temp._data, temp_size, cudaMemcpyHostToDevice);
            check_status(target_device);
            free(source_device, temp);
        }
        else{
            MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename SOURCE_SPEC::LAYOUT, false>> temp;
            malloc(target_device, temp);
            copy(source_device, target_device, source, temp);
//            {
//                constexpr TI BLOCKSIZE_COLS = 32;
//                constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
//                dim3 grid(N_BLOCKS_COLS);
//                dim3 block(BLOCKSIZE_COLS);
//                containers::cuda::kernels::copy<DEVICE_CUDA, TARGET_SPEC, typename decltype(temp)::SPEC><<<grid, block>>>(target, temp);
//                check_status(target_device);
//            }
            copy(target_device, target_device, temp, target);
        }
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(devices::CPU<SOURCE_DEV_SPEC>& source_device, devices::CUDA<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyHostToDevice);
            check_status(target_device);
        }
        else{
            copy_layout_mismatch(source_device, target_device, source, target);
        }
    }

    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy_layout_mismatch(devices::CUDA<SOURCE_DEV_SPEC>& source_device, devices::CPU<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
//        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS>> temp_gpu, temp_cpu;
        using TEMP_SPEC = typename decltype(temp_gpu)::SPEC;
        malloc(source_device, temp_gpu);
//        {
//            constexpr TI BLOCKSIZE_COLS = 32;
//            constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
//            dim3 grid(N_BLOCKS_COLS);
//            dim3 block(BLOCKSIZE_COLS);
//            containers::cuda::kernels::copy<DEVICE_CUDA, typename decltype(temp_gpu)::SPEC, SOURCE_SPEC><<<grid, block>>>(temp_gpu, source);
//            check_status(source_device);
//        }
        copy(source_device, source_device, source, temp_gpu);
        malloc(target_device, temp_cpu);
        cudaMemcpy(temp_cpu._data, temp_gpu._data, TEMP_SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
        check_status(source_device);
        free(source_device, temp_gpu);
        copy(target_device, target_device, temp_cpu, target);
        free(target_device, temp_cpu);
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(devices::CUDA<SOURCE_DEV_SPEC>& source_device, devices::CPU<TARGET_DEV_SPEC>& target_device, const Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
            check_status(target_device);
        }
        else{
            copy_layout_mismatch(source_device, target_device, source, target);
        }
    }

    template<typename DEV_SPEC, typename SPEC, typename VALUE_T>
    void set_all(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& m, VALUE_T value){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        containers::cuda::kernels::set_all<DEVICE, SPEC, VALUE_T><<<grid, block>>>(m, value);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void randn(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& m, typename SPEC::T mean, typename SPEC::T std, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        containers::cuda::kernels::randn<<<grid, block>>>(tag_device, m, mean, std, rng);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void randn(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& m, RNG& rng){
        randn(device, m, 0, 1, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif