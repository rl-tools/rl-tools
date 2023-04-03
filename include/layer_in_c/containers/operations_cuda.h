#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CUDA_H

#include <layer_in_c/containers.h>
#include <layer_in_c/devices/cuda.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace layer_in_c{
#ifndef LAYER_IN_C_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
        /* for checking the pitch
        {
            size_t pitch;
            cudaMallocPitch(&matrix._data, &pitch, SPEC::COLS * sizeof(typename SPEC::T), SPEC::ROWS);

        }
        */
#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        auto result = cudaMalloc(&matrix._data, SPEC::SIZE_BYTES);
        check_status(device);

#ifdef LAYER_IN_C_DEBUG_CONTAINER_CHECK_MALLOC
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
        template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
        __global__ void
        copy(Matrix<TARGET_SPEC> target, Matrix<SOURCE_SPEC> source) {
            static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
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
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy_layout_mismatch(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        using DEVICE = devices::CUDA<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        containers::cuda::kernels::copy<DEVICE, TARGET_SPEC, SOURCE_SPEC><<<grid, block>>>(target, source);
        check_status(target_device);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice);
            check_status(source_device);
        }
        else{
            copy_layout_mismatch(target_device, source_device, target, source);
        }
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy_layout_mismatch(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
//        static_assert(!TARGET_SPEC::IS_VIEW);
        using DEVICE_CUDA = devices::CUDA<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(!TARGET_SPEC::IS_VIEW){
            // make a temporary copy of the source matrix (with the same layout as the target) and then copy it directly
            MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename TARGET_SPEC::LAYOUT, false>> temp;
            using TEMP_SPEC = typename decltype(temp)::SPEC;
            static_assert(TEMP_SPEC::SIZE_BYTES == TARGET_SPEC::SIZE_BYTES);
            malloc(source_device, temp);
            copy(source_device, source_device, temp, source);
            auto temp_size = TEMP_SPEC::SIZE_BYTES;
            cudaMemcpy(target._data, temp._data, temp_size, cudaMemcpyHostToDevice);
            check_status(target_device);
            free(source_device, temp);
        }
        else{
            MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS, typename SOURCE_SPEC::LAYOUT, false>> temp;
            malloc(target_device, temp);
            copy(target_device, source_device, temp, source);
//            {
//                constexpr TI BLOCKSIZE_COLS = 32;
//                constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
//                dim3 grid(N_BLOCKS_COLS);
//                dim3 block(BLOCKSIZE_COLS);
//                containers::cuda::kernels::copy<DEVICE_CUDA, TARGET_SPEC, typename decltype(temp)::SPEC><<<grid, block>>>(target, temp);
//                check_status(target_device);
//            }
            copy(target_device, target_device, target, temp);
        }
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CUDA<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyHostToDevice);
            check_status(target_device);
        }
        else{
            copy_layout_mismatch(target_device, source_device, target, source);
        }
    }

    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy_layout_mismatch(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        static_assert(utils::typing::is_same_v<typename TARGET_SPEC::T, typename SOURCE_SPEC::T>);
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        MatrixDynamic<matrix::Specification<T, TI, SPEC::ROWS, SPEC::COLS>> temp_gpu, temp_cpu;
        using TEMP_SPEC = typename decltype(temp_gpu)::SPEC;
        malloc(source_device, temp_gpu);
//        {
//            constexpr TI BLOCKSIZE_COLS = 32;
//            constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(TARGET_SPEC::COLS, BLOCKSIZE_COLS);
//            dim3 grid(N_BLOCKS_COLS);
//            dim3 block(BLOCKSIZE_COLS);
//            containers::cuda::kernels::copy<DEVICE_CUDA, typename decltype(temp_gpu)::SPEC, SOURCE_SPEC><<<grid, block>>>(temp_gpu, source);
//            check_status(source_device);
//        }
        copy(source_device, source_device, temp_gpu, source);
        malloc(target_device, temp_cpu);
        cudaMemcpy(temp_cpu._data, temp_gpu._data, TEMP_SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
        check_status(source_device);
        free(source_device, temp_gpu);
        copy(target_device, target_device, target, temp_cpu);
        free(target_device, temp_cpu);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CUDA<SOURCE_DEV_SPEC>& source_device, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source){
        using DEVICE_CUDA = devices::CUDA<SOURCE_DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(containers::check_memory_layout<TARGET_SPEC, SOURCE_SPEC>){
            cudaMemcpy(target._data, source._data, SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
            check_status(source_device);
        }
        else{
            copy_layout_mismatch(target_device, source_device, target, source);
        }
    }

    template<typename DEV_SPEC, typename SPEC, typename VALUE_T>
    void set_all(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC>& m, VALUE_T value){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = LAYER_IN_C_DEVICES_CUDA_CEIL(SPEC::COLS, BLOCKSIZE_COLS);
        dim3 grid(N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE_COLS);
        containers::cuda::kernels::set_all<DEVICE, SPEC, VALUE_T><<<grid, block>>>(m, value);
        check_status(device);
    }
}

#endif