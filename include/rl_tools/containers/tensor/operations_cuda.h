#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_CUDA_H

#include "tensor.h"
#include "../../mode/mode.h"

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
#if !defined(RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS)
    template<typename DEV_SPEC, typename T, typename T_TI, T_TI SIZE, bool CONST>
    void malloc(devices::CUDA<DEV_SPEC>& device, tensor::TensorDynamic<T, T_TI, SIZE, CONST>& tensor){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, tensor._data == nullptr, "Tensor is already allocated");
#endif
        T *temp = nullptr;
        // auto result = cudaMalloc(&temp, SIZE_BYTES);
        constexpr TI SIZE_BYTES = SIZE * sizeof(T);
        auto result = cudaMallocAsync(&temp, SIZE_BYTES, device.stream);
        cudaDeviceSynchronize();
        cudaStreamSynchronize(device.stream);
        tensor._data = temp;
        check_status(device);
        count_malloc(device, SIZE_BYTES);

#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate container: " << cudaGetErrorString(result) << std::endl;
        }
#endif

    }
    template <typename DEV_SPEC, typename T, typename T_TI, T_TI SIZE, bool CONST>
    void free(devices::CUDA<DEV_SPEC>& device, tensor::TensorDynamic<T, T_TI, SIZE, CONST>& tensor){
        cudaFree(tensor._data);
        check_status(device);
    }
#endif
    namespace tensor::kernels {
        template<typename DEVICE, typename FROM_SPEC, typename TO_SPEC>
        __global__
        void copy(DEVICE device, Tensor<FROM_SPEC> from, Tensor<TO_SPEC> to){
            static_assert(length(typename FROM_SPEC::SHAPE{}) == 1);
            using TI = typename DEVICE::index_t;
            constexpr TI SIZE = get<0>(typename FROM_SPEC::SHAPE{});
            TI step_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(step_i < SIZE){
                set(device, to, get(device, from, step_i), step_i);
            }
        }

    }
    // Copy function when on CUDA device
    template<typename FROM_DEV_SPEC, typename TO_DEV_SPEC, typename FROM_SPEC, typename TO_SPEC,
        typename utils::typing::enable_if<devices::CUDA<FROM_DEV_SPEC>::TAG, int>::type = 0>
    __device__
    void copy(devices::CUDA<FROM_DEV_SPEC>& from_device, devices::CUDA<TO_DEV_SPEC>& to_device, Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to) {
        using FROM_DEVICE = devices::CUDA<FROM_DEV_SPEC>;
        using TI = typename FROM_DEVICE::index_t;

        static_assert(tensor::same_dimensions<FROM_SPEC, TO_SPEC>(), "Dimensions must be the same");

        if constexpr(length(typename FROM_SPEC::SHAPE{}) > 1) {
            for(TI i = 0; i < get<0>(typename FROM_SPEC::SHAPE{}); ++i) {
                auto next_from = view(from_device, from, i);
                auto next_to = view(to_device, to, i);
                copy(from_device, to_device, next_from, next_to);
            }
        } else {
            // Inside kernels, just execute the copy
            for(TI i = 0; i < get<0>(typename FROM_SPEC::SHAPE{}); i++) {
                set(to_device, to, get(from_device, from, i), i);
            }
        }
    }
    // Copy function when on Host
    template<typename FROM_DEV_SPEC, typename TO_DEV_SPEC, typename FROM_SPEC, typename TO_SPEC,
        typename std::enable_if<!devices::CUDA<FROM_DEV_SPEC>::TAG, int>::type = 0>
    void copy(devices::CUDA<FROM_DEV_SPEC>& from_device, devices::CUDA<TO_DEV_SPEC>& to_device, Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        using FROM_DEVICE = devices::CUDA<FROM_DEV_SPEC>;
        using TI = typename FROM_DEVICE::index_t;
        static_assert(tensor::same_dimensions<FROM_SPEC, TO_SPEC>());
        if constexpr(length(typename FROM_SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename FROM_SPEC::SHAPE{}); ++i){
                auto next_from = view(from_device, from, i);
                auto next_to = view(to_device, to, i);
                copy(from_device, to_device, next_from, next_to);
            }
        }
        else{
            using DEVICE = devices::CUDA<FROM_DEV_SPEC>;
            using T = typename FROM_SPEC::T;
            using TI = typename FROM_SPEC::TI;
            constexpr TI BLOCKSIZE_COLS = 32;
            constexpr TI SIZE = get<0>(typename FROM_SPEC::SHAPE{});
            constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(SIZE, BLOCKSIZE_COLS);
            dim3 grid(N_BLOCKS_COLS);
            dim3 block(BLOCKSIZE_COLS);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            tensor::kernels::copy<<<grid, block, 0, from_device.stream>>>(tag_device, from, to);
            check_status(from_device);
        }
    }
    template<typename FROM_DEV_SPEC, typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(devices::CUDA<FROM_DEV_SPEC>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to) {
        // static_assert(false, "Cannot copy Tensor from CUDA to non-CUDA device yet");
        std::cout << 'hello' << std::endl;
    }
    template<typename FROM_DEVICE, typename TO_DEV_SPEC, typename FROM_SPEC, typename TO_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(FROM_DEVICE& from_device, devices::CUDA<TO_DEV_SPEC>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to) {
        // static_assert(false, "Cannot copy Tensor from CUDA to non-CUDA device yet");
        std::cout << 'hello' << std::endl;
    }
    namespace tensor::kernels {


    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
