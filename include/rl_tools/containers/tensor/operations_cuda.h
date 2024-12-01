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
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
