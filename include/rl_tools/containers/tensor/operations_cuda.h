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
    namespace tensor::kernels{
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
        if constexpr(!tensor::same_dimensions_shape<typename FROM_SPEC::STRIDE, typename TO_SPEC::STRIDE>()){
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
        else{
            cudaMemcpyAsync(to._data, from._data, FROM_SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice, from_device.stream);
        }
    }
    template<typename FROM_DEV_SPEC, typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(devices::CUDA<FROM_DEV_SPEC>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to) {
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::SHAPE, typename TO_SPEC::SHAPE>());
        constexpr bool SAME_STRIDE = tensor::same_dimensions_shape<typename FROM_SPEC::STRIDE, typename TO_SPEC::STRIDE>();
        static_assert(SAME_STRIDE);
        if constexpr(SAME_STRIDE){
            cudaMemcpyAsync(to._data, from._data, FROM_SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost, from_device.stream);
        }
        else{

        }
    }
    template<typename FROM_DEVICE, typename TO_DEV_SPEC, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, devices::CUDA<TO_DEV_SPEC>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to) {
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::SHAPE, typename TO_SPEC::SHAPE>());
        constexpr bool SAME_STRIDE = tensor::same_dimensions_shape<typename FROM_SPEC::STRIDE, typename TO_SPEC::STRIDE>();
        static_assert(SAME_STRIDE);
        if constexpr(SAME_STRIDE){
            cudaMemcpyAsync(to._data, from._data, FROM_SPEC::SIZE_BYTES, cudaMemcpyHostToDevice, to_device.stream);
        }
        else{

        }
    }
    namespace tensor::kernels {
        template<typename DEV_SPEC, typename SPEC, typename OPERATION>
        __global__
        void unary_operation(devices::CUDA<DEV_SPEC> device, const OPERATION op, Tensor<SPEC> t){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            static_assert(SPEC::SHAPE::LENGTH == 1);
            constexpr TI SIZE = SPEC::SHAPE::template GET<0>;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if (thread_i < SIZE) {
                T t_value = get(device, t, thread_i);
                T result_value = OPERATION::operation(device, op, t_value);
                set(device, t, result_value, thread_i);
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename OPERATION, typename SPEC_OUTPUT>
        __global__
        void unary_operation(devices::CUDA<DEV_SPEC> device, const OPERATION op, Tensor<SPEC> t, Tensor<SPEC_OUTPUT> output) {
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            static_assert(SPEC::SHAPE::LENGTH == 1);
            constexpr TI SIZE = SPEC::SHAPE::template GET<0>;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if (thread_i < SIZE) {
                T t_value = get(device, t, thread_i);
                T result_value = OPERATION::operation(device, op, t_value);
                set(device, output, result_value, thread_i);
            }
        }

    }
    // template<typename DEV_SPEC, typename SPEC, typename OPERATION, typename SPEC_OUTPUT,
    //     typename utils::typing::enable_if<devices::CUDA<DEV_SPEC>::TAG, int>::type = 0>
    // __device__
    // void unary_operation(devices::CUDA<DEV_SPEC>& device, const OPERATION& op, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
    //     using DEVICE = devices::CUDA<DEV_SPEC>;
    //     using T = typename SPEC::T;
    //     using TI = typename DEVICE::index_t;
    //     if constexpr(length(typename SPEC::SHAPE{}) > 1){
    //         for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
    //             auto next_t = view(device, t, i);
    //             auto next_output = view(device, output, i);
    //             unary_operation(device, op, next_t, next_output);
    //         }
    //     }
    //     else{
    //         using DEVICE = devices::CUDA<DEV_SPEC>;
    //         using T = typename SPEC::T;
    //         using TI = typename SPEC::TI;
    //         constexpr TI BLOCKSIZE_COLS = 32;
    //         constexpr TI SIZE = SPEC::SHAPE::template GET<0>;
    //         constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(SIZE, BLOCKSIZE_COLS);
    //         dim3 grid(N_BLOCKS_COLS);
    //         dim3 block(BLOCKSIZE_COLS);
    //         devices::cuda::TAG<DEVICE, true> tag_device{};
    //         tensor::kernels::unary_operation<<<grid, block, 0, device.stream>>>(tag_device, op, t, output);
    //         check_status(device);
    //     }
    // }


    // Unary operation when on Host device
    template<typename DEV_SPEC, typename SPEC, typename OPERATION, typename SPEC_OUTPUT,
        typename std::enable_if<!devices::CUDA<DEV_SPEC>::TAG, int>::type = 0>
    void unary_operation(devices::CUDA<DEV_SPEC>& device, const OPERATION& op, Tensor<SPEC>& t, Tensor<SPEC_OUTPUT>& output){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) > 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); ++i){
                auto next_t = view(device, t, i);
                auto next_output = view(device, output, i);
                unary_operation(device, op, next_t, next_output);
            }
        }
        else{
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            constexpr TI BLOCKSIZE_COLS = 32;
            constexpr TI SIZE = SPEC::SHAPE::template GET<0>;
            constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(SIZE, BLOCKSIZE_COLS);
            dim3 grid(N_BLOCKS_COLS);
            dim3 block(BLOCKSIZE_COLS);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            tensor::kernels::unary_operation<<<grid, block, 0, device.stream>>>(tag_device, op, t, output);
            check_status(device);
        }
    }

    namespace tensor::kernels{
        template<typename DEV_SPEC, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER, typename RESULT_SPEC>
        __global__ void unary_associative_reduce(devices::CUDA<DEV_SPEC> device, const tensor::UnaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION> op, const Tensor<SPEC> t, Tensor<RESULT_SPEC> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i == 0) {
                ACCUMULATOR_TYPE output = _unary_associative_reduce(device, op, t, op.initial_value);
                set(device, result, output, 0);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC, auto UNARY_REDUCE_OPERATION, typename ACCUMULATOR_TYPE, typename CURRENT_TYPE, typename OPERATION_PARAMETER, typename RESULT_SPEC>
    void unary_associative_reduce(devices::CUDA<DEV_SPEC>& device, const tensor::UnaryReduceOperation<OPERATION_PARAMETER, ACCUMULATOR_TYPE, CURRENT_TYPE, UNARY_REDUCE_OPERATION>& op, const Tensor<SPEC>& t, Tensor<RESULT_SPEC>& result){
        static_assert(RESULT_SPEC::SHAPE::LENGTH == 1);
        static_assert(RESULT_SPEC::SHAPE::template GET<0> == 1);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        dim3 grid(1);
        dim3 block(1);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        tensor::kernels::unary_associative_reduce<<<grid, block, 0, device.stream>>>(tag_device, op, t, result);
        check_status(device);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
