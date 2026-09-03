#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_HELPER_OPERATIONS_CUDA_H

#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::gru::helper{

    namespace nn::layers::gru::kernels {
        template<typename DEV_SPEC, typename SPEC_BIAS, typename SPEC_OUT>
        __global__
        void set_bias_inplace_kernel( devices::CUDA<DEV_SPEC> device, const Tensor<SPEC_BIAS> bias, Tensor<SPEC_OUT> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI     = typename DEVICE::index_t;
            using T      = typename SPEC_BIAS::T;

            constexpr TI ROWS = SPEC_OUT::SHAPE::template GET<0>;
            constexpr TI COLS = SPEC_OUT::SHAPE::template GET<1>;
            static_assert(SPEC_BIAS::SHAPE::template GET<0> == COLS);

            TI i = blockIdx.x * blockDim.x + threadIdx.x;
            TI j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < ROWS && j < COLS) {
                T value = get(device, bias, j);
                set(device, result, value, i, j);
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC_BIAS, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void set_bias_inplace(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI     = typename DEVICE::index_t;
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1, "bias must be 1D [HIDDEN_DIM]");
        static_assert(length(typename SPEC_OUT::SHAPE{})  == 2, "result must be 2D [BATCH_SIZE, HIDDEN_DIM]");
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{}), "bias length == result second dimension (HIDDEN_DIM)");

        constexpr TI BATCH_SIZE = SPEC_OUT::SHAPE:: template GET<0>;
        constexpr TI HIDDEN_DIM = SPEC_OUT::SHAPE:: template GET<1>;
        constexpr TI BLOCK = 32;

        constexpr TI BLOCKSIZE = 32;
        constexpr TI ROWS = SPEC_OUT::SHAPE:: template GET<0>;
        constexpr TI COLS = SPEC_OUT::SHAPE:: template GET<1>;
        constexpr TI N_BLOCKS_ROWS = RL_TOOLS_DEVICES_CUDA_CEIL(ROWS, BLOCKSIZE);
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE);
        dim3 grid(N_BLOCKS_ROWS, N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE, BLOCKSIZE);

        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::set_bias_inplace_kernel<<<grid, block, 0, device.stream>>>(tag_device, bias, result);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void matrix_multiply_transpose_bias(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_transpose_bias");
#endif
        // Y = WX
        // Y^T = X^T W^T
        // W = t1, X^T = t2, Y^T = result
        // Y^T = result = t2 t1^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{})); // INPUT_DIM
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        // for(TI i=0; i < get<0>(typename SPEC_OUT::SHAPE{}); i++){
        //     for(TI j=0; j < get<1>(typename SPEC_OUT::SHAPE{}); j++){
        //         T bias_value = get(device, bias, j);
        //         set(device, result, bias_value, i, j);
        //     }
        // }
        set_bias_inplace(device, bias, result);
        auto t1_transpose = permute(device, t1, tensor::PermutationSpec<1, 0>{});
        matrix_multiply_accumulate(device, t2, t1_transpose, result);
//        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
//            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
//                T acc = get(device, bias, i);
//                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
//                    acc += get(device, t1, i, k) * get(device, t2, j, k);
//                }
//                set(device, result, acc, j, i);
//            }
//        }
    }

    namespace nn::layers::gru::kernels {
        template<typename DEV_SPEC, typename SPEC_BIAS, typename SPEC_OUT>
        __global__
        void add_bias_inplace_kernel( devices::CUDA<DEV_SPEC> device, const Tensor<SPEC_BIAS> bias, Tensor<SPEC_OUT> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI     = typename DEVICE::index_t;
            using T      = typename SPEC_BIAS::T;

            constexpr TI HIDDEN_DIM = SPEC_BIAS::SHAPE:: template GET<0>;
            constexpr TI BATCH_SIZE = SPEC_OUT::SHAPE:: template GET<0>;

            TI i = blockIdx.x * blockDim.x + threadIdx.x;
            TI j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < BATCH_SIZE && j < HIDDEN_DIM) {
                T value = get(device, result, i, j) + get(device, bias, j);
                set(device, result, value, i, j);
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC_BIAS, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void add_bias_inplace( devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI     = typename DEVICE::index_t;
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1, "bias must be 1D [HIDDEN_DIM]");
        static_assert(length(typename SPEC_OUT::SHAPE{})  == 2, "result must be 2D [BATCH_SIZE, HIDDEN_DIM]");
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{}), "bias length == result second dimension (HIDDEN_DIM)");

        constexpr TI BATCH_SIZE = SPEC_OUT::SHAPE:: template GET<0>;
        constexpr TI HIDDEN_DIM = SPEC_OUT::SHAPE:: template GET<1>;
        constexpr TI BLOCK = 32;

        constexpr TI BLOCKSIZE = 32;
        constexpr TI ROWS = SPEC_OUT::SHAPE:: template GET<0>;
        constexpr TI COLS = SPEC_OUT::SHAPE:: template GET<1>;
        constexpr TI N_BLOCKS_ROWS = RL_TOOLS_DEVICES_CUDA_CEIL(ROWS, BLOCKSIZE);
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE);
        dim3 grid(N_BLOCKS_ROWS, N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE, BLOCKSIZE);

        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::add_bias_inplace_kernel<<<grid, block, 0, device.stream>>>(tag_device, bias, result);
        check_status(device);
    }



    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void matrix_multiply_transpose_bias_accumulate(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_transpose_bias_accumulate");
#endif
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_2::SHAPE{})); // INPUT_DIM
        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        // for(TI i=0; i < get<0>(typename SPEC_OUT::SHAPE{}); i++){
        //     for(TI j=0; j < get<1>(typename SPEC_OUT::SHAPE{}); j++){
        //         T value = get(device, result, i, j) + get(device, bias, j);
        //         set(device, result, value, i, j);
        //     }
        // }
        add_bias_inplace(device, bias, result);
        auto t1_transpose = permute(device, t1, tensor::PermutationSpec<1, 0>{});
        matrix_multiply_accumulate(device, t2, t1_transpose, result);
//        for(TI i=0; i < get<0>(typename SPEC_1::SHAPE{}); ++i){
//            for(TI j=0; j < get<0>(typename SPEC_2::SHAPE{}); ++j){
//                T acc = get(device, result, j, i) + get(device, bias, i);
//                for(TI k=0; k < get<1>(typename SPEC_1::SHAPE{}); ++k){
//                    acc += get(device, t1, i, k) * get(device, t2, j, k);
//                }
//                set(device, result, acc, j, i);
//            }
//        }
    }


    namespace nn::layers::gru::kernels{
        template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT>
        __global__
        void matrix_multiply_broadcast_transpose_bias(devices::CUDA<DEV_SPEC> device, const Tensor<SPEC_1> t1, const Tensor<SPEC_2> t2, const Tensor<SPEC_BIAS> bias, Tensor<SPEC_OUT> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename SPEC_1::T;
            static_assert(SPEC_1::SHAPE::LENGTH == 2);
            constexpr TI ROWS = SPEC_OUT::SHAPE:: template GET<1>;
            constexpr TI COLS = SPEC_OUT::SHAPE:: template GET<0>;
            static_assert(SPEC_2::SHAPE::LENGTH == 1); // only one row
            constexpr TI INNER = SPEC_1::SHAPE:: template GET<1>;
            static_assert(INNER == SPEC_2::SHAPE:: template GET<0>);
            TI i = threadIdx.x + blockIdx.x * blockDim.x;
            TI j = threadIdx.y + blockIdx.y * blockDim.y;
            if(i < ROWS && j < COLS){
                T acc = get(device, bias, i);
                for(TI k=0; k < INNER; ++k){
                    acc += get(device, t1, i, k) * get(device, t2, k);
                }
                set(device, result, acc, j, i);
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_BIAS, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void matrix_multiply_broadcast_transpose_bias(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, const Tensor<SPEC_BIAS>& bias, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("gru::matrix_multiply_broadcast_transpose_bias");
#endif
        // Y^T = WX^T
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 1);
        static_assert(length(typename SPEC_OUT::SHAPE{}) == 2);
        static_assert(get<1>(typename SPEC_1::SHAPE{}) == get<0>(typename SPEC_2::SHAPE{})); // INPUT_DIM
//        static_assert(get<0>(typename SPEC_2::SHAPE{}) == get<0>(typename SPEC_OUT::SHAPE{})); // BATCH_SIZE
        static_assert(get<0>(typename SPEC_1::SHAPE{}) == get<1>(typename SPEC_OUT::SHAPE{})); // HIDDEN_DIM
        static_assert(length(typename SPEC_BIAS::SHAPE{}) == 1);
        static_assert(get<0>(typename SPEC_BIAS::SHAPE{}) == get<0>(typename SPEC_1::SHAPE{}));

        // using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;

        constexpr TI BLOCKSIZE = 32;
        constexpr TI ROWS = SPEC_OUT::SHAPE:: template GET<1>;
        constexpr TI COLS = SPEC_OUT::SHAPE:: template GET<0>;
        constexpr TI N_BLOCKS_ROWS = RL_TOOLS_DEVICES_CUDA_CEIL(ROWS, BLOCKSIZE);
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE);
        dim3 grid(N_BLOCKS_ROWS, N_BLOCKS_COLS);
        dim3 block(BLOCKSIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::matrix_multiply_broadcast_transpose_bias<<<grid, block, 0, device.stream>>>(tag_device, t1, t2, bias, result);
        check_status(device);
    }
}

namespace rl_tools::nn::layers::gru::mode{
    template <typename SPEC, typename DEV_SPEC, typename MODE, typename MODE_SPEC, typename TI, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    bool reset_sample(devices::CUDA<DEV_SPEC>& device, const Mode<nn::layers::gru::ResetMode<MODE, MODE_SPEC>>& mode, TI step_i, TI sample_i){
        using RTI = typename MODE_SPEC::TI;
        using RESET_SPEC = typename MODE_SPEC::RESET_CONTAINER_TYPE::SPEC;
        auto offset = (RTI)step_i * get<0>(typename RESET_SPEC::STRIDE{}) + (RTI)sample_i * get<1>(typename RESET_SPEC::STRIDE{});
        typename RESET_SPEC::T host_val;
        cudaMemcpy(&host_val, data(mode.reset_container) + offset, sizeof(host_val), cudaMemcpyDeviceToHost);
        return host_val;
    }
}

namespace rl_tools{
    namespace nn::layers::gru::kernels{
        template<typename DEV_SPEC, typename SPEC_FACTOR, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT>
        __global__
        void multiply_subtract_broadcast_kernel(devices::CUDA<DEV_SPEC> device, Tensor<SPEC_FACTOR> factor, Tensor<SPEC_1> t1, Tensor<SPEC_2> t2, Tensor<SPEC_OUTPUT> t_output){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename SPEC_FACTOR::T;
            constexpr TI ROWS = SPEC_FACTOR::SHAPE::template GET<0>;
            constexpr TI COLS = SPEC_FACTOR::SHAPE::template GET<1>;
            TI i = threadIdx.x + blockIdx.x * blockDim.x;
            TI j = threadIdx.y + blockIdx.y * blockDim.y;
            if(i < ROWS && j < COLS){
                T factor_value = get(device, factor, i, j);
                T t1_value = get(device, t1, j);
                T t2_value = get(device, t2, i, j);
                set(device, t_output, factor_value * (t1_value - t2_value), i, j);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC_FACTOR, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void multiply_subtract_broadcast(devices::CUDA<DEV_SPEC>& device, Tensor<SPEC_FACTOR>& factor, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI ROWS = SPEC_FACTOR::SHAPE::template GET<0>;
        constexpr TI COLS = SPEC_FACTOR::SHAPE::template GET<1>;
        constexpr TI BLOCKSIZE = 16;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(ROWS, BLOCKSIZE), RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE));
        dim3 block(BLOCKSIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::multiply_subtract_broadcast_kernel<<<grid, block, 0, device.stream>>>(tag_device, factor, t1, t2, t_output);
        check_status(device);
    }

    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_OUTPUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void multiply_accumulate_reduce(devices::CUDA<DEV_SPEC>& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUTPUT>& t_output){
        static_assert(length(typename SPEC_1::SHAPE{}) == 2);
        static_assert(length(typename SPEC_2::SHAPE{}) == 2);
        static_assert(length(typename SPEC_OUTPUT::SHAPE{}) == 1);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC_1::T;
        using TI = typename DEVICE::index_t;
        constexpr TI ROWS = SPEC_1::SHAPE::template GET<0>;
        constexpr TI COLS = SPEC_1::SHAPE::template GET<1>;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, ROWS, COLS>>> temp;
        malloc(device, temp);
        multiply(device, t1, t2, temp);
        reduce_sum<true>(device, temp, t_output);
        free(device, temp);
    }

    namespace nn::layers::gru::kernels{
        template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
        __global__
        void matrix_multiply_broadcast_accumulate_kernel(devices::CUDA<DEV_SPEC> device, const Tensor<SPEC_1> t1, const Tensor<SPEC_2> t2, Tensor<SPEC_OUT> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename SPEC_1::T;
            constexpr TI ROWS = SPEC_1::SHAPE::template GET<0>;
            constexpr TI INNER = SPEC_1::SHAPE::template GET<1>;
            constexpr TI COLS = SPEC_OUT::SHAPE::template GET<1>;
            TI row_i = threadIdx.x + blockIdx.x * blockDim.x;
            TI col_j = threadIdx.y + blockIdx.y * blockDim.y;
            if(row_i < ROWS && col_j < COLS){
                T acc = get(device, result, row_i, col_j);
                T t2_value = get(device, t2, col_j);
                for(TI k = 0; k < INNER; ++k){
                    acc += get(device, t1, row_i, k) * t2_value;
                }
                set(device, result, acc, row_i, col_j);
            }
        }

        template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_OUT>
        __global__
        void matrix_multiply_accumulate_reduce_kernel(devices::CUDA<DEV_SPEC> device, const Tensor<SPEC_1> t1, const Tensor<SPEC_2> t2, Tensor<SPEC_OUT> result){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename SPEC_1::T;
            constexpr TI ROWS = SPEC_1::SHAPE::template GET<0>;
            constexpr TI INNER = SPEC_1::SHAPE::template GET<1>;
            constexpr TI COLS = SPEC_2::SHAPE::template GET<1>;
            TI col_j = threadIdx.x + blockIdx.x * blockDim.x;
            if(col_j < COLS){
                T acc = get(device, result, col_j);
                for(TI row_i = 0; row_i < ROWS; ++row_i){
                    for(TI k = 0; k < INNER; ++k){
                        acc += get(device, t1, row_i, k) * get(device, t2, k, col_j);
                    }
                }
                set(device, result, acc, col_j);
            }
        }
    }

    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void matrix_multiply_broadcast_accumulate(devices::CUDA<DEV_SPEC>& device, Tensor<SPEC_1>& t1, Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE = 16;
        constexpr TI ROWS = SPEC_1::SHAPE::template GET<0>;
        constexpr TI COLS = SPEC_OUT::SHAPE::template GET<1>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(ROWS, BLOCKSIZE), RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE));
        dim3 block(BLOCKSIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::matrix_multiply_broadcast_accumulate_kernel<<<grid, block, 0, device.stream>>>(tag_device, t1, t2, result);
        check_status(device);
    }

    template<typename DEV_SPEC, typename SPEC_1, typename SPEC_2, typename SPEC_OUT, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void matrix_multiply_accumulate_reduce(devices::CUDA<DEV_SPEC>& device, const Tensor<SPEC_1>& t1, const Tensor<SPEC_2>& t2, Tensor<SPEC_OUT>& result){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI COLS = SPEC_2::SHAPE::template GET<1>;
        constexpr TI BLOCKSIZE = 32;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(COLS, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::matrix_multiply_accumulate_reduce_kernel<<<grid, block, 0, device.stream>>>(tag_device, t1, t2, result);
        check_status(device);
    }

    // per-batch-element reset of CUDA-resident GRU state from a device mask (the generic
    // _reset_sequential is a host loop over the mask)
    namespace nn::layers::gru::kernels{
        template<typename DEV_SPEC, typename INITIAL_SPEC, typename STATE_SPEC, typename STEP_SPEC, typename MASK>
        __global__
        void reset_sequential_kernel(devices::CUDA<DEV_SPEC> device, const Tensor<INITIAL_SPEC> initial_hidden_state, Tensor<STATE_SPEC> state, Tensor<STEP_SPEC> step, const MASK mask){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename STATE_SPEC::T;
            constexpr TI BATCH_SIZE = STATE_SPEC::SHAPE::template GET<0>;
            constexpr TI HIDDEN_DIM = STATE_SPEC::SHAPE::template GET<1>;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE && nn::layers::gru::mode::reset_mask_value(device, mask, batch_i)){
                set(device, step, (typename STEP_SPEC::T)0, batch_i);
                for(TI hidden_i = 0; hidden_i < HIDDEN_DIM; hidden_i++){
                    set(device, state, (T)get(device, initial_hidden_state, hidden_i), batch_i, hidden_i);
                }
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename STATE_SPEC, typename BASE_MODE, typename MODE_SPEC, typename rl_tools::utils::typing::enable_if<!DEV_SPEC::TAG, int>::type = 0>
    void _reset_sequential(devices::CUDA<DEV_SPEC>& device, const nn::layers::gru::LayerForward<SPEC>& layer, nn::layers::gru::State<STATE_SPEC>& state, mode::sequential::ResetMask<BASE_MODE, MODE_SPEC>& mode){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = get<0>(typename decltype(state.state)::SPEC::SHAPE{});
        static_assert(nn::layers::gru::mode::ResetMaskSize<decltype(mode.mask)>::VALUE == BATCH_SIZE, "The reset mask for GRU layers must have an entry for each batch element.");
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::gru::kernels::reset_sequential_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, layer.initial_hidden_state.parameters, state.state, state.step, mode.mask);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
