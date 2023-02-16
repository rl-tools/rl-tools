#ifndef LAYER_IN_C_NN_LOSS_FUNCTIONS_MSE_OPERATIONS_CUDA_H
#define LAYER_IN_C_NN_LOSS_FUNCTIONS_MSE_OPERATIONS_CUDA_H

#include <layer_in_c/devices/cuda.h>

namespace layer_in_c::nn::loss_functions {
    namespace internal::mse{
        template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
        __global__
        void d_mse_d_x_kernel(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, typename SPEC_A::T loss_weight = 1) {
            static_assert(containers::check_structure<SPEC_A, SPEC_B>);
            static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
            using T = typename SPEC_A::T;
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            constexpr TI BATCH_SIZE = SPEC_A::ROWS;
            constexpr TI OUTPUT_DIM = SPEC_A::COLS;

            TI output_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
            TI output_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
            if(output_pos_x < OUTPUT_DIM && output_pos_y < BATCH_SIZE){
//                TI index = output_pos_y * OUTPUT_DIM + output_pos_x;
                T diff = get(a, output_pos_y, output_pos_x) - get(b, output_pos_y, output_pos_x);
                set(d_a, output_pos_y, output_pos_x, 2*diff/(SPEC_A::ROWS * SPEC_A::COLS) * loss_weight);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B>
    typename SPEC_A::T mse(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, typename SPEC_A::T loss_weight = 1) {
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        T acc = 0;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++) {
//                TI index = row_i * SPEC_A::COLS + col_i;
                T diff = get(a, row_i, col_i) - get(b, row_i, col_i);
                acc += diff * diff;
            }
        }
        return acc * loss_weight / (SPEC_A::ROWS * SPEC_A::COLS);
    }

    template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    void d_mse_d_x(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, typename SPEC_A::T loss_weight = 1) {
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BATCH_SIZE = SPEC_A::ROWS;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t OUTPUT_DIM = SPEC_A::COLS;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_BATCH = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_BATCH = LAYER_IN_C_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_ACTIVATION_BATCH);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = LAYER_IN_C_DEVICES_CUDA_CEIL(OUTPUT_DIM, BLOCKSIZE_ACTIVATION_OUTPUT);
        dim3 activation_grid(N_BLOCKS_ACTIVATION_OUTPUT, N_BLOCKS_ACTIVATION_BATCH);
        dim3 activation_block(BLOCKSIZE_ACTIVATION_OUTPUT, BLOCKSIZE_ACTIVATION_BATCH);
        internal::mse::d_mse_d_x_kernel<<<activation_grid, activation_block>>>(device, a, b, d_a, loss_weight);
        check_status(device);
    }
}

#endif
