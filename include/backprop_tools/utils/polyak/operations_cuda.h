#ifndef BACKPROP_TOOLS_NN_UTILS_POLYAK_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_NN_UTILS_POLYAK_OPERATIONS_CUDA_H


namespace backprop_tools::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    namespace internal {
        template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC, bool SQUARE=false>
        __global__
        void update_kernel(Matrix<TARGET_SPEC> target, const Matrix<SOURCE_SPEC> source, const typename TARGET_SPEC::T polyak) {
            static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
            using SPEC = TARGET_SPEC;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI ROWS = SPEC::ROWS;
            constexpr TI COLS = SPEC::COLS;
            TI col_i = threadIdx.x + blockIdx.x * blockDim.x;
            TI row_i = threadIdx.y + blockIdx.y * blockDim.y;
            if(col_i < COLS && row_i < ROWS){
                T s = get(source, row_i, col_i);
                if constexpr(SQUARE){
                    s *= s;
                }
                set(target, row_i, col_i, polyak * get(target, row_i, col_i) + (1 - polyak) * s);
            }
        }
    }
    template<typename DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC, bool SQUARE=false>
    void update(devices::CUDA<DEV_SPEC>& dev, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source, const typename TARGET_SPEC::T polyak) {
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using SPEC = TARGET_SPEC;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ROWS = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_COLS = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ROWS = BACKPROP_TOOLS_DEVICES_CUDA_CEIL(SPEC::ROWS, BLOCKSIZE_ROWS);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_COLS = BACKPROP_TOOLS_DEVICES_CUDA_CEIL(SPEC::COLS, BLOCKSIZE_COLS);
        dim3 activation_grid(N_BLOCKS_COLS, N_BLOCKS_ROWS);
        dim3 activation_block(BLOCKSIZE_COLS, BLOCKSIZE_ROWS);
        internal::update_kernel<DEVICE, TARGET_SPEC, SOURCE_SPEC, SQUARE><<<activation_grid, activation_block>>>(target, source, polyak);
        check_status(dev);
    }
    template<typename DEV_SPEC, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update_squared(devices::CUDA<DEV_SPEC>& dev, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source, const typename TARGET_SPEC::T polyak) {
        update<DEV_SPEC, TARGET_SPEC, SOURCE_SPEC, true>(dev, target, source, polyak);
    }
}


#endif