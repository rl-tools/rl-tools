#include "adam.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace nn::optimizers::adam::cuda {
        template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
        __global__
        void update_kernel(devices::CUDA<DEV_SPEC>& device, nn::parameters::Adam::instance<SPEC> p, nn::optimizers::Adam<PARAMETERS> optimizer) {
            // fully fused adam update
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC::CONTAINER::T;
            using TI = typename DEVICE::index_t;

            TI col_i = blockIdx.x * blockDim.x + threadIdx.x;
            TI row_i = blockIdx.y * blockDim.y + threadIdx.y;
            if(col_i < SPEC::CONTAINER::COLS && row_i < SPEC::CONTAINER::ROWS){
                T d_weight = get(p.gradient, row_i, col_i);
                T d_weight_first_order_moment = PARAMETERS::BETA_1 * get(p.gradient_first_order_moment, row_i, col_i) + (1 - PARAMETERS::BETA_1) * d_weight;
                set(p.gradient_first_order_moment, row_i, col_i, d_weight_first_order_moment);
                T d_weight_second_order_moment = PARAMETERS::BETA_2 * get(p.gradient_second_order_moment, row_i, col_i) + (1 - PARAMETERS::BETA_2) * d_weight * d_weight;
                set(p.gradient_second_order_moment, row_i, col_i, d_weight_second_order_moment);
                T weight_update = optimizer.alpha * optimizer.first_order_moment_bias_correction * d_weight_first_order_moment / (math::sqrt(typename DEVICE::SPEC::MATH_DEVICE_ACCURATE(), d_weight_second_order_moment * optimizer.second_order_moment_bias_correction) + PARAMETERS::EPSILON);
                if constexpr(utils::typing::is_same_v<typename SPEC::CATEGORY_TAG, nn::parameters::categories::Weights> &&  PARAMETERS::WEIGHT_DECAY > 0){
                    weight_update += get(p.parameters, row_i, col_i) * PARAMETERS::WEIGHT_DECAY / 2;
                }
                increment(p.parameters, row_i, col_i, -weight_update);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
    void update(devices::CUDA<DEV_SPEC>& device, nn::parameters::Adam::instance<SPEC>& p, nn::optimizers::Adam<PARAMETERS>& optimizer) {
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_INPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::CONTAINER::ROWS, BLOCKSIZE_ACTIVATION_OUTPUT);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_INPUT = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::CONTAINER::COLS, BLOCKSIZE_ACTIVATION_INPUT);
        dim3 activation_grid(N_BLOCKS_ACTIVATION_INPUT, N_BLOCKS_ACTIVATION_OUTPUT);
        dim3 activation_block(BLOCKSIZE_ACTIVATION_INPUT, BLOCKSIZE_ACTIVATION_OUTPUT);
        nn::optimizers::adam::cuda::update_kernel<<<activation_grid, activation_block>>>(device, p, optimizer);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
