#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_CUDA_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers/matrix/matrix.h"


#include "layer.h"
#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    __global__
    void evaluate_kernel(devices::CUDA<DEV_SPEC> device, const nn::layers::sample_and_squash::LayerForward<SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<OUTPUT_SPEC> output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC> buffer, RNG rng, const Mode<MODE> mode = Mode<mode::Default<>>{}) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState rng_state;
        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            evaluate_per_sample(device, layer, input, output, buffer, rng, batch_step_i, mode);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::sample_and_squash::LayerForward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rng = rl_tools::random::next(device.random, rng);
        evaluate_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, layer, input, output, buffer, rng, mode);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    __global__
    void forward_kernel(devices::CUDA<DEV_SPEC> device, nn::layers::sample_and_squash::LayerGradient<SPEC> layer, const Matrix<INPUT_SPEC> input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC> buffer, RNG rng, const Mode<MODE> mode = Mode<mode::Default<>>{}){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
//        using BUFFERS = nn::layers::sample_and_squash::LayerGradient<BUFFER_SPEC>;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(BATCH_SIZE == SPEC::BATCH_SIZE);
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        curandState rng_state;
        curand_init(rng, batch_step_i, 0, &rng_state);
        if(batch_step_i < BATCH_SIZE){
            forward_per_sample(device, layer, input, buffer, rng, batch_step_i, mode);
        }
    }

    template <typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rng = rl_tools::random::next(device.random, rng);
        forward_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, layer, input, buffer, rng, mode);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    __global__
    void backward_full_kernel(devices::CUDA<DEV_SPEC> device, nn::layers::sample_and_squash::LayerGradient<SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_INPUT_SPEC> d_input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC> buffer, const Mode<MODE> mode = Mode<mode::Default<>>{}) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
//        using BUFFERS = nn::layers::sample_and_squash::LayerGradient<BUFFER_SPEC>;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(BATCH_SIZE == SPEC::BATCH_SIZE);
        TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(batch_step_i < BATCH_SIZE){
            T log_alpha = get(layer.log_alpha.parameters, 0, 0);
            T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
            T d_log_alpha = backward_full_per_sample(device, layer, input, d_output, d_input, buffer, alpha, batch_step_i, mode)/BATCH_SIZE;
            atomicAdd(layer.log_alpha.gradient._data, d_log_alpha);
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        backward_full_kernel<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, layer, input, d_output, d_input, buffer, mode);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

