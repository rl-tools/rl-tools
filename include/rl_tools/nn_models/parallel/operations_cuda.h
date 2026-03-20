#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include "../../devices/cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::parallel::cuda{
        template <typename T, typename TI>
        __global__ void concatenate_kernel(const T* a, const T* b, T* output, TI last_dim_a, TI last_dim_b, TI total_elements){
            TI idx = blockIdx.x * blockDim.x + threadIdx.x;
            TI last_dim_out = last_dim_a + last_dim_b;
            if(idx < total_elements){
                TI row = idx / last_dim_out;
                TI col = idx % last_dim_out;
                if(col < last_dim_a){
                    output[idx] = a[row * last_dim_a + col];
                }
                else{
                    output[idx] = b[row * last_dim_b + (col - last_dim_a)];
                }
            }
        }

        template <typename T, typename TI>
        __global__ void split_kernel(const T* d_output, T* d_a, T* d_b, TI last_dim_a, TI last_dim_b, TI total_elements){
            TI idx = blockIdx.x * blockDim.x + threadIdx.x;
            TI last_dim_out = last_dim_a + last_dim_b;
            if(idx < total_elements){
                TI row = idx / last_dim_out;
                TI col = idx % last_dim_out;
                if(col < last_dim_a){
                    d_a[row * last_dim_a + col] = d_output[idx];
                }
                else{
                    d_b[row * last_dim_b + (col - last_dim_a)] = d_output[idx];
                }
            }
        }
    }

    template <typename DEV_SPEC, typename A_SPEC, typename B_SPEC, typename OUTPUT_SPEC>
    void _concatenate_cuda(devices::CUDA<DEV_SPEC>& device, const Tensor<A_SPEC>& a, const Tensor<B_SPEC>& b, Tensor<OUTPUT_SPEC>& output){
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        using T = typename OUTPUT_SPEC::T;
        using A_SHAPE = typename A_SPEC::SHAPE;
        using B_SHAPE = typename B_SPEC::SHAPE;
        using OUTPUT_SHAPE = typename OUTPUT_SPEC::SHAPE;
        constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
        constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
        constexpr TI LAST_DIM_OUT = get_last(OUTPUT_SHAPE{});
        static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
        constexpr TI TOTAL = product(OUTPUT_SHAPE{});
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI GRID_SIZE = (TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
        nn_models::parallel::cuda::concatenate_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, device.stream>>>(
            data(a), data(b), data(output), LAST_DIM_A, LAST_DIM_B, TOTAL);
    }

    template <typename DEV_SPEC, typename D_OUTPUT_SPEC, typename D_A_SPEC, typename D_B_SPEC>
    void _split_cuda(devices::CUDA<DEV_SPEC>& device, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_A_SPEC>& d_a, Tensor<D_B_SPEC>& d_b){
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        using T = typename D_OUTPUT_SPEC::T;
        using A_SHAPE = typename D_A_SPEC::SHAPE;
        using B_SHAPE = typename D_B_SPEC::SHAPE;
        using OUTPUT_SHAPE = typename D_OUTPUT_SPEC::SHAPE;
        constexpr TI LAST_DIM_A = get_last(A_SHAPE{});
        constexpr TI LAST_DIM_B = get_last(B_SHAPE{});
        constexpr TI LAST_DIM_OUT = get_last(OUTPUT_SHAPE{});
        static_assert(LAST_DIM_OUT == LAST_DIM_A + LAST_DIM_B);
        constexpr TI TOTAL = product(OUTPUT_SHAPE{});
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI GRID_SIZE = (TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
        nn_models::parallel::cuda::split_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, device.stream>>>(
            data(d_output), data(d_a), data(d_b), LAST_DIM_A, LAST_DIM_B, TOTAL);
    }

    // ======================== CUDA evaluate ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, model.pipeline_a, input_a, buffer.intermediate_a, buffer.buffer_a, rng, mode);
        evaluate(device, model.pipeline_b, input_b, buffer.intermediate_b, buffer.buffer_b, rng, mode);
        _concatenate_cuda(device, buffer.intermediate_a, buffer.intermediate_b, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            evaluate(device, model.head, buffer.concatenated, output, buffer.head_buffer, rng, mode);
        }
        else{
            copy(device, device, buffer.concatenated, output);
        }
    }

    // ======================== CUDA forward (no output) ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_A, typename INPUT_B, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model.pipeline_a, input_a, buffer.buffer_a, rng, mode);
        forward(device, model.pipeline_b, input_b, buffer.buffer_b, rng, mode);
        auto output_a = rl_tools::output(device, model.pipeline_a);
        auto output_b = rl_tools::output(device, model.pipeline_b);
        copy(device, device, output_a, buffer.intermediate_a);
        copy(device, device, output_b, buffer.intermediate_b);
        _concatenate_cuda(device, buffer.intermediate_a, buffer.intermediate_b, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            forward(device, model.head, buffer.concatenated, buffer.head_buffer, rng, mode);
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        }
        else{
            copy(device, device, buffer.concatenated, model.output);
        }
    }

    // ======================== CUDA forward (with output) ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model, input_a, input_b, buffer, rng, mode);
        copy(device, device, model.output, output);
    }

    // ======================== CUDA backward_full ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_cuda(device, buffer.d_concatenated, buffer.d_output_a, buffer.d_output_b);
        }
        else{
            _split_cuda(device, d_output, buffer.d_output_a, buffer.d_output_b);
        }
        backward_full(device, model.pipeline_a, input_a, buffer.d_output_a, d_input_a, buffer.buffer_a, mode);
        backward_full(device, model.pipeline_b, input_b, buffer.d_output_b, d_input_b, buffer.buffer_b, mode);
    }

    // ======================== CUDA backward (gradients only) ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, D_OUTPUT& d_output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_cuda(device, buffer.d_concatenated, buffer.d_output_a, buffer.d_output_b);
        }
        else{
            _split_cuda(device, d_output, buffer.d_output_a, buffer.d_output_b);
        }
        backward(device, model.pipeline_a, input_a, buffer.d_output_a, buffer.buffer_a, mode);
        backward(device, model.pipeline_b, input_b, buffer.d_output_b, buffer.buffer_b, mode);
    }

    // ======================== CUDA backward_input ========================
    template <typename DEV_SPEC, typename SPEC, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleBackward<SPEC>& model, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_input(device, model.head, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_cuda(device, buffer.d_concatenated, buffer.d_output_a, buffer.d_output_b);
        }
        else{
            _split_cuda(device, d_output, buffer.d_output_a, buffer.d_output_b);
        }
        backward_input(device, model.pipeline_a, buffer.d_output_a, d_input_a, buffer.buffer_a, mode);
        backward_input(device, model.pipeline_b, buffer.d_output_b, d_input_b, buffer.buffer_b, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
