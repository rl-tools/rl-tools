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
        __global__ void copy_to_concat_kernel(const T* src, T* dst, TI src_dim, TI dst_dim, TI offset, TI num_elements){
            TI idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < num_elements){
                TI row = idx / src_dim;
                TI col = idx % src_dim;
                dst[row * dst_dim + offset + col] = src[idx];
            }
        }

        template <typename T, typename TI>
        __global__ void copy_from_concat_kernel(const T* src, T* dst, TI src_dim, TI dst_dim, TI offset, TI num_elements){
            TI idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < num_elements){
                TI row = idx / dst_dim;
                TI col = idx % dst_dim;
                dst[idx] = src[row * src_dim + offset + col];
            }
        }
    }

    template <auto NUM_BRANCHES, auto I = 0, auto OFFSET = 0, typename DEV_SPEC, typename INTERMEDIATES, typename OUTPUT_SPEC>
    void _concatenate_n_cuda(devices::CUDA<DEV_SPEC>& device, const INTERMEDIATES& intermediates, Tensor<OUTPUT_SPEC>& output){
        if constexpr(I < NUM_BRANCHES){
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            using T = typename OUTPUT_SPEC::T;
            using OUTPUT_SHAPE = typename OUTPUT_SPEC::SHAPE;
            constexpr TI DST_DIM = get_last(OUTPUT_SHAPE{});
            constexpr TI LEADING = product(OUTPUT_SHAPE{}) / DST_DIM;
            const auto& src = get<I>(intermediates);
            using SRC_SHAPE = typename utils::typing::remove_reference_t<decltype(src)>::SPEC::SHAPE;
            constexpr TI SRC_DIM = get_last(SRC_SHAPE{});
            constexpr TI TOTAL_SRC = LEADING * SRC_DIM;
            constexpr TI BLOCK_SIZE = 256;
            constexpr TI GRID_SIZE = (TOTAL_SRC + BLOCK_SIZE - 1) / BLOCK_SIZE;
            nn_models::parallel::cuda::copy_to_concat_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, device.stream>>>(
                data(src), data(output), SRC_DIM, DST_DIM, static_cast<TI>(OFFSET), TOTAL_SRC);
            _concatenate_n_cuda<NUM_BRANCHES, I + 1, OFFSET + SRC_DIM>(device, intermediates, output);
        }
    }

    template <auto NUM_BRANCHES, auto I = 0, auto OFFSET = 0, typename DEV_SPEC, typename D_OUTPUT_SPEC, typename D_OUTPUTS>
    void _split_n_cuda(devices::CUDA<DEV_SPEC>& device, const Tensor<D_OUTPUT_SPEC>& d_output, D_OUTPUTS& d_outputs){
        if constexpr(I < NUM_BRANCHES){
            using TI = typename devices::CUDA<DEV_SPEC>::index_t;
            using T = typename D_OUTPUT_SPEC::T;
            using SRC_SHAPE = typename D_OUTPUT_SPEC::SHAPE;
            constexpr TI SRC_DIM = get_last(SRC_SHAPE{});
            auto& dst = get<I>(d_outputs);
            using DST_SHAPE = typename utils::typing::remove_reference_t<decltype(dst)>::SPEC::SHAPE;
            constexpr TI DST_DIM = get_last(DST_SHAPE{});
            constexpr TI TOTAL_DST = product(DST_SHAPE{});
            constexpr TI BLOCK_SIZE = 256;
            constexpr TI GRID_SIZE = (TOTAL_DST + BLOCK_SIZE - 1) / BLOCK_SIZE;
            nn_models::parallel::cuda::copy_from_concat_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, device.stream>>>(
                data(d_output), data(dst), SRC_DIM, DST_DIM, static_cast<TI>(OFFSET), TOTAL_DST);
            _split_n_cuda<NUM_BRANCHES, I + 1, OFFSET + DST_DIM>(device, d_output, d_outputs);
        }
    }

    // ======================== CUDA evaluate_step ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename STATE_SPEC, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate_step(devices::CUDA<DEV_SPEC>& device, const nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, nn_models::parallel::ModuleState<STATE_SPEC>& state, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename SPEC::TI;
        using OUTPUT_TENSOR_SHAPE = typename OUTPUT::SPEC::SHAPE;
        constexpr TI OUTPUT_LAST_DIM = get_last(OUTPUT_TENSOR_SHAPE{});
        constexpr TI LEADING = product(OUTPUT_TENSOR_SHAPE{}) / OUTPUT_LAST_DIM;
        using CONCAT_SHAPE = typename utils::typing::remove_reference_t<decltype(buffer.concatenated)>::SPEC::SHAPE;
        constexpr TI CONCAT_LAST_DIM = get_last(CONCAT_SHAPE{});
        auto concat_view = view_memory<tensor::Shape<TI, LEADING, CONCAT_LAST_DIM>>(device, buffer.concatenated);
        nn_models::parallel::_evaluate_step_branches(device, model, inputs, state, concat_view, buffer, rng, mode);
        if constexpr(SPEC::HAS_HEAD){
            auto output_2d = reshape_row_major(device, output, tensor::Shape<TI, LEADING, OUTPUT_LAST_DIM>{});
            evaluate_step(device, model.head, concat_view, state.head_state, output_2d, buffer.head_buffer, rng, mode);
        }
        else{
            copy(device, device, concat_view, output);
        }
    }

    // ======================== CUDA evaluate ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_TUPLE& inputs, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::parallel::_evaluate_branches(device, model, inputs, buffer, rng, mode);
        _concatenate_n_cuda<SPEC::NUM_BRANCHES>(device, buffer.intermediates, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            evaluate(device, model.head, buffer.concatenated, output, buffer.head_buffer, rng, mode);
        }
        else{
            copy(device, device, buffer.concatenated, output);
        }
    }

    // ======================== CUDA forward ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_TUPLE& inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::parallel::_forward_branches(device, model, inputs, buffer, rng, mode);
        _concatenate_n_cuda<SPEC::NUM_BRANCHES>(device, buffer.intermediates, buffer.concatenated);
        if constexpr(SPEC::HAS_HEAD){
            forward(device, model.head, buffer.concatenated, buffer.head_buffer, rng, mode);
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        }
        else{
            copy(device, device, buffer.concatenated, model.output);
        }
    }

    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_TUPLE& inputs, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, model, inputs, buffer, rng, mode);
        copy(device, device, model.output, output);
    }

    // ======================== CUDA backward_full ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename D_OUTPUT, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, D_OUTPUT& d_output, D_INPUT_TUPLE& d_inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_full_branches(device, model, inputs, d_inputs, buffer, mode);
    }

    // ======================== CUDA backward (gradients only) ========================
    template <typename DEV_SPEC, typename SPEC, typename INPUT_TUPLE, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleGradient<SPEC>& model, const INPUT_TUPLE& inputs, D_OUTPUT& d_output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_branches(device, model, inputs, buffer, mode);
    }

    // ======================== CUDA backward_input ========================
    template <typename DEV_SPEC, typename SPEC, typename D_OUTPUT, typename D_INPUT_TUPLE, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, nn_models::parallel::ModuleBackward<SPEC>& model, D_OUTPUT& d_output, D_INPUT_TUPLE& d_inputs, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(SPEC::HAS_HEAD){
            backward_input(device, model.head, d_output, buffer.d_concatenated, buffer.head_buffer, mode);
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, buffer.d_concatenated, buffer.d_outputs);
        }
        else{
            _split_n_cuda<SPEC::NUM_BRANCHES>(device, d_output, buffer.d_outputs);
        }
        nn_models::parallel::_backward_input_branches(device, model, d_inputs, buffer, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
