#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_UPSAMPLE2D_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_UPSAMPLE2D_OPERATIONS_CUDA_H
#include "../../../devices/cuda.h"
#include "../../../nn/nn.h"
#include "../../../mode/mode.h"
#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::upsample2d::cuda {
    template<typename T, typename TI>
    __global__ void bilinear_upsample_forward(
        const T* __restrict__ input, T* __restrict__ output,
        TI N, TI IH, TI IW, TI OH, TI OW, TI C
    ) {
        TI idx = (TI)blockIdx.x * blockDim.x + threadIdx.x;
        TI total = N * OH * OW * C;
        if (idx >= total) return;
        TI c_i = idx % C;
        TI ox = (idx / C) % OW;
        TI oy = (idx / (C * OW)) % OH;
        TI bi = idx / (C * OW * OH);
        float src_y = ((float)oy + 0.5f) * (float)IH / (float)OH - 0.5f;
        float src_x = ((float)ox + 0.5f) * (float)IW / (float)OW - 0.5f;
        int iy0 = src_y < 0.f ? 0 : (int)src_y;
        int ix0 = src_x < 0.f ? 0 : (int)src_x;
        int iy1 = iy0 + 1 < (int)IH ? iy0 + 1 : (int)IH - 1;
        int ix1 = ix0 + 1 < (int)IW ? ix0 + 1 : (int)IW - 1;
        if (iy0 >= (int)IH) iy0 = (int)IH - 1;
        if (ix0 >= (int)IW) ix0 = (int)IW - 1;
        float fy = src_y - (float)iy0;
        float fx = src_x - (float)ix0;
        if (fy < 0.f) fy = 0.f;
        if (fx < 0.f) fx = 0.f;
        const T* base = input + (size_t)bi * IH * IW * C;
        float v = (1.f - fy) * (1.f - fx) * (float)base[((size_t)iy0 * IW + ix0) * C + c_i]
                + (1.f - fy) * fx          * (float)base[((size_t)iy0 * IW + ix1) * C + c_i]
                + fy          * (1.f - fx) * (float)base[((size_t)iy1 * IW + ix0) * C + c_i]
                + fy          * fx          * (float)base[((size_t)iy1 * IW + ix1) * C + c_i];
        output[idx] = (T)v;
    }

    // Input-centric backward: one thread per input element, gathers from output pixels
    // that reference it. Deterministic — no atomicAdd needed.
    template<typename T, typename TI>
    __global__ void bilinear_upsample_backward(
        const T* __restrict__ d_output, T* __restrict__ d_input,
        TI N, TI IH, TI IW, TI OH, TI OW, TI C
    ) {
        TI idx = (TI)blockIdx.x * blockDim.x + threadIdx.x;
        TI total = N * IH * IW * C;
        if (idx >= total) return;
        TI c_i = idx % C;
        TI ix = (idx / C) % IW;
        TI iy = (idx / (C * IW)) % IH;
        TI bi = idx / (C * IW * IH);
        float oy_lo = ((float)iy - 0.5f) * (float)OH / (float)IH - 0.5f;
        float oy_hi = ((float)iy + 1.5f) * (float)OH / (float)IH - 0.5f;
        int oy_start = 0 > (int)ceilf(oy_lo) ? 0 : (int)ceilf(oy_lo);
        int oy_end = (int)OH < (int)ceilf(oy_hi) ? (int)OH : (int)ceilf(oy_hi);
        float ox_lo = ((float)ix - 0.5f) * (float)OW / (float)IW - 0.5f;
        float ox_hi = ((float)ix + 1.5f) * (float)OW / (float)IW - 0.5f;
        int ox_start = 0 > (int)ceilf(ox_lo) ? 0 : (int)ceilf(ox_lo);
        int ox_end = (int)OW < (int)ceilf(ox_hi) ? (int)OW : (int)ceilf(ox_hi);
        float acc = 0.f;
        const T* d_out_base = d_output + (size_t)bi * OH * OW * C;
        for (int oy = oy_start; oy < oy_end; oy++) {
            float src_y = ((float)oy + 0.5f) * (float)IH / (float)OH - 0.5f;
            int iy0 = src_y < 0.f ? 0 : (int)src_y;
            int iy1 = iy0 + 1 < (int)IH ? iy0 + 1 : (int)IH - 1;
            if (iy0 >= (int)IH) iy0 = (int)IH - 1;
            float fy = src_y - (float)iy0;
            if (fy < 0.f) fy = 0.f;
            float wy;
            if ((int)iy == iy0 && (int)iy == iy1) wy = 1.f;
            else if ((int)iy == iy0) wy = 1.f - fy;
            else if ((int)iy == iy1) wy = fy;
            else continue;
            for (int ox = ox_start; ox < ox_end; ox++) {
                float src_x = ((float)ox + 0.5f) * (float)IW / (float)OW - 0.5f;
                int ix0 = src_x < 0.f ? 0 : (int)src_x;
                int ix1 = ix0 + 1 < (int)IW ? ix0 + 1 : (int)IW - 1;
                if (ix0 >= (int)IW) ix0 = (int)IW - 1;
                float fx = src_x - (float)ix0;
                if (fx < 0.f) fx = 0.f;
                float wx;
                if ((int)ix == ix0 && (int)ix == ix1) wx = 1.f;
                else if ((int)ix == ix0) wx = 1.f - fx;
                else if ((int)ix == ix1) wx = fx;
                else continue;
                acc += wy * wx * (float)d_out_base[((size_t)oy * OW + ox) * C + c_i];
            }
        }
        d_input[idx] = (T)acc;
    }
}

namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::upsample2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::upsample2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI total = N * OH * OW * C;
        constexpr TI BLOCK = 256;
        constexpr TI GRID = (total + BLOCK - 1) / BLOCK;
        nn::layers::upsample2d::cuda::bilinear_upsample_forward<T, TI><<<GRID, BLOCK, 0, device.stream>>>(
            input._data, output._data, N, IH, IW, OH, OW, C);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::upsample2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::upsample2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::upsample2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::upsample2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename D_OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI total = N * IH * IW * C;
        constexpr TI BLOCK = 256;
        constexpr TI GRID = (total + BLOCK - 1) / BLOCK;
        nn::layers::upsample2d::cuda::bilinear_upsample_backward<T, TI><<<GRID, BLOCK, 0, device.stream>>>(
            d_output._data, d_input._data, N, IH, IW, OH, OW, C);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
