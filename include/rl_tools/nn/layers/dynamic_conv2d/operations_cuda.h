#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../nn/nn.h"
#include "../../../mode/mode.h"
#include "../../../utils/assert/operations_cuda.h"
#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::dynamic_conv2d::cuda::kernels{
        template <bool STORE_PRE_ACTIVATIONS, bool IS_FULL_CONV, nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION, typename T, int IH, int IW, int OC, int IC, int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW>
        __global__ void dynamic_conv2d_forward_kernel(
            const T* __restrict__ input_data,
            const T* __restrict__ kernel_weights,
            T* __restrict__ pre_activations,
            T* __restrict__ output,
            int batch_size
        ){
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total = batch_size * OH * OW * OC;
            if(idx >= total) return;
            const int c_out = idx % OC;
            const int ow = (idx / OC) % OW;
            const int oh = (idx / OC / OW) % OH;
            const int bi = idx / OC / OW / OH;
            float acc = 0.0f;
            if constexpr(IS_FULL_CONV){
                for(int c_in = 0; c_in < IC; c_in++){
                    for(int kh = 0; kh < KH; kh++){
                        for(int kw = 0; kw < KW; kw++){
                            const int ih = oh * SH + kh - PH;
                            const int iw = ow * SW + kw - PW;
                            if(ih >= 0 && ih < IH && iw >= 0 && iw < IW){
                                float d = (float)input_data[bi * IH * IW * IC + ih * IW * IC + iw * IC + c_in];
                                float w = (float)kernel_weights[bi * OC * IC * KH * KW + c_out * IC * KH * KW + c_in * KH * KW + kh * KW + kw];
                                acc += d * w;
                            }
                        }
                    }
                }
            } else {
                for(int kh = 0; kh < KH; kh++){
                    for(int kw = 0; kw < KW; kw++){
                        const int ih = oh * SH + kh - PH;
                        const int iw = ow * SW + kw - PW;
                        if(ih >= 0 && ih < IH && iw >= 0 && iw < IW){
                            float d = (float)input_data[bi * IH * IW * OC + ih * IW * OC + iw * OC + c_out];
                            float w = (float)kernel_weights[bi * OC * KH * KW + c_out * KH * KW + kh * KW + kw];
                            acc += d * w;
                        }
                    }
                }
            }
            if constexpr(STORE_PRE_ACTIVATIONS){
                pre_activations[idx] = (T)acc;
            }
            if constexpr(ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::IDENTITY){
                output[idx] = (T)acc;
            } else if constexpr(ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
                output[idx] = (T)(acc > 0.0f ? acc : 0.0f);
            } else {
                output[idx] = (T)activation<devices::math::CUDA, float, ACTIVATION_FUNCTION>(acc);
            }
        }

        template <bool COMPUTE_D_DATA, bool COMPUTE_D_KW, bool IS_FULL_CONV, nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION, typename T, int IH, int IW, int OC, int IC, int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW>
        __global__ void dynamic_conv2d_backward_kernel(
            const T* __restrict__ input_data,
            const T* __restrict__ kernel_weights,
            const T* __restrict__ pre_activations,
            const T* __restrict__ d_output,
            float* __restrict__ d_data_acc,
            float* __restrict__ d_kw_acc,
            int batch_size
        ){
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total = batch_size * OH * OW * OC;
            if(idx >= total) return;
            const int c_out = idx % OC;
            const int ow = (idx / OC) % OW;
            const int oh = (idx / OC / OW) % OH;
            const int bi = idx / OC / OW / OH;
            float pre_act = (float)pre_activations[idx];
            float d_act = d_activation_d_x<devices::math::CUDA, float, ACTIVATION_FUNCTION>(pre_act);
            float d_pre = d_act * (float)d_output[idx];
            if constexpr(IS_FULL_CONV){
                for(int c_in = 0; c_in < IC; c_in++){
                    for(int kh = 0; kh < KH; kh++){
                        for(int kw = 0; kw < KW; kw++){
                            const int ih = oh * SH + kh - PH;
                            const int iw = ow * SW + kw - PW;
                            if(ih >= 0 && ih < IH && iw >= 0 && iw < IW){
                                if constexpr(COMPUTE_D_DATA){
                                    float w_val = (float)kernel_weights[bi * OC * IC * KH * KW + c_out * IC * KH * KW + c_in * KH * KW + kh * KW + kw];
                                    atomicAdd(&d_data_acc[bi * IH * IW * IC + ih * IW * IC + iw * IC + c_in], w_val * d_pre);
                                }
                                if constexpr(COMPUTE_D_KW){
                                    float d_val = (float)input_data[bi * IH * IW * IC + ih * IW * IC + iw * IC + c_in];
                                    atomicAdd(&d_kw_acc[bi * OC * IC * KH * KW + c_out * IC * KH * KW + c_in * KH * KW + kh * KW + kw], d_val * d_pre);
                                }
                            }
                        }
                    }
                }
            } else {
                for(int kh = 0; kh < KH; kh++){
                    for(int kw = 0; kw < KW; kw++){
                        const int ih = oh * SH + kh - PH;
                        const int iw = ow * SW + kw - PW;
                        if(ih >= 0 && ih < IH && iw >= 0 && iw < IW){
                            if constexpr(COMPUTE_D_DATA){
                                float w_val = (float)kernel_weights[bi * OC * KH * KW + c_out * KH * KW + kh * KW + kw];
                                atomicAdd(&d_data_acc[bi * IH * IW * OC + ih * IW * OC + iw * OC + c_out], w_val * d_pre);
                            }
                            if constexpr(COMPUTE_D_KW){
                                float d_val = (float)input_data[bi * IH * IW * OC + ih * IW * OC + iw * OC + c_out];
                                atomicAdd(&d_kw_acc[bi * OC * KH * KW + c_out * KH * KW + kh * KW + kw], d_val * d_pre);
                            }
                        }
                    }
                }
            }
        }

        template <typename T_SRC, typename T_DST>
        __global__ void cast_acc_to_output_kernel(
            const T_SRC* __restrict__ src,
            T_DST* __restrict__ dst,
            int n
        ){
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;
            dst[idx] = (T_DST)src[idx];
        }
    }

    // ======================== evaluate ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::dynamic_conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::dynamic_conv2d::check_data_output<LAYER_SPEC, DATA_SPEC, OUTPUT_SPEC>);
        using T = typename OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr bool IS_FULL_CONV = LAYER_SPEC::IS_FULL_CONV;
        constexpr int IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr int IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr int OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr int IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr int OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr int OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr int KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr int KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr int SH = LAYER_SPEC::STRIDE_H;
        constexpr int SW = LAYER_SPEC::STRIDE_W;
        constexpr int PH = LAYER_SPEC::PADDING_H;
        constexpr int PW = LAYER_SPEC::PADDING_W;
        constexpr TI TOTAL = BATCH_SIZE * OH * OW * OC;
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BLOCK_SIZE);
        nn::layers::dynamic_conv2d::cuda::kernels::dynamic_conv2d_forward_kernel<false, IS_FULL_CONV, LAYER_SPEC::ACTIVATION_FUNCTION, T, IH, IW, OC, IC, OH, OW, KH, KW, SH, SW, PH, PW><<<N_BLOCKS, BLOCK_SIZE, 0, device.stream>>>(
            data._data,
            kernel_weights._data,
            nullptr,
            output._data,
            BATCH_SIZE
        );
        check_status(device);
    }

    // ======================== forward (LayerBackward — stores pre-activations) ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::dynamic_conv2d::check_data_output<LAYER_SPEC, DATA_SPEC, OUTPUT_SPEC>);
        using T = typename OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr bool IS_FULL_CONV = LAYER_SPEC::IS_FULL_CONV;
        constexpr int IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr int IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr int OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr int IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr int OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr int OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr int KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr int KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr int SH = LAYER_SPEC::STRIDE_H;
        constexpr int SW = LAYER_SPEC::STRIDE_W;
        constexpr int PH = LAYER_SPEC::PADDING_H;
        constexpr int PW = LAYER_SPEC::PADDING_W;
        constexpr TI TOTAL = BATCH_SIZE * OH * OW * OC;
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BLOCK_SIZE);
        nn::layers::dynamic_conv2d::cuda::kernels::dynamic_conv2d_forward_kernel<true, IS_FULL_CONV, LAYER_SPEC::ACTIVATION_FUNCTION, T, IH, IW, OC, IC, OH, OW, KH, KW, SH, SW, PH, PW><<<N_BLOCKS, BLOCK_SIZE, 0, device.stream>>>(
            data._data,
            kernel_weights._data,
            layer.pre_activations._data,
            output._data,
            BATCH_SIZE
        );
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, static_cast<nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>&>(layer), data, kernel_weights, layer.output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, layer, data, kernel_weights, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward_input ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename KERNEL_WEIGHTS_SPEC, typename D_OUTPUT_SPEC, typename D_DATA_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, const nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_DATA_SPEC>& d_data, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename D_DATA_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr bool IS_FULL_CONV = LAYER_SPEC::IS_FULL_CONV;
        constexpr int IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr int IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr int OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr int IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr int OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr int OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr int KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr int KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr int SH = LAYER_SPEC::STRIDE_H;
        constexpr int SW = LAYER_SPEC::STRIDE_W;
        constexpr int PH = LAYER_SPEC::PADDING_H;
        constexpr int PW = LAYER_SPEC::PADDING_W;
        constexpr TI OUTPUT_TOTAL = BATCH_SIZE * OH * OW * OC;
        constexpr TI INPUT_TOTAL = BATCH_SIZE * IH * IW * IC;
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI N_BLOCKS_BWD = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_TOTAL, BLOCK_SIZE);
        constexpr TI N_BLOCKS_CAST = RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_TOTAL, BLOCK_SIZE);
        check_cuda_call(device, cudaMemsetAsync(buffer.d_data_acc._data, 0, sizeof(ACCUMULATOR_TYPE) * INPUT_TOTAL, device.stream), "cudaMemsetAsync d_data_acc");
        nn::layers::dynamic_conv2d::cuda::kernels::dynamic_conv2d_backward_kernel<true, false, IS_FULL_CONV, LAYER_SPEC::ACTIVATION_FUNCTION, typename D_OUTPUT_SPEC::T, IH, IW, OC, IC, OH, OW, KH, KW, SH, SW, PH, PW><<<N_BLOCKS_BWD, BLOCK_SIZE, 0, device.stream>>>(
            (const typename D_OUTPUT_SPEC::T*)nullptr,
            kernel_weights._data,
            layer.pre_activations._data,
            d_output._data,
            buffer.d_data_acc._data,
            nullptr,
            BATCH_SIZE
        );
        nn::layers::dynamic_conv2d::cuda::kernels::cast_acc_to_output_kernel<ACCUMULATOR_TYPE, T><<<N_BLOCKS_CAST, BLOCK_SIZE, 0, device.stream>>>(
            buffer.d_data_acc._data,
            d_data._data,
            INPUT_TOTAL
        );
        check_status(device);
    }

    // ======================== backward_kernel_weights ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename D_OUTPUT_SPEC, typename D_KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_kernel_weights(devices::CUDA<DEV_SPEC>& device, const nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_KERNEL_WEIGHTS_SPEC>& d_kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename D_KERNEL_WEIGHTS_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr bool IS_FULL_CONV = LAYER_SPEC::IS_FULL_CONV;
        constexpr int IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr int IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr int OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr int IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr int OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr int OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr int KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr int KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr int SH = LAYER_SPEC::STRIDE_H;
        constexpr int SW = LAYER_SPEC::STRIDE_W;
        constexpr int PH = LAYER_SPEC::PADDING_H;
        constexpr int PW = LAYER_SPEC::PADDING_W;
        constexpr TI OUTPUT_TOTAL = BATCH_SIZE * OH * OW * OC;
        constexpr TI KW_TOTAL = IS_FULL_CONV ? (BATCH_SIZE * OC * IC * KH * KW) : (BATCH_SIZE * IC * KH * KW);
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI N_BLOCKS_BWD = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_TOTAL, BLOCK_SIZE);
        constexpr TI N_BLOCKS_CAST = RL_TOOLS_DEVICES_CUDA_CEIL(KW_TOTAL, BLOCK_SIZE);
        check_cuda_call(device, cudaMemsetAsync(buffer.d_kernel_weights_acc._data, 0, sizeof(ACCUMULATOR_TYPE) * KW_TOTAL, device.stream), "cudaMemsetAsync d_kernel_weights_acc");
        nn::layers::dynamic_conv2d::cuda::kernels::dynamic_conv2d_backward_kernel<false, true, IS_FULL_CONV, LAYER_SPEC::ACTIVATION_FUNCTION, typename D_OUTPUT_SPEC::T, IH, IW, OC, IC, OH, OW, KH, KW, SH, SW, PH, PW><<<N_BLOCKS_BWD, BLOCK_SIZE, 0, device.stream>>>(
            data._data,
            (const typename D_OUTPUT_SPEC::T*)nullptr,
            layer.pre_activations._data,
            d_output._data,
            nullptr,
            buffer.d_kernel_weights_acc._data,
            BATCH_SIZE
        );
        nn::layers::dynamic_conv2d::cuda::kernels::cast_acc_to_output_kernel<ACCUMULATOR_TYPE, T><<<N_BLOCKS_CAST, BLOCK_SIZE, 0, device.stream>>>(
            buffer.d_kernel_weights_acc._data,
            d_kernel_weights._data,
            KW_TOTAL
        );
        check_status(device);
    }

    // ======================== backward_full ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename D_OUTPUT_SPEC, typename D_DATA_SPEC, typename D_KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_DATA_SPEC>& d_data, Tensor<D_KERNEL_WEIGHTS_SPEC>& d_kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T_DATA = typename D_DATA_SPEC::T;
        using T_KW = typename D_KERNEL_WEIGHTS_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr bool IS_FULL_CONV = LAYER_SPEC::IS_FULL_CONV;
        constexpr int IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr int IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr int OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr int IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr int OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr int OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr int KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr int KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr int SH = LAYER_SPEC::STRIDE_H;
        constexpr int SW = LAYER_SPEC::STRIDE_W;
        constexpr int PH = LAYER_SPEC::PADDING_H;
        constexpr int PW = LAYER_SPEC::PADDING_W;
        constexpr TI OUTPUT_TOTAL = BATCH_SIZE * OH * OW * OC;
        constexpr TI INPUT_TOTAL = BATCH_SIZE * IH * IW * IC;
        constexpr TI KW_TOTAL = IS_FULL_CONV ? (BATCH_SIZE * OC * IC * KH * KW) : (BATCH_SIZE * IC * KH * KW);
        constexpr TI BLOCK_SIZE = 256;
        constexpr TI N_BLOCKS_BWD = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_TOTAL, BLOCK_SIZE);
        constexpr TI N_BLOCKS_CAST_DATA = RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_TOTAL, BLOCK_SIZE);
        constexpr TI N_BLOCKS_CAST_KW = RL_TOOLS_DEVICES_CUDA_CEIL(KW_TOTAL, BLOCK_SIZE);
        check_cuda_call(device, cudaMemsetAsync(buffer.d_data_acc._data, 0, sizeof(ACCUMULATOR_TYPE) * INPUT_TOTAL, device.stream), "cudaMemsetAsync d_data_acc");
        check_cuda_call(device, cudaMemsetAsync(buffer.d_kernel_weights_acc._data, 0, sizeof(ACCUMULATOR_TYPE) * KW_TOTAL, device.stream), "cudaMemsetAsync d_kernel_weights_acc");
        nn::layers::dynamic_conv2d::cuda::kernels::dynamic_conv2d_backward_kernel<true, true, IS_FULL_CONV, LAYER_SPEC::ACTIVATION_FUNCTION, typename D_OUTPUT_SPEC::T, IH, IW, OC, IC, OH, OW, KH, KW, SH, SW, PH, PW><<<N_BLOCKS_BWD, BLOCK_SIZE, 0, device.stream>>>(
            data._data,
            kernel_weights._data,
            layer.pre_activations._data,
            d_output._data,
            buffer.d_data_acc._data,
            buffer.d_kernel_weights_acc._data,
            BATCH_SIZE
        );
        nn::layers::dynamic_conv2d::cuda::kernels::cast_acc_to_output_kernel<ACCUMULATOR_TYPE, T_DATA><<<N_BLOCKS_CAST_DATA, BLOCK_SIZE, 0, device.stream>>>(
            buffer.d_data_acc._data,
            d_data._data,
            INPUT_TOTAL
        );
        nn::layers::dynamic_conv2d::cuda::kernels::cast_acc_to_output_kernel<ACCUMULATOR_TYPE, T_KW><<<N_BLOCKS_CAST_KW, BLOCK_SIZE, 0, device.stream>>>(
            buffer.d_kernel_weights_acc._data,
            d_kernel_weights._data,
            KW_TOTAL
        );
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
