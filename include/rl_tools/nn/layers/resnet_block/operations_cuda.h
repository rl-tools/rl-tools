#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_RESNET_BLOCK_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_RESNET_BLOCK_OPERATIONS_CUDA_H
#include "../../../devices/cuda.h"
#include "../../../nn/nn.h"
#include "../../../mode/mode.h"
#include "../conv2d/operations_cuda.h"
#include "layer.h"
#include <cudnn.h>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::resnet_block::cuda::kernels{
        template<typename DEV_SPEC, typename T>
        __global__
        void add_relu(devices::CUDA<DEV_SPEC> device, T* output, const T* shortcut, unsigned int n){
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < n){
                T val = output[i] + shortcut[i];
                output[i] = val > T(0) ? val : T(0);
            }
        }
        template<typename DEV_SPEC, typename T>
        __global__
        void d_relu_and_add(devices::CUDA<DEV_SPEC> device, const T* output, const T* d_output, T* d_pre_relu, unsigned int n){
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < n){
                d_pre_relu[i] = output[i] > T(0) ? d_output[i] : T(0);
            }
        }
        template<typename DEV_SPEC, typename T>
        __global__
        void add_inplace(devices::CUDA<DEV_SPEC> device, T* dst, const T* src, unsigned int n){
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < n){
                dst[i] += src[i];
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::resnet_block::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::resnet_block::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI OUTPUT_SIZE = N * OH * OW * OC;
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV1_SPEC>&>(layer.conv1), input, buffer.intermediate, buffer.conv1_buffer, rng, mode);
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV2_SPEC>&>(layer.conv2), buffer.intermediate, output, buffer.conv2_buffer, rng, mode);
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
            evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::DOWNSAMPLE_SPEC>&>(layer.downsample.conv), input, buffer.shortcut, buffer.downsample_buffer, rng, mode);
        }
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_SIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        const T* shortcut_ptr = LAYER_SPEC::HAS_DOWNSAMPLE ? buffer.shortcut._data : input._data;
        nn::layers::resnet_block::cuda::kernels::add_relu<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, output._data, shortcut_ptr, OUTPUT_SIZE);
        check_status(device);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::resnet_block::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::resnet_block::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI OUTPUT_SIZE = N * OH * OW * OC;
        forward(device, layer.conv1, input, buffer.intermediate, buffer.conv1_buffer, rng, mode);
        forward(device, layer.conv2, buffer.intermediate, output, buffer.conv2_buffer, rng, mode);
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
            forward(device, layer.downsample.conv, input, buffer.shortcut, buffer.downsample_buffer, rng, mode);
        }
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_SIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        const T* shortcut_ptr = LAYER_SPEC::HAS_DOWNSAMPLE ? buffer.shortcut._data : input._data;
        nn::layers::resnet_block::cuda::kernels::add_relu<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, output._data, shortcut_ptr, OUTPUT_SIZE);
        check_status(device);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::resnet_block::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        using TI = typename DEVICE::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI OUTPUT_SIZE = N * OH * OW * OC;
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_SIZE, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::resnet_block::cuda::kernels::d_relu_and_add<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, layer.output._data, d_output._data, buffer.shortcut._data, OUTPUT_SIZE);
        auto conv1_out = rl_tools::output(device, layer.conv1);
        backward(device, layer.conv2, conv1_out, buffer.shortcut, buffer.conv2_buffer, mode);
        auto d_conv1_out = buffer.intermediate;
        backward_input(device, static_cast<const nn::layers::conv2d::LayerBackward<typename LAYER_SPEC::CONV2_SPEC>&>(layer.conv2), buffer.shortcut, d_conv1_out, buffer.conv2_buffer, mode);
        backward(device, layer.conv1, input, d_conv1_out, buffer.conv1_buffer, mode);
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
            backward(device, layer.downsample.conv, input, buffer.shortcut, buffer.downsample_buffer, mode);
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::resnet_block::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        using TI = typename DEVICE::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OUTPUT_SIZE = N * OH * OW * OC;
        constexpr TI INPUT_SIZE = N * IH * IW * IC;
        constexpr TI BLOCKSIZE = 256;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        // d_pre_relu = d_output * relu'(output)
        nn::layers::resnet_block::cuda::kernels::d_relu_and_add<<<RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_SIZE, BLOCKSIZE), BLOCKSIZE, 0, device.stream>>>(tag_device, layer.output._data, d_output._data, buffer.shortcut._data, OUTPUT_SIZE);
        // backward_full through conv2
        auto conv1_out = rl_tools::output(device, layer.conv1);
        auto& d_pre_relu = buffer.shortcut;
        backward_full(device, layer.conv2, conv1_out, d_pre_relu, buffer.intermediate, buffer.conv2_buffer, mode);
        // backward_full through conv1
        backward_full(device, layer.conv1, input, buffer.intermediate, d_input, buffer.conv1_buffer, mode);
        // shortcut path
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
            backward_full(device, layer.downsample.conv, input, d_pre_relu, buffer.d_input_buffer, buffer.downsample_buffer, mode);
            nn::layers::resnet_block::cuda::kernels::add_inplace<<<RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_SIZE, BLOCKSIZE), BLOCKSIZE, 0, device.stream>>>(tag_device, d_input._data, buffer.d_input_buffer._data, INPUT_SIZE);
        } else {
            nn::layers::resnet_block::cuda::kernels::add_inplace<<<RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_SIZE, BLOCKSIZE), BLOCKSIZE, 0, device.stream>>>(tag_device, d_input._data, d_pre_relu._data, INPUT_SIZE);
        }
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC>
    void zero_gradient(devices::CUDA<DEV_SPEC>& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer) {
        zero_gradient(device, layer.conv1);
        zero_gradient(device, layer.conv2);
        zero_gradient(device, layer.downsample);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
