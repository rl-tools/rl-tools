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
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::resnet_block::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::resnet_block::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE::index_t;

        constexpr TI N  = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI OUTPUT_SIZE = N * OH * OW * OC;

        // Conv1: input -> buffer.intermediate (conv + BN + ReLU)
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV1_SPEC>&>(layer.conv1), input, buffer.intermediate, buffer.conv1_buffer, rng, mode);

        // Conv2: buffer.intermediate -> output (conv + BN + identity activation)
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV2_SPEC>&>(layer.conv2), buffer.intermediate, output, buffer.conv2_buffer, rng, mode);

        // Downsample shortcut if needed
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
            evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::DOWNSAMPLE_SPEC>&>(layer.downsample.conv), input, buffer.shortcut, buffer.downsample_buffer, rng, mode);
        }

        // Fused add + ReLU: output = max(0, output + shortcut)
        {
            constexpr TI BLOCKSIZE = 256;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OUTPUT_SIZE, BLOCKSIZE);
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};

            const T* shortcut_ptr;
            if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE){
                shortcut_ptr = buffer.shortcut._data;
            }
            else{
                shortcut_ptr = input._data;
            }
            nn::layers::resnet_block::cuda::kernels::add_relu<<<grid, block, 0, device.stream>>>(tag_device, output._data, shortcut_ptr, OUTPUT_SIZE);
            check_status(device);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
