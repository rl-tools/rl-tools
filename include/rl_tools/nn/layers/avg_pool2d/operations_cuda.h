#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_AVG_POOL2D_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_AVG_POOL2D_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../nn/nn.h"
#include "../../../mode/mode.h"
#include "layer.h"

#include <cudnn.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::avg_pool2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        constexpr TI N  = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI C  = LAYER_SPEC::INPUT_CHANNELS;

        constexpr cudnnDataType_t cudnn_dtype = nn::cuda::get_cudnn_dtype<T>();

        // Input tensor descriptor (NHWC)
        cudnnTensorDescriptor_t x_desc;
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, C, IH, IW);

        // Output tensor descriptor: global avg pool produces (N, C, 1, 1) which is stored as (N, C) contiguously
        cudnnTensorDescriptor_t y_desc;
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, C, 1, 1);

        // Global average pooling: kernel covers the full spatial extent
        cudnnPoolingDescriptor_t pool_desc;
        cudnnCreatePoolingDescriptor(&pool_desc);
        cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, IH, IW, 0, 0, 1, 1);

        // Pooling forward
        T alpha = 1, beta = 0;
        cudnnStatus_t stat = cudnnPoolingForward(device.cudnn_handle, pool_desc,
            &alpha, x_desc, input._data,
            &beta, y_desc, output._data);
        if(stat != CUDNN_STATUS_SUCCESS){
            std::cerr << "cuDNN avg pooling forward failed: " << cudnnGetErrorString(stat) << std::endl;
        }

        // Cleanup
        cudnnDestroyPoolingDescriptor(pool_desc);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);

        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
