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
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::avg_pool2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd;
        check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor avgpool.xd");
        check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, C, IH, IW), "cudnnSetTensor4dDescriptor avgpool.xd");
        check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor avgpool.yd");
        check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, C, 1, 1), "cudnnSetTensor4dDescriptor avgpool.yd");
        cudnnPoolingDescriptor_t pd;
        check_cudnn_call(device, cudnnCreatePoolingDescriptor(&pd), "cudnnCreatePoolingDescriptor avgpool.pd");
        check_cudnn_call(device, cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, IH, IW, 0, 0, 1, 1), "cudnnSetPooling2dDescriptor avgpool.pd");
        T a = 1, b = 0;
        check_cudnn_call(device, cudnnPoolingForward(device.cudnn_handle, pd, &a, xd, input._data, &b, yd, output._data), "cudnnPoolingForward avgpool");
        check_cudnn_call(device, cudnnDestroyPoolingDescriptor(pd), "cudnnDestroyPoolingDescriptor avgpool.pd");
        check_cudnn_call(device, cudnnDestroyTensorDescriptor(xd), "cudnnDestroyTensorDescriptor avgpool.xd");
        check_cudnn_call(device, cudnnDestroyTensorDescriptor(yd), "cudnnDestroyTensorDescriptor avgpool.yd");
        check_status(device);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::avg_pool2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::avg_pool2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename D_OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd;
        check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor avgpool_bwd.xd");
        check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, C, IH, IW), "cudnnSetTensor4dDescriptor avgpool_bwd.xd");
        check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor avgpool_bwd.yd");
        check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, C, 1, 1), "cudnnSetTensor4dDescriptor avgpool_bwd.yd");
        cudnnPoolingDescriptor_t pd;
        check_cudnn_call(device, cudnnCreatePoolingDescriptor(&pd), "cudnnCreatePoolingDescriptor avgpool_bwd.pd");
        check_cudnn_call(device, cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, IH, IW, 0, 0, 1, 1), "cudnnSetPooling2dDescriptor avgpool_bwd.pd");
        T a = 1, b = 0;
        check_cudnn_call(device, cudnnPoolingBackward(device.cudnn_handle, pd, &a, yd, layer.output._data, yd, d_output._data, xd, input._data, &b, xd, d_input._data), "cudnnPoolingBackward avgpool");
        check_cudnn_call(device, cudnnDestroyPoolingDescriptor(pd), "cudnnDestroyPoolingDescriptor avgpool_bwd.pd");
        check_cudnn_call(device, cudnnDestroyTensorDescriptor(xd), "cudnnDestroyTensorDescriptor avgpool_bwd.xd");
        check_cudnn_call(device, cudnnDestroyTensorDescriptor(yd), "cudnnDestroyTensorDescriptor avgpool_bwd.yd");
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
