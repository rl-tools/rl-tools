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
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd;
        cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, C, IH, IW);
        cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, C, 1, 1);
        cudnnPoolingDescriptor_t pd; cudnnCreatePoolingDescriptor(&pd);
        cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, IH, IW, 0, 0, 1, 1);
        T a = 1, b = 0;
        cudnnPoolingForward(device.cudnn_handle, pd, &a, xd, input._data, &b, yd, output._data);
        cudnnDestroyPoolingDescriptor(pd); cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd);
        check_status(device);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::avg_pool2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::avg_pool2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename D_OUTPUT_SPEC::T;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE, IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd;
        cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, C, IH, IW);
        cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, C, 1, 1);
        cudnnPoolingDescriptor_t pd; cudnnCreatePoolingDescriptor(&pd);
        cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, IH, IW, 0, 0, 1, 1);
        T a = 1, b = 0;
        cudnnPoolingBackward(device.cudnn_handle, pd, &a, yd, layer.output._data, yd, d_output._data, xd, input._data, &b, xd, d_input._data);
        cudnnDestroyPoolingDescriptor(pd); cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
