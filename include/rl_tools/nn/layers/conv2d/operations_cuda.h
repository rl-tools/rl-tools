#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../nn/nn.h"
#include "../../../mode/mode.h"
#include "layer.h"

#include <cudnn.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;

        constexpr TI N  = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI KH = LAYER_SPEC::KERNEL_HEIGHT;
        constexpr TI KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr TI SH = LAYER_SPEC::STRIDE_H;
        constexpr TI SW = LAYER_SPEC::STRIDE_W;
        constexpr TI PH = LAYER_SPEC::PADDING_H;
        constexpr TI PW = LAYER_SPEC::PADDING_W;

        constexpr cudnnDataType_t cudnn_dtype = nn::cuda::get_cudnn_dtype<T>();

        // Input tensor descriptor (NHWC)
        cudnnTensorDescriptor_t x_desc;
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, IC, IH, IW);

        // Output tensor descriptor (NHWC)
        cudnnTensorDescriptor_t y_desc;
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, OC, OH, OW);

        // Filter descriptor (NCHW: weights stored as [OC, IC, KH, KW])
        cudnnFilterDescriptor_t w_desc;
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnSetFilter4dDescriptor(w_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, OC, IC, KH, KW);

        // Convolution descriptor
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, cudnn_dtype);

        // Find best algorithm
        int algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t algo_perf;
        cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, x_desc, w_desc, conv_desc, y_desc, 1, &algo_count, &algo_perf);
        cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

        // Ensure workspace
        size_t ws_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size);
        if(ws_size > 0){
            ensure_cudnn_workspace(device, ws_size);
        }

        // Convolution forward
        {
            T alpha = 1, beta = 0;
            cudnnStatus_t stat = cudnnConvolutionForward(device.cudnn_handle,
                &alpha, x_desc, input._data,
                w_desc, layer.weights.parameters._data,
                conv_desc, algo,
                device.cudnn_workspace, device.cudnn_workspace_size,
                &beta, y_desc, output._data);
            if(stat != CUDNN_STATUS_SUCCESS){
                std::cerr << "cuDNN convolution forward failed: " << cudnnGetErrorString(stat) << std::endl;
            }
        }

        // Add bias (output += bias, broadcast over N, H, W)
        {
            cudnnTensorDescriptor_t b_desc;
            cudnnCreateTensorDescriptor(&b_desc);
            cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, 1, OC, 1, 1);
            T alpha = 1, beta = 1;
            cudnnAddTensor(device.cudnn_handle, &alpha, b_desc, layer.biases.parameters._data, &beta, y_desc, output._data);
            cudnnDestroyTensorDescriptor(b_desc);
        }

        // Batch normalization (inference: use running statistics)
        if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            cudnnTensorDescriptor_t bn_desc;
            cudnnCreateTensorDescriptor(&bn_desc);
            cudnnDeriveBNTensorDescriptor(bn_desc, y_desc, CUDNN_BATCHNORM_SPATIAL);
            T alpha = 1, beta = 0;
            cudnnStatus_t stat = cudnnBatchNormalizationForwardInference(
                device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                y_desc, output._data,
                y_desc, output._data,
                bn_desc,
                layer.norm.gamma.parameters._data,
                layer.norm.beta.parameters._data,
                layer.norm.running_mean.parameters._data,
                layer.norm.running_var.parameters._data,
                (double)LAYER_SPEC::NORM_EPSILON);
            if(stat != CUDNN_STATUS_SUCCESS){
                std::cerr << "cuDNN batch norm inference failed: " << cudnnGetErrorString(stat) << std::endl;
            }
            cudnnDestroyTensorDescriptor(bn_desc);
        }

        // Activation
        if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
            cudnnActivationDescriptor_t act_desc;
            cudnnCreateActivationDescriptor(&act_desc);
            cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
            T alpha = 1, beta = 0;
            cudnnActivationForward(device.cudnn_handle, act_desc,
                &alpha, y_desc, output._data,
                &beta, y_desc, output._data);
            cudnnDestroyActivationDescriptor(act_desc);
        }

        // Cleanup descriptors
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);

        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
