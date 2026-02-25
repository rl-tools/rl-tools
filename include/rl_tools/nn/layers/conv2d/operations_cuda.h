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

        cudnnTensorDescriptor_t x_desc;
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, IC, IH, IW);

        cudnnTensorDescriptor_t y_desc;
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, N, OC, OH, OW);

        cudnnFilterDescriptor_t w_desc;
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnSetFilter4dDescriptor(w_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, OC, IC, KH, KW);

        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, cudnn_dtype);

        int algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t algo_perf;
        cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, x_desc, w_desc, conv_desc, y_desc, 1, &algo_count, &algo_perf);
        cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

        size_t ws_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size);
        if(ws_size > 0){
            ensure_cudnn_workspace(device, ws_size);
        }

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

        {
            cudnnTensorDescriptor_t b_desc;
            cudnnCreateTensorDescriptor(&b_desc);
            cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NHWC, cudnn_dtype, 1, OC, 1, 1);
            T alpha = 1, beta = 1;
            cudnnAddTensor(device.cudnn_handle, &alpha, b_desc, layer.biases.parameters._data, &beta, y_desc, output._data);
            cudnnDestroyTensorDescriptor(b_desc);
        }

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

        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);

        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI KH = LAYER_SPEC::KERNEL_HEIGHT, KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr TI SH = LAYER_SPEC::STRIDE_H, SW = LAYER_SPEC::STRIDE_W;
        constexpr TI PH = LAYER_SPEC::PADDING_H, PW = LAYER_SPEC::PADDING_W;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
        cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
        cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
        cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NCHW, OC, IC, KH, KW);
        cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt);
        int ac; cudnnConvolutionFwdAlgoPerf_t ap;
        cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, xd, wd, cd, yd, 1, &ac, &ap);
        size_t ws = 0; cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, xd, wd, cd, yd, ap.algo, &ws);
        if(ws > 0) ensure_cudnn_workspace(device, ws);
        { T a = 1, b = 0; cudnnConvolutionForward(device.cudnn_handle, &a, xd, input._data, wd, layer.weights.parameters._data,
            cd, ap.algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, yd, layer.pre_activations._data); }
        { cudnnTensorDescriptor_t bd; cudnnCreateTensorDescriptor(&bd);
          cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1);
          T a = 1, b = 1; cudnnAddTensor(device.cudnn_handle, &a, bd, layer.biases.parameters._data, &b, yd, layer.pre_activations._data);
          cudnnDestroyTensorDescriptor(bd); }
        if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            cudnnTensorDescriptor_t bnd; cudnnCreateTensorDescriptor(&bnd);
            cudnnDeriveBNTensorDescriptor(bnd, yd, CUDNN_BATCHNORM_SPATIAL);
            if constexpr(mode::is<MODE, mode::Evaluation>){
                T a = 1, b = 0;
                cudnnBatchNormalizationForwardInference(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                    yd, layer.pre_activations._data, yd, output._data, bnd,
                    layer.norm.gamma.parameters._data, layer.norm.beta.parameters._data,
                    layer.norm.running_mean.parameters._data, layer.norm.running_var.parameters._data, (double)LAYER_SPEC::NORM_EPSILON);
            } else {
                T a = 1, b = 0;
                cudnnBatchNormalizationForwardTraining(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                    yd, layer.pre_activations._data, yd, output._data, bnd,
                    layer.norm.gamma.parameters._data, layer.norm.beta.parameters._data,
                    (double)LAYER_SPEC::BN_MOMENTUM,
                    layer.norm.running_mean.parameters._data, layer.norm.running_var.parameters._data,
                    (double)LAYER_SPEC::NORM_EPSILON, layer.norm_cache.mean._data, layer.norm_cache.inv_std._data);
            }
            cudnnDestroyTensorDescriptor(bnd);
        } else {
            cudaMemcpyAsync(output._data, layer.pre_activations._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream);
        }
        if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
            cudnnActivationDescriptor_t ad; cudnnCreateActivationDescriptor(&ad);
            cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
            T a = 1, b = 0; cudnnActivationForward(device.cudnn_handle, ad, &a, yd, output._data, &b, yd, output._data);
            cudnnDestroyActivationDescriptor(ad);
        }
        cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd);
        cudnnDestroyFilterDescriptor(wd); cudnnDestroyConvolutionDescriptor(cd);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI KH = LAYER_SPEC::KERNEL_HEIGHT, KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr TI SH = LAYER_SPEC::STRIDE_H, SW = LAYER_SPEC::STRIDE_W;
        constexpr TI PH = LAYER_SPEC::PADDING_H, PW = LAYER_SPEC::PADDING_W;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
        cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
        cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
        cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NCHW, OC, IC, KH, KW);
        cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt);
        T* d_conv_out = layer.output._data;
        if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
            cudnnActivationDescriptor_t ad; cudnnCreateActivationDescriptor(&ad);
            cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
            T a = 1, b = 0;
            cudnnActivationBackward(device.cudnn_handle, ad, &a, yd, layer.output._data, yd, d_output._data, yd, layer.output._data, &b, yd, d_output._data);
            cudnnDestroyActivationDescriptor(ad);
        }
        if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            cudnnTensorDescriptor_t bnd; cudnnCreateTensorDescriptor(&bnd);
            cudnnDeriveBNTensorDescriptor(bnd, yd, CUDNN_BATCHNORM_SPATIAL);
            T ad1 = 1, bd1 = 0, ap1 = 1, bp1 = 1;
            cudnnBatchNormalizationBackward(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &ad1, &bd1, &ap1, &bp1,
                yd, layer.pre_activations._data, yd, d_output._data, yd, d_conv_out,
                bnd, layer.norm.gamma.parameters._data, layer.norm.gamma.gradient._data, layer.norm.beta.gradient._data,
                (double)LAYER_SPEC::NORM_EPSILON, layer.norm_cache.mean._data, layer.norm_cache.inv_std._data);
            cudnnDestroyTensorDescriptor(bnd);
        } else {
            cudaMemcpyAsync(d_conv_out, d_output._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream);
        }
        { cudnnTensorDescriptor_t bd; cudnnCreateTensorDescriptor(&bd);
          cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1);
          T a = 1, b = 1;
          cudnnConvolutionBackwardBias(device.cudnn_handle, &a, yd, d_conv_out, &b, bd, layer.biases.gradient._data);
          cudnnDestroyTensorDescriptor(bd); }
        { int ac2; cudnnConvolutionBwdFilterAlgoPerf_t ap2;
          cudnnGetConvolutionBackwardFilterAlgorithm_v7(device.cudnn_handle, xd, yd, cd, wd, 1, &ac2, &ap2);
          size_t ws = 0; cudnnGetConvolutionBackwardFilterWorkspaceSize(device.cudnn_handle, xd, yd, cd, wd, ap2.algo, &ws);
          if(ws > 0) ensure_cudnn_workspace(device, ws);
          T a = 1, b = 1;
          cudnnConvolutionBackwardFilter(device.cudnn_handle, &a, xd, input._data, yd, d_conv_out,
              cd, ap2.algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, wd, layer.weights.gradient._data); }
        { int ac3; cudnnConvolutionBwdDataAlgoPerf_t ap3;
          cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, 1, &ac3, &ap3);
          size_t ws = 0; cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, ap3.algo, &ws);
          if(ws > 0) ensure_cudnn_workspace(device, ws);
          T a = 1, b = 0;
          cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_conv_out,
              cd, ap3.algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data); }
        cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd);
        cudnnDestroyFilterDescriptor(wd); cudnnDestroyConvolutionDescriptor(cd);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::conv2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        Tensor<tensor::Specification<T, TI, typename LAYER_SPEC::INPUT_SHAPE>> d_input_tmp;
        malloc(device, d_input_tmp);
        backward_full(device, layer, input, d_output, d_input_tmp, buffer, mode);
        free(device, d_input_tmp);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, const nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI KH = LAYER_SPEC::KERNEL_HEIGHT, KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr TI SH = LAYER_SPEC::STRIDE_H, SW = LAYER_SPEC::STRIDE_W;
        constexpr TI PH = LAYER_SPEC::PADDING_H, PW = LAYER_SPEC::PADDING_W;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        cudnnTensorDescriptor_t xd, yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
        cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
        cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
        cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NCHW, OC, IC, KH, KW);
        cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt);
        int ac; cudnnConvolutionBwdDataAlgoPerf_t ap;
        cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, 1, &ac, &ap);
        size_t ws = 0; cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, ap.algo, &ws);
        if(ws > 0) ensure_cudnn_workspace(device, ws);
        T a = 1, b = 0;
        cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_output._data,
            cd, ap.algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data);
        cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd);
        cudnnDestroyFilterDescriptor(wd); cudnnDestroyConvolutionDescriptor(cd);
        check_status(device);
    }

    template<typename DEV_SPEC, typename SPEC>
    void zero_gradient(devices::CUDA<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        cudaMemsetAsync(layer.weights.gradient._data, 0, decltype(layer.weights.gradient)::SPEC::SIZE_BYTES, device.stream);
        cudaMemsetAsync(layer.biases.gradient._data, 0, decltype(layer.biases.gradient)::SPEC::SIZE_BYTES, device.stream);
        if constexpr(SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            cudaMemsetAsync(layer.norm.gamma.gradient._data, 0, decltype(layer.norm.gamma.gradient)::SPEC::SIZE_BYTES, device.stream);
            cudaMemsetAsync(layer.norm.beta.gradient._data, 0, decltype(layer.norm.beta.gradient)::SPEC::SIZE_BYTES, device.stream);
        }
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
