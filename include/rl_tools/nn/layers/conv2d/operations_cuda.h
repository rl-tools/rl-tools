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
    namespace nn::layers::conv2d::cuda::kernels{
        template<typename T>
        __global__
        void compute_inv_std_from_running_var(const T* running_var, T* inv_std, T* dst_mean, const T* running_mean, T eps, unsigned int n){
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < n){
                inv_std[i] = T(1) / sqrt(running_var[i] + eps);
                dst_mean[i] = running_mean[i];
            }
        }
        template<typename T>
        __global__
        void bn_eval_backward(
            const T* d_norm_out,
            const T* pre_act,
            const T* mean,
            const T* inv_std,
            const T* gamma,
            T* d_conv_out,
            T* d_gamma,
            T* d_beta,
            unsigned int spatial,
            unsigned int OC
        ){
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int total = spatial * OC;
            if(idx < total){
                unsigned int c = idx % OC;
                T z_hat = (pre_act[idx] - mean[c]) * inv_std[c];
                T dno = d_norm_out[idx];
                d_conv_out[idx] = dno * gamma[c] * inv_std[c];
                atomicAdd(&d_gamma[c], dno * z_hat);
                atomicAdd(&d_beta[c], dno);
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TI = typename devices::CUDA<DEV_SPEC>::index_t;
        constexpr TI N  = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT, IW = LAYER_SPEC::INPUT_WIDTH, IC = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT, OW = LAYER_SPEC::OUTPUT_WIDTH, OC = LAYER_SPEC::OUTPUT_CHANNELS;
        constexpr TI KH = LAYER_SPEC::KERNEL_HEIGHT, KW = LAYER_SPEC::KERNEL_WIDTH;
        constexpr TI SH = LAYER_SPEC::STRIDE_H, SW = LAYER_SPEC::STRIDE_W;
        constexpr TI PH = LAYER_SPEC::PADDING_H, PW = LAYER_SPEC::PADDING_W;
        constexpr cudnnDataType_t dt = nn::cuda::get_cudnn_dtype<T>();
        constexpr bool HAS_BN = LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM;
        constexpr bool HAS_RELU = LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU;
        constexpr bool FUSE_RELU = HAS_RELU && !HAS_BN;

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bd = nullptr, bnd = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t fused_ad = nullptr, relu_ad = nullptr;
        static cudnnConvolutionFwdAlgo_t cached_algo;
        static size_t cached_ws = 0;
        static bool initialized = false;
        if(!initialized){
            cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
            cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
            cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW);
            cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt); cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
            cudnnCreateTensorDescriptor(&bd); cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1);
            cudnnCreateActivationDescriptor(&fused_ad);
            if constexpr(FUSE_RELU){
                cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
                cached_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else {
                cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0);
                int ac; cudnnConvolutionFwdAlgoPerf_t ap;
                cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, xd, wd, cd, yd, 1, &ac, &ap);
                cached_algo = ap.algo;
            }
            cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, xd, wd, cd, yd, cached_algo, &cached_ws);
            if constexpr(HAS_BN){
                cudnnCreateTensorDescriptor(&bnd); cudnnDeriveBNTensorDescriptor(bnd, yd, CUDNN_BATCHNORM_SPATIAL);
                if constexpr(HAS_RELU){
                    cudnnCreateActivationDescriptor(&relu_ad); cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
                }
            }
            initialized = true;
        }
        if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
        { T a1 = 1, a2 = 0;
          cudnnConvolutionBiasActivationForward(device.cudnn_handle, &a1, xd, input._data, wd, layer.weights.parameters._data,
              cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size,
              &a2, yd, output._data, bd, layer.biases.parameters._data, fused_ad, yd, output._data); }
        if constexpr(HAS_BN){
            T a = 1, b = 0;
            cudnnBatchNormalizationForwardInference(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                yd, output._data, yd, output._data, bnd,
                layer.norm.gamma.parameters._data, layer.norm.beta.parameters._data,
                layer.norm.running_mean.parameters._data, layer.norm.running_var.parameters._data, (double)LAYER_SPEC::NORM_EPSILON);
            if constexpr(HAS_RELU){
                T ra = 1, rb = 0; cudnnActivationForward(device.cudnn_handle, relu_ad, &ra, yd, output._data, &rb, yd, output._data);
            }
        }
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
        constexpr bool HAS_BN = LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM;
        constexpr bool HAS_RELU = LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU;
        constexpr bool FUSE_RELU = HAS_RELU && !HAS_BN;

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bd = nullptr, bnd = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t fused_ad = nullptr, relu_ad = nullptr;
        static cudnnConvolutionFwdAlgo_t cached_algo;
        static size_t cached_ws = 0;
        static bool initialized = false;
        if(!initialized){
            cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
            cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
            cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW);
            cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt); cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
            cudnnCreateTensorDescriptor(&bd); cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1);
            cudnnCreateActivationDescriptor(&fused_ad);
            if constexpr(FUSE_RELU){
                cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
                cached_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else {
                cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0);
                int ac; cudnnConvolutionFwdAlgoPerf_t ap;
                cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, xd, wd, cd, yd, 1, &ac, &ap);
                cached_algo = ap.algo;
            }
            cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, xd, wd, cd, yd, cached_algo, &cached_ws);
            if constexpr(HAS_BN){
                cudnnCreateTensorDescriptor(&bnd); cudnnDeriveBNTensorDescriptor(bnd, yd, CUDNN_BATCHNORM_SPATIAL);
            }
            if constexpr(HAS_BN && HAS_RELU){
                cudnnCreateActivationDescriptor(&relu_ad); cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
            }
            initialized = true;
        }
        if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
        if constexpr(HAS_BN){
            { T a1 = 1, a2 = 0;
              cudnnConvolutionBiasActivationForward(device.cudnn_handle, &a1, xd, input._data, wd, layer.weights.parameters._data,
                  cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size,
                  &a2, yd, layer.pre_activations._data, bd, layer.biases.parameters._data, fused_ad, yd, layer.pre_activations._data); }
            if constexpr(mode::is<MODE, mode::Evaluation>){
                T a = 1, b = 0;
                cudnnBatchNormalizationForwardInference(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                    yd, layer.pre_activations._data, yd, output._data, bnd,
                    layer.norm.gamma.parameters._data, layer.norm.beta.parameters._data,
                    layer.norm.running_mean.parameters._data, layer.norm.running_var.parameters._data, (double)LAYER_SPEC::NORM_EPSILON);
                constexpr TI BLOCKSIZE = 256;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(OC, BLOCKSIZE);
                devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
                nn::layers::conv2d::cuda::kernels::compute_inv_std_from_running_var<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(
                    layer.norm.running_var.parameters._data, layer.norm_cache.inv_std._data,
                    layer.norm_cache.mean._data, layer.norm.running_mean.parameters._data,
                    (T)LAYER_SPEC::NORM_EPSILON, OC);
            } else {
                T a = 1, b = 0;
                cudnnBatchNormalizationForwardTraining(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                    yd, layer.pre_activations._data, yd, output._data, bnd,
                    layer.norm.gamma.parameters._data, layer.norm.beta.parameters._data,
                    (double)LAYER_SPEC::BN_MOMENTUM,
                    layer.norm.running_mean.parameters._data, layer.norm.running_var.parameters._data,
                    (double)LAYER_SPEC::NORM_EPSILON, layer.norm_cache.mean._data, layer.norm_cache.inv_std._data);
            }
            if constexpr(HAS_RELU){
                T a = 1, b = 0; cudnnActivationForward(device.cudnn_handle, relu_ad, &a, yd, output._data, &b, yd, output._data);
            }
        } else {
            T* dst = FUSE_RELU ? output._data : layer.pre_activations._data;
            { T a1 = 1, a2 = 0;
              cudnnConvolutionBiasActivationForward(device.cudnn_handle, &a1, xd, input._data, wd, layer.weights.parameters._data,
                  cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size,
                  &a2, yd, dst, bd, layer.biases.parameters._data, fused_ad, yd, dst); }
            if constexpr(!FUSE_RELU){
                cudaMemcpyAsync(output._data, layer.pre_activations._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream);
            }
        }
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

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bias_d = nullptr, bnd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t relu_ad = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionBwdFilterAlgo_t cached_bf_algo;
        static cudnnConvolutionBwdDataAlgo_t cached_bd_algo;
        static size_t cached_bf_ws = 0, cached_bd_ws = 0;
        static bool bf_ok = false, bd_ok = false;
        static bool initialized = false;
        if(!initialized){
            cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
            cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
            cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt); cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
            cudnnCreateTensorDescriptor(&bias_d); cudnnSetTensor4dDescriptor(bias_d, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1);
            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
                cudnnCreateActivationDescriptor(&relu_ad); cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
            }
            if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
                cudnnCreateTensorDescriptor(&bnd); cudnnDeriveBNTensorDescriptor(bnd, yd, CUDNN_BATCHNORM_SPATIAL);
            }
            cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW);
            { constexpr int MA = 8; int ac; cudnnConvolutionBwdFilterAlgoPerf_t ap[MA];
              cudnnGetConvolutionBackwardFilterAlgorithm_v7(device.cudnn_handle, xd, yd, cd, wd, MA, &ac, ap);
              for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_bf_algo = ap[i].algo; bf_ok = true; break; } }
              if(bf_ok) cudnnGetConvolutionBackwardFilterWorkspaceSize(device.cudnn_handle, xd, yd, cd, wd, cached_bf_algo, &cached_bf_ws); }
            { constexpr int MA = 8; int ac; cudnnConvolutionBwdDataAlgoPerf_t ap[MA];
              cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, MA, &ac, ap);
              for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_bd_algo = ap[i].algo; bd_ok = true; break; } }
              if(bd_ok) cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, cached_bd_algo, &cached_bd_ws); }
            initialized = true;
        }

        T* d_conv_out = layer.output._data;
        if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
            T a = 1, b = 0;
            cudnnActivationBackward(device.cudnn_handle, relu_ad, &a, yd, layer.output._data, yd, d_output._data, yd, layer.output._data, &b, yd, d_output._data);
        }
        if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            constexpr bool IS_EVAL = mode::is<MODE, mode::Evaluation>;
            if constexpr(IS_EVAL){
                constexpr TI SPATIAL = N * OH * OW;
                constexpr TI TOTAL = SPATIAL * OC;
                constexpr TI BN_BS = 256;
                constexpr TI BN_NB = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BN_BS);
                nn::layers::conv2d::cuda::kernels::bn_eval_backward<<<BN_NB, BN_BS, 0, device.stream>>>(
                    d_output._data, layer.pre_activations._data,
                    layer.norm_cache.mean._data, layer.norm_cache.inv_std._data,
                    layer.norm.gamma.parameters._data,
                    d_conv_out,
                    layer.norm.gamma.gradient._data, layer.norm.beta.gradient._data,
                    SPATIAL, OC);
            } else {
                T ad1 = 1, bd1 = 0, ap1 = 1, bp1 = 1;
                cudnnBatchNormalizationBackward(device.cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &ad1, &bd1, &ap1, &bp1,
                    yd, layer.pre_activations._data, yd, d_output._data, yd, d_conv_out,
                    bnd, layer.norm.gamma.parameters._data, layer.norm.gamma.gradient._data, layer.norm.beta.gradient._data,
                    (double)LAYER_SPEC::NORM_EPSILON, layer.norm_cache.mean._data, layer.norm_cache.inv_std._data);
            }
        } else {
            cudaMemcpyAsync(d_conv_out, d_output._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream);
        }
        { T a = 1, b = 1;
          cudnnConvolutionBackwardBias(device.cudnn_handle, &a, yd, d_conv_out, &b, bias_d, layer.biases.gradient._data); }

        if(bf_ok){
            if(cached_bf_ws > 0) ensure_cudnn_workspace(device, cached_bf_ws);
            T a = 1, b = 1;
            cudnnConvolutionBackwardFilter(device.cudnn_handle, &a, xd, input._data, yd, d_conv_out,
                cd, cached_bf_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, wd, layer.weights.gradient._data);
        }
        if(bd_ok){
            if(cached_bd_ws > 0) ensure_cudnn_workspace(device, cached_bd_ws);
            T a = 1, b = 0;
            cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_conv_out,
                cd, cached_bd_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data);
        }
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

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionBwdDataAlgo_t cached_algo;
        static size_t cached_ws = 0;
        static bool algo_ok = false;
        static bool initialized = false;
        if(!initialized){
            cudnnCreateTensorDescriptor(&xd); cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW);
            cudnnCreateTensorDescriptor(&yd); cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW);
            cudnnCreateConvolutionDescriptor(&cd); cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt); cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
            cudnnCreateFilterDescriptor(&wd); cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW);
            constexpr int MA = 8; int ac; cudnnConvolutionBwdDataAlgoPerf_t ap[MA];
            cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, MA, &ac, ap);
            for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_algo = ap[i].algo; algo_ok = true; break; } }
            if(algo_ok) cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, cached_algo, &cached_ws);
            initialized = true;
        }
        if(algo_ok){
            if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
            T a = 1, b = 0;
            cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_output._data,
                cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data);
        }
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
