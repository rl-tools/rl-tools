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
        template<typename T, typename TI>
        __global__
        void bn_stats_training(
            const T* pre_act,
            T* mean,
            T* inv_std,
            T* running_mean,
            T* running_var,
            T momentum,
            T eps,
            TI spatial,
            TI OC
        ){
            TI c = (TI)blockIdx.x;
            if(c >= OC){
                return;
            }
            __shared__ T s_sum[256];
            __shared__ T s_sq_sum[256];
            TI tid = (TI)threadIdx.x;
            T local_sum = 0;
            T local_sq_sum = 0;
            for(TI i = tid; i < spatial; i += (TI)blockDim.x){
                TI idx = i * OC + c;
                T v = pre_act[idx];
                local_sum += v;
                local_sq_sum += v * v;
            }
            s_sum[tid] = local_sum;
            s_sq_sum[tid] = local_sq_sum;
            __syncthreads();
            for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
                if(threadIdx.x < s){
                    s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
                    s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + s];
                }
                __syncthreads();
            }
            if(threadIdx.x == 0){
                T m = s_sum[0] / (T)spatial;
                T sq_m = s_sq_sum[0] / (T)spatial;
                T var = sq_m - m * m;
                if(var < (T)0){
                    var = (T)0;
                }
                mean[c] = m;
                inv_std[c] = (T)1 / sqrt(var + eps);
                T rm = running_mean[c];
                T rv = running_var[c];
                running_mean[c] = ((T)1 - momentum) * rm + momentum * m;
                running_var[c] = ((T)1 - momentum) * rv + momentum * var;
            }
        }
        template<typename T, typename TI>
        __global__
        void bn_stats_eval(
            const T* running_mean,
            const T* running_var,
            T* mean,
            T* inv_std,
            T eps,
            TI OC
        ){
            TI i = (TI)blockIdx.x * (TI)blockDim.x + (TI)threadIdx.x;
            if(i < OC){
                mean[i] = running_mean[i];
                inv_std[i] = (T)1 / sqrt(running_var[i] + eps);
            }
        }
        template<typename T, typename TI>
        __global__
        void bn_forward_eval_running(
            const T* pre_act,
            const T* running_mean,
            const T* running_var,
            const T* gamma,
            const T* beta,
            T* output,
            T eps,
            TI total,
            TI OC
        ){
            TI idx = (TI)blockIdx.x * (TI)blockDim.x + (TI)threadIdx.x;
            if(idx < total){
                TI c = idx % OC;
                T inv_std = (T)1 / sqrt(running_var[c] + eps);
                T z_hat = (pre_act[idx] - running_mean[c]) * inv_std;
                output[idx] = gamma[c] * z_hat + beta[c];
            }
        }
        template<typename T, typename TI>
        __global__
        void bn_forward_affine(
            const T* pre_act,
            const T* mean,
            const T* inv_std,
            const T* gamma,
            const T* beta,
            T* output,
            TI total,
            TI OC
        ){
            TI idx = (TI)blockIdx.x * (TI)blockDim.x + (TI)threadIdx.x;
            if(idx < total){
                TI c = idx % OC;
                T z_hat = (pre_act[idx] - mean[c]) * inv_std[c];
                output[idx] = gamma[c] * z_hat + beta[c];
            }
        }
        template<typename T, typename TI>
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
            TI spatial,
            TI OC
        ){
            TI c = (TI)blockIdx.x;
            if(c >= OC){
                return;
            }
            __shared__ T s_d_gamma[256];
            __shared__ T s_d_beta[256];
            TI tid = (TI)threadIdx.x;
            T local_d_gamma = 0;
            T local_d_beta = 0;
            for(TI i = tid; i < spatial; i += (TI)blockDim.x){
                TI idx = i * OC + c;
                T z_hat = (pre_act[idx] - mean[c]) * inv_std[c];
                T dno = d_norm_out[idx];
                d_conv_out[idx] = dno * gamma[c] * inv_std[c];
                local_d_gamma += dno * z_hat;
                local_d_beta += dno;
            }
            s_d_gamma[tid] = local_d_gamma;
            s_d_beta[tid] = local_d_beta;
            __syncthreads();
            for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
                if(threadIdx.x < s){
                    s_d_gamma[threadIdx.x] += s_d_gamma[threadIdx.x + s];
                    s_d_beta[threadIdx.x] += s_d_beta[threadIdx.x + s];
                }
                __syncthreads();
            }
            if(threadIdx.x == 0){
                atomicAdd(&d_gamma[c], s_d_gamma[0]);
                atomicAdd(&d_beta[c], s_d_beta[0]);
            }
        }
        template<typename T>
        __global__
        void bn_training_backward(
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
            unsigned int c = blockIdx.x;
            if(c >= OC){
                return;
            }
            __shared__ T s_sum_dz_hat[256];
            __shared__ T s_sum_dz_hat_z_hat[256];
            __shared__ T s_d_gamma[256];
            __shared__ T s_d_beta[256];
            unsigned int tid = threadIdx.x;
            T local_sum_dz_hat = 0;
            T local_sum_dz_hat_z_hat = 0;
            T local_d_gamma = 0;
            T local_d_beta = 0;
            T gamma_c = gamma[c];
            T mean_c = mean[c];
            T inv_std_c = inv_std[c];
            for(unsigned int i = tid; i < spatial; i += blockDim.x){
                unsigned int idx = i * OC + c;
                T z_hat = (pre_act[idx] - mean_c) * inv_std_c;
                T dno = d_norm_out[idx];
                T d_z_hat = dno * gamma_c;
                local_sum_dz_hat += d_z_hat;
                local_sum_dz_hat_z_hat += d_z_hat * z_hat;
                local_d_gamma += dno * z_hat;
                local_d_beta += dno;
            }
            s_sum_dz_hat[tid] = local_sum_dz_hat;
            s_sum_dz_hat_z_hat[tid] = local_sum_dz_hat_z_hat;
            s_d_gamma[tid] = local_d_gamma;
            s_d_beta[tid] = local_d_beta;
            __syncthreads();
            for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
                if(tid < s){
                    s_sum_dz_hat[tid] += s_sum_dz_hat[tid + s];
                    s_sum_dz_hat_z_hat[tid] += s_sum_dz_hat_z_hat[tid + s];
                    s_d_gamma[tid] += s_d_gamma[tid + s];
                    s_d_beta[tid] += s_d_beta[tid + s];
                }
                __syncthreads();
            }
            T sum_dz_hat = s_sum_dz_hat[0];
            T sum_dz_hat_z_hat = s_sum_dz_hat_z_hat[0];
            T inv_n = (T)1 / (T)spatial;
            for(unsigned int i = tid; i < spatial; i += blockDim.x){
                unsigned int idx = i * OC + c;
                T z_hat = (pre_act[idx] - mean_c) * inv_std_c;
                T d_z_hat = d_norm_out[idx] * gamma_c;
                d_conv_out[idx] = inv_std_c * inv_n * ((T)spatial * d_z_hat - sum_dz_hat - z_hat * sum_dz_hat_z_hat);
            }
            if(tid == 0){
                atomicAdd(&d_gamma[c], s_d_gamma[0]);
                atomicAdd(&d_beta[c], s_d_beta[0]);
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

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bd = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t fused_ad = nullptr, relu_ad = nullptr;
        static cudnnConvolutionFwdAlgo_t cached_algo;
        static size_t cached_ws = 0;
        static bool initialized = false;
        if(!initialized){
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor conv2d_eval.xd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW), "cudnnSetTensor4dDescriptor conv2d_eval.xd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor conv2d_eval.yd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW), "cudnnSetTensor4dDescriptor conv2d_eval.yd");
            check_cudnn_call(device, cudnnCreateFilterDescriptor(&wd), "cudnnCreateFilterDescriptor conv2d_eval.wd");
            check_cudnn_call(device, cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW), "cudnnSetFilter4dDescriptor conv2d_eval.wd");
            check_cudnn_call(device, cudnnCreateConvolutionDescriptor(&cd), "cudnnCreateConvolutionDescriptor conv2d_eval.cd");
            check_cudnn_call(device, cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt), "cudnnSetConvolution2dDescriptor conv2d_eval.cd");
            check_cudnn_call(device, cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION), "cudnnSetConvolutionMathType conv2d_eval.cd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&bd), "cudnnCreateTensorDescriptor conv2d_eval.bd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1), "cudnnSetTensor4dDescriptor conv2d_eval.bd");
            check_cudnn_call(device, cudnnCreateActivationDescriptor(&fused_ad), "cudnnCreateActivationDescriptor conv2d_eval.fused_ad");
            if constexpr(FUSE_RELU){
                check_cudnn_call(device, cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_eval.fused_ad");
                cached_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else {
                check_cudnn_call(device, cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_eval.fused_ad");
                int ac; cudnnConvolutionFwdAlgoPerf_t ap;
                check_cudnn_call(device, cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, xd, wd, cd, yd, 1, &ac, &ap), "cudnnGetConvolutionForwardAlgorithm_v7 conv2d_eval");
                cached_algo = ap.algo;
            }
            check_cudnn_call(device, cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, xd, wd, cd, yd, cached_algo, &cached_ws), "cudnnGetConvolutionForwardWorkspaceSize conv2d_eval");
            if constexpr(HAS_BN && HAS_RELU){
                check_cudnn_call(device, cudnnCreateActivationDescriptor(&relu_ad), "cudnnCreateActivationDescriptor conv2d_eval.relu_ad");
                check_cudnn_call(device, cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_eval.relu_ad");
            }
            initialized = true;
        }
        if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
        if constexpr(HAS_BN){
            // Keep BN paths on explicit conv+bias to avoid fragile fused-call behavior.
            T a = 1, b = 0;
            cudnnStatus_t stat = cudnnConvolutionForward(device.cudnn_handle, &a, xd, input._data, wd, layer.weights.parameters._data,
                cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, yd, output._data);
            check_cudnn_call(device, stat, "cudnnConvolutionForward conv2d_eval");
            T ba = 1, bb = 1;
            stat = cudnnAddTensor(device.cudnn_handle, &ba, bd, layer.biases.parameters._data, &bb, yd, output._data);
            check_cudnn_call(device, stat, "cudnnAddTensor conv2d_eval");
        }
        else{
            T a1 = 1, a2 = 0;
            check_cudnn_call(device, cudnnConvolutionBiasActivationForward(device.cudnn_handle, &a1, xd, input._data, wd, layer.weights.parameters._data,
                cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size,
                &a2, yd, output._data, bd, layer.biases.parameters._data, fused_ad, yd, output._data), "cudnnConvolutionBiasActivationForward conv2d_eval");
        }
        if constexpr(HAS_BN){
            constexpr TI BN_ELEMENT_BLOCK = 256;
            constexpr TI TOTAL = N * OH * OW * OC;
            constexpr TI N_BLOCKS_ELEMENT = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BN_ELEMENT_BLOCK);
            nn::layers::conv2d::cuda::kernels::bn_forward_eval_running<T, TI><<<N_BLOCKS_ELEMENT, BN_ELEMENT_BLOCK, 0, device.stream>>>(
                output._data,
                layer.norm.running_mean.parameters._data,
                layer.norm.running_var.parameters._data,
                layer.norm.gamma.parameters._data,
                layer.norm.beta.parameters._data,
                output._data,
                (T)LAYER_SPEC::NORM_EPSILON,
                TOTAL,
                OC
            );
            if constexpr(HAS_RELU){
                T ra = 1, rb = 0;
                check_cudnn_call(device, cudnnActivationForward(device.cudnn_handle, relu_ad, &ra, yd, output._data, &rb, yd, output._data), "cudnnActivationForward conv2d_eval");
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

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bd = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t fused_ad = nullptr, relu_ad = nullptr;
        static cudnnConvolutionFwdAlgo_t cached_algo;
        static size_t cached_ws = 0;
        static bool initialized = false;
        if(!initialized){
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor conv2d_fwd.xd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW), "cudnnSetTensor4dDescriptor conv2d_fwd.xd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor conv2d_fwd.yd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW), "cudnnSetTensor4dDescriptor conv2d_fwd.yd");
            check_cudnn_call(device, cudnnCreateFilterDescriptor(&wd), "cudnnCreateFilterDescriptor conv2d_fwd.wd");
            check_cudnn_call(device, cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW), "cudnnSetFilter4dDescriptor conv2d_fwd.wd");
            check_cudnn_call(device, cudnnCreateConvolutionDescriptor(&cd), "cudnnCreateConvolutionDescriptor conv2d_fwd.cd");
            check_cudnn_call(device, cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt), "cudnnSetConvolution2dDescriptor conv2d_fwd.cd");
            check_cudnn_call(device, cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION), "cudnnSetConvolutionMathType conv2d_fwd.cd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&bd), "cudnnCreateTensorDescriptor conv2d_fwd.bd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1), "cudnnSetTensor4dDescriptor conv2d_fwd.bd");
            check_cudnn_call(device, cudnnCreateActivationDescriptor(&fused_ad), "cudnnCreateActivationDescriptor conv2d_fwd.fused_ad");
            if constexpr(FUSE_RELU){
                check_cudnn_call(device, cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_fwd.fused_ad");
                cached_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else {
                check_cudnn_call(device, cudnnSetActivationDescriptor(fused_ad, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_fwd.fused_ad");
                int ac; cudnnConvolutionFwdAlgoPerf_t ap;
                check_cudnn_call(device, cudnnGetConvolutionForwardAlgorithm_v7(device.cudnn_handle, xd, wd, cd, yd, 1, &ac, &ap), "cudnnGetConvolutionForwardAlgorithm_v7 conv2d_fwd");
                cached_algo = ap.algo;
            }
            check_cudnn_call(device, cudnnGetConvolutionForwardWorkspaceSize(device.cudnn_handle, xd, wd, cd, yd, cached_algo, &cached_ws), "cudnnGetConvolutionForwardWorkspaceSize conv2d_fwd");
            if constexpr(HAS_BN && HAS_RELU){
                check_cudnn_call(device, cudnnCreateActivationDescriptor(&relu_ad), "cudnnCreateActivationDescriptor conv2d_fwd.relu_ad");
                check_cudnn_call(device, cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_fwd.relu_ad");
            }
            initialized = true;
        }
        if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
        if constexpr(HAS_BN){
            constexpr TI BN_CHANNEL_BLOCK = 256;
            constexpr TI BN_ELEMENT_BLOCK = 256;
            constexpr TI SPATIAL = N * OH * OW;
            constexpr TI TOTAL = SPATIAL * OC;
            { T a = 1, b = 0;
              cudnnStatus_t stat = cudnnConvolutionForward(device.cudnn_handle, &a, xd, input._data, wd, layer.weights.parameters._data,
                  cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, yd, layer.pre_activations._data);
              check_cudnn_call(device, stat, "cudnnConvolutionForward conv2d_fwd");
              T ba = 1, bb = 1;
              stat = cudnnAddTensor(device.cudnn_handle, &ba, bd, layer.biases.parameters._data, &bb, yd, layer.pre_activations._data);
              check_cudnn_call(device, stat, "cudnnAddTensor conv2d_fwd");
            }
            if constexpr(mode::is<MODE, mode::Evaluation>){
                constexpr TI N_BLOCKS_CHANNEL = RL_TOOLS_DEVICES_CUDA_CEIL(OC, BN_CHANNEL_BLOCK);
                constexpr TI N_BLOCKS_ELEMENT = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BN_ELEMENT_BLOCK);
                nn::layers::conv2d::cuda::kernels::bn_stats_eval<T, TI><<<N_BLOCKS_CHANNEL, BN_CHANNEL_BLOCK, 0, device.stream>>>(
                    layer.norm.running_mean.parameters._data,
                    layer.norm.running_var.parameters._data,
                    layer.norm_cache.mean._data,
                    layer.norm_cache.inv_std._data,
                    (T)LAYER_SPEC::NORM_EPSILON,
                    OC
                );
                nn::layers::conv2d::cuda::kernels::bn_forward_affine<T, TI><<<N_BLOCKS_ELEMENT, BN_ELEMENT_BLOCK, 0, device.stream>>>(
                    layer.pre_activations._data,
                    layer.norm_cache.mean._data,
                    layer.norm_cache.inv_std._data,
                    layer.norm.gamma.parameters._data,
                    layer.norm.beta.parameters._data,
                    output._data,
                    TOTAL,
                    OC
                );
            } else {
                nn::layers::conv2d::cuda::kernels::bn_stats_training<T, TI><<<OC, BN_CHANNEL_BLOCK, 0, device.stream>>>(
                    layer.pre_activations._data,
                    layer.norm_cache.mean._data,
                    layer.norm_cache.inv_std._data,
                    layer.norm.running_mean.parameters._data,
                    layer.norm.running_var.parameters._data,
                    (T)LAYER_SPEC::BN_MOMENTUM,
                    (T)LAYER_SPEC::NORM_EPSILON,
                    SPATIAL,
                    OC
                );
                constexpr TI N_BLOCKS_ELEMENT = RL_TOOLS_DEVICES_CUDA_CEIL(TOTAL, BN_ELEMENT_BLOCK);
                nn::layers::conv2d::cuda::kernels::bn_forward_affine<T, TI><<<N_BLOCKS_ELEMENT, BN_ELEMENT_BLOCK, 0, device.stream>>>(
                    layer.pre_activations._data,
                    layer.norm_cache.mean._data,
                    layer.norm_cache.inv_std._data,
                    layer.norm.gamma.parameters._data,
                    layer.norm.beta.parameters._data,
                    output._data,
                    TOTAL,
                    OC
                );
            }
            if constexpr(HAS_RELU){
                T a = 1, b = 0;
                check_cudnn_call(device, cudnnActivationForward(device.cudnn_handle, relu_ad, &a, yd, output._data, &b, yd, output._data), "cudnnActivationForward conv2d_fwd");
            }
        } else {
            T* dst = FUSE_RELU ? output._data : layer.pre_activations._data;
            { T a1 = 1, a2 = 0;
              check_cudnn_call(device, cudnnConvolutionBiasActivationForward(device.cudnn_handle, &a1, xd, input._data, wd, layer.weights.parameters._data,
                  cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size,
                  &a2, yd, dst, bd, layer.biases.parameters._data, fused_ad, yd, dst), "cudnnConvolutionBiasActivationForward conv2d_fwd"); }
            if constexpr(!FUSE_RELU){
                check_cuda_call(device, cudaMemcpyAsync(output._data, layer.pre_activations._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream), "cudaMemcpyAsync conv2d_fwd preact_to_output");
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

        static cudnnTensorDescriptor_t xd = nullptr, yd = nullptr, bias_d = nullptr;
        static cudnnConvolutionDescriptor_t cd = nullptr;
        static cudnnActivationDescriptor_t relu_ad = nullptr;
        static cudnnFilterDescriptor_t wd = nullptr;
        static cudnnConvolutionBwdFilterAlgo_t cached_bf_algo;
        static cudnnConvolutionBwdDataAlgo_t cached_bd_algo;
        static size_t cached_bf_ws = 0, cached_bd_ws = 0;
        static bool bf_ok = false, bd_ok = false;
        static bool initialized = false;
        if(!initialized){
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor conv2d_bwd.xd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW), "cudnnSetTensor4dDescriptor conv2d_bwd.xd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor conv2d_bwd.yd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW), "cudnnSetTensor4dDescriptor conv2d_bwd.yd");
            check_cudnn_call(device, cudnnCreateConvolutionDescriptor(&cd), "cudnnCreateConvolutionDescriptor conv2d_bwd.cd");
            check_cudnn_call(device, cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt), "cudnnSetConvolution2dDescriptor conv2d_bwd.cd");
            check_cudnn_call(device, cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION), "cudnnSetConvolutionMathType conv2d_bwd.cd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&bias_d), "cudnnCreateTensorDescriptor conv2d_bwd.bias_d");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(bias_d, CUDNN_TENSOR_NHWC, dt, 1, OC, 1, 1), "cudnnSetTensor4dDescriptor conv2d_bwd.bias_d");
            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
                check_cudnn_call(device, cudnnCreateActivationDescriptor(&relu_ad), "cudnnCreateActivationDescriptor conv2d_bwd.relu_ad");
                check_cudnn_call(device, cudnnSetActivationDescriptor(relu_ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor conv2d_bwd.relu_ad");
            }
            check_cudnn_call(device, cudnnCreateFilterDescriptor(&wd), "cudnnCreateFilterDescriptor conv2d_bwd.wd");
            check_cudnn_call(device, cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW), "cudnnSetFilter4dDescriptor conv2d_bwd.wd");
            { constexpr int MA = 8; int ac; cudnnConvolutionBwdFilterAlgoPerf_t ap[MA];
              check_cudnn_call(device, cudnnGetConvolutionBackwardFilterAlgorithm_v7(device.cudnn_handle, xd, yd, cd, wd, MA, &ac, ap), "cudnnGetConvolutionBackwardFilterAlgorithm_v7 conv2d_bwd");
              for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_bf_algo = ap[i].algo; bf_ok = true; break; } }
              if(bf_ok) check_cudnn_call(device, cudnnGetConvolutionBackwardFilterWorkspaceSize(device.cudnn_handle, xd, yd, cd, wd, cached_bf_algo, &cached_bf_ws), "cudnnGetConvolutionBackwardFilterWorkspaceSize conv2d_bwd"); }
            { constexpr int MA = 8; int ac; cudnnConvolutionBwdDataAlgoPerf_t ap[MA];
              check_cudnn_call(device, cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, MA, &ac, ap), "cudnnGetConvolutionBackwardDataAlgorithm_v7 conv2d_bwd");
              for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_bd_algo = ap[i].algo; bd_ok = true; break; } }
              if(bd_ok) check_cudnn_call(device, cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, cached_bd_algo, &cached_bd_ws), "cudnnGetConvolutionBackwardDataWorkspaceSize conv2d_bwd"); }
            initialized = true;
        }

        T* d_conv_out = layer.output._data;
        if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::RELU){
            T a = 1, b = 0;
            check_cudnn_call(device, cudnnActivationBackward(device.cudnn_handle, relu_ad, &a, yd, layer.output._data, yd, d_output._data, yd, layer.output._data, &b, yd, d_output._data), "cudnnActivationBackward conv2d_bwd");
        }
        if constexpr(LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            constexpr bool IS_EVAL = mode::is<MODE, mode::Evaluation>;
            if constexpr(IS_EVAL){
                constexpr TI SPATIAL = N * OH * OW;
                constexpr TI BN_BS = 256;
                nn::layers::conv2d::cuda::kernels::bn_eval_backward<T, TI><<<OC, BN_BS, 0, device.stream>>>(
                    d_output._data, layer.pre_activations._data,
                    layer.norm_cache.mean._data, layer.norm_cache.inv_std._data,
                    layer.norm.gamma.parameters._data,
                    d_conv_out,
                    layer.norm.gamma.gradient._data, layer.norm.beta.gradient._data,
                    SPATIAL, OC);
            } else {
                constexpr TI SPATIAL = N * OH * OW;
                constexpr TI BN_BS = 256;
                nn::layers::conv2d::cuda::kernels::bn_training_backward<<<OC, BN_BS, 0, device.stream>>>(
                    d_output._data, layer.pre_activations._data,
                    layer.norm_cache.mean._data, layer.norm_cache.inv_std._data,
                    layer.norm.gamma.parameters._data,
                    d_conv_out,
                    layer.norm.gamma.gradient._data, layer.norm.beta.gradient._data,
                    SPATIAL, OC);
            }
        } else {
            check_cuda_call(device, cudaMemcpyAsync(d_conv_out, d_output._data, N*OH*OW*OC*sizeof(T), cudaMemcpyDeviceToDevice, device.stream), "cudaMemcpyAsync conv2d_bwd d_output_to_d_conv_out");
        }
        { T a = 1, b = 1;
          check_cudnn_call(device, cudnnConvolutionBackwardBias(device.cudnn_handle, &a, yd, d_conv_out, &b, bias_d, layer.biases.gradient._data), "cudnnConvolutionBackwardBias conv2d_bwd"); }

        if(bf_ok){
            if(cached_bf_ws > 0) ensure_cudnn_workspace(device, cached_bf_ws);
            T a = 1, b = 1;
            check_cudnn_call(device, cudnnConvolutionBackwardFilter(device.cudnn_handle, &a, xd, input._data, yd, d_conv_out,
                cd, cached_bf_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, wd, layer.weights.gradient._data), "cudnnConvolutionBackwardFilter conv2d_bwd");
        }
        if(bd_ok){
            if(cached_bd_ws > 0) ensure_cudnn_workspace(device, cached_bd_ws);
            T a = 1, b = 0;
            check_cudnn_call(device, cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_conv_out,
                cd, cached_bd_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data), "cudnnConvolutionBackwardData conv2d_bwd");
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
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&xd), "cudnnCreateTensorDescriptor conv2d_bwd_input.xd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, dt, N, IC, IH, IW), "cudnnSetTensor4dDescriptor conv2d_bwd_input.xd");
            check_cudnn_call(device, cudnnCreateTensorDescriptor(&yd), "cudnnCreateTensorDescriptor conv2d_bwd_input.yd");
            check_cudnn_call(device, cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, dt, N, OC, OH, OW), "cudnnSetTensor4dDescriptor conv2d_bwd_input.yd");
            check_cudnn_call(device, cudnnCreateConvolutionDescriptor(&cd), "cudnnCreateConvolutionDescriptor conv2d_bwd_input.cd");
            check_cudnn_call(device, cudnnSetConvolution2dDescriptor(cd, PH, PW, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, dt), "cudnnSetConvolution2dDescriptor conv2d_bwd_input.cd");
            check_cudnn_call(device, cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION), "cudnnSetConvolutionMathType conv2d_bwd_input.cd");
            check_cudnn_call(device, cudnnCreateFilterDescriptor(&wd), "cudnnCreateFilterDescriptor conv2d_bwd_input.wd");
            check_cudnn_call(device, cudnnSetFilter4dDescriptor(wd, dt, CUDNN_TENSOR_NHWC, OC, IC, KH, KW), "cudnnSetFilter4dDescriptor conv2d_bwd_input.wd");
            constexpr int MA = 8; int ac; cudnnConvolutionBwdDataAlgoPerf_t ap[MA];
            check_cudnn_call(device, cudnnGetConvolutionBackwardDataAlgorithm_v7(device.cudnn_handle, wd, yd, cd, xd, MA, &ac, ap), "cudnnGetConvolutionBackwardDataAlgorithm_v7 conv2d_bwd_input");
            for(int i = 0; i < ac; i++){ if(ap[i].status == CUDNN_STATUS_SUCCESS){ cached_algo = ap[i].algo; algo_ok = true; break; } }
            if(algo_ok) check_cudnn_call(device, cudnnGetConvolutionBackwardDataWorkspaceSize(device.cudnn_handle, wd, yd, cd, xd, cached_algo, &cached_ws), "cudnnGetConvolutionBackwardDataWorkspaceSize conv2d_bwd_input");
            initialized = true;
        }
        if(algo_ok){
            if(cached_ws > 0) ensure_cudnn_workspace(device, cached_ws);
            T a = 1, b = 0;
            check_cudnn_call(device, cudnnConvolutionBackwardData(device.cudnn_handle, &a, wd, layer.weights.parameters._data, yd, d_output._data,
                cd, cached_algo, device.cudnn_workspace, device.cudnn_workspace_size, &b, xd, d_input._data), "cudnnConvolutionBackwardData conv2d_bwd_input");
        }
        check_status(device);
    }

    template<typename DEV_SPEC, typename SPEC>
    void zero_gradient(devices::CUDA<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        check_cuda_call(device, cudaMemsetAsync(layer.weights.gradient._data, 0, decltype(layer.weights.gradient)::SPEC::SIZE_BYTES, device.stream), "cudaMemsetAsync conv2d weights.gradient");
        check_cuda_call(device, cudaMemsetAsync(layer.biases.gradient._data, 0, decltype(layer.biases.gradient)::SPEC::SIZE_BYTES, device.stream), "cudaMemsetAsync conv2d biases.gradient");
        if constexpr(SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM){
            check_cuda_call(device, cudaMemsetAsync(layer.norm.gamma.gradient._data, 0, decltype(layer.norm.gamma.gradient)::SPEC::SIZE_BYTES, device.stream), "cudaMemsetAsync conv2d norm.gamma.gradient");
            check_cuda_call(device, cudaMemsetAsync(layer.norm.beta.gradient._data, 0, decltype(layer.norm.beta.gradient)::SPEC::SIZE_BYTES, device.stream), "cudaMemsetAsync conv2d norm.beta.gradient");
        }
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
