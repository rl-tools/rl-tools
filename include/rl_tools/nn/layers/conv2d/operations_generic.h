#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_GENERIC_H

#include "../../../containers/tensor/tensor.h"
#include "../../../nn/parameters/operations_generic.h"

#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== NormForwardState malloc / free / copy / zero_gradient / update / _reset_optimizer_state ========================
    // NONE
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&) {}
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD&, TD&, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC1>&, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC2>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE&, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE&, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&, OPTIMIZER&) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE&, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&, OPTIMIZER&) {}
    template<typename DEVICE, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC1::TYPE_POLICY::DEFAULT abs_diff(DEVICE&, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC1>&, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC2>&) { return 0; }
    template<typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::NONE, SPEC>&, const Mode<MODE>&) { return false; }

    // BATCH_NORM
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm) {
        malloc(device, norm.gamma);
        malloc(device, norm.beta);
        malloc(device, norm.running_mean);
        malloc(device, norm.running_var);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm) {
        free(device, norm.gamma);
        free(device, norm.beta);
        free(device, norm.running_mean);
        free(device, norm.running_var);
    }
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC1>& src, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC2>& dst) {
        copy(sd, td, src.gamma, dst.gamma);
        copy(sd, td, src.beta, dst.beta);
        copy(sd, td, src.running_mean, dst.running_mean);
        copy(sd, td, src.running_var, dst.running_var);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm) {
        zero_gradient(device, norm.gamma);
        zero_gradient(device, norm.beta);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm, OPTIMIZER& optimizer) {
        update(device, norm.gamma, optimizer);
        update(device, norm.beta, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, norm.gamma, optimizer);
        _reset_optimizer_state(device, norm.beta, optimizer);
    }
    template<typename DEVICE, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC1>& a, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC2>& b) {
        using T = typename SPEC1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, a.gamma, b.gamma);
        acc += abs_diff(device, a.beta, b.beta);
        acc += abs_diff(device, a.running_mean, b.running_mean);
        acc += abs_diff(device, a.running_var, b.running_var);
        return acc;
    }
    template<typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& norm, const Mode<MODE>& mode) {
        return is_nan(device, norm.gamma, mode) || is_nan(device, norm.beta, mode) || is_nan(device, norm.running_mean, mode) || is_nan(device, norm.running_var, mode);
    }

    // LAYER_NORM
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm) {
        malloc(device, norm.gamma);
        malloc(device, norm.beta);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm) {
        free(device, norm.gamma);
        free(device, norm.beta);
    }
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC1>& src, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC2>& dst) {
        copy(sd, td, src.gamma, dst.gamma);
        copy(sd, td, src.beta, dst.beta);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm) {
        zero_gradient(device, norm.gamma);
        zero_gradient(device, norm.beta);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm, OPTIMIZER& optimizer) {
        update(device, norm.gamma, optimizer);
        update(device, norm.beta, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, norm.gamma, optimizer);
        _reset_optimizer_state(device, norm.beta, optimizer);
    }
    template<typename DEVICE, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC1>& a, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC2>& b) {
        using T = typename SPEC1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, a.gamma, b.gamma);
        acc += abs_diff(device, a.beta, b.beta);
        return acc;
    }
    template<typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::conv2d::NormForwardState<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& norm, const Mode<MODE>& mode) {
        return is_nan(device, norm.gamma, mode) || is_nan(device, norm.beta, mode);
    }

    // ======================== NormBackwardCache malloc / free / copy ========================
    // NONE
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::NONE, SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::NONE, SPEC>&) {}
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD&, TD&, const nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::NONE, SPEC1>&, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::NONE, SPEC2>&) {}
    // BATCH_NORM
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& cache) {
        malloc(device, cache.mean);
        malloc(device, cache.inv_std);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC>& cache) {
        free(device, cache.mean);
        free(device, cache.inv_std);
    }
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC1>& src, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::BATCH_NORM, SPEC2>& dst) {
        copy(sd, td, src.mean, dst.mean);
        copy(sd, td, src.inv_std, dst.inv_std);
    }
    // LAYER_NORM
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& cache) {
        malloc(device, cache.mean);
        malloc(device, cache.inv_std);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC>& cache) {
        free(device, cache.mean);
        free(device, cache.inv_std);
    }
    template<typename SD, typename TD, typename SPEC1, typename SPEC2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC1>& src, nn::layers::conv2d::NormBackwardCache<nn::layers::conv2d::Normalization::LAYER_NORM, SPEC2>& dst) {
        copy(sd, td, src.mean, dst.mean);
        copy(sd, td, src.inv_std, dst.inv_std);
    }

    // ======================== Layer malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer) {
        malloc(device, layer.weights);
        malloc(device, layer.biases);
        malloc(device, layer.norm);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer) {
        free(device, layer.weights);
        free(device, layer.biases);
        free(device, layer.norm);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::LayerBackward<SPEC>& layer) {
        malloc(device, (nn::layers::conv2d::LayerForward<SPEC>&) layer);
        malloc(device, layer.pre_activations);
        malloc(device, layer.norm_cache);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::LayerBackward<SPEC>& layer) {
        free(device, (nn::layers::conv2d::LayerForward<SPEC>&) layer);
        free(device, layer.pre_activations);
        free(device, layer.norm_cache);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        malloc(device, (nn::layers::conv2d::LayerBackward<SPEC>&) layer);
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        free(device, (nn::layers::conv2d::LayerBackward<SPEC>&) layer);
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::State& state) { }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn::layers::conv2d::State& source, nn::layers::conv2d::State& target){}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn::layers::conv2d::LayerForward<SPEC>& layer, nn::layers::conv2d::State& state, RNG&, Mode<MODE> mode = Mode<mode::Default<>>{}) { }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::State& state) { }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::conv2d::Buffer& buffer) { }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::conv2d::Buffer& buffer) { }

    // ======================== init_weights ========================
    template<typename DEVICE, typename SPEC, typename INITIALIZER_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer, const nn::layers::conv2d::KaimingUniform<INITIALIZER_SPEC>& initializer, RNG& rng){
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        T gain;
        if constexpr(INITIALIZER_SPEC::INIT_LEGACY){
            T negative_slope = math::sqrt(device.math, (T)5);
            gain = math::sqrt(device.math, (T)2.0 / (1 + negative_slope * negative_slope));
        }
        else{
            gain = math::sqrt(device.math, (T)2.0) * INITIALIZER_SPEC::SCALE;
        }
        // fan_in for Conv2d = INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH
        T fan = (T)(SPEC::INPUT_CHANNELS * SPEC::KERNEL_HEIGHT * SPEC::KERNEL_WIDTH);
        T std = gain / math::sqrt(device.math, fan);
        T weight_bound = math::sqrt(device.math, (T)3.0) * std;
        T bias_bound = 1/math::sqrt(device.math, fan);
        using PARAMETER_TYPE = typename decltype(layer.weights.parameters)::SPEC::T;
        for(TI oc = 0; oc < SPEC::OUTPUT_CHANNELS; oc++) {
            if constexpr(INITIALIZER_SPEC::INIT_LEGACY) {
                set(device, layer.biases.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -bias_bound, bias_bound, rng), oc);
            }
            else{
                set(device, layer.biases.parameters, (PARAMETER_TYPE)0, oc);
            }
            for(TI ic = 0; ic < SPEC::INPUT_CHANNELS; ic++) {
                for(TI kh = 0; kh < SPEC::KERNEL_HEIGHT; kh++) {
                    for(TI kw = 0; kw < SPEC::KERNEL_WIDTH; kw++) {
                        set(device, layer.weights.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -weight_bound, weight_bound, rng), oc, ic, kh, kw);
                    }
                }
            }
        }
        // Initialize normalization parameters
        if constexpr(SPEC::NORMALIZATION != nn::layers::conv2d::Normalization::NONE) {
            for(TI oc = 0; oc < SPEC::OUTPUT_CHANNELS; oc++) {
                set(device, layer.norm.gamma.parameters, (PARAMETER_TYPE)1, oc);
                set(device, layer.norm.beta.parameters, (PARAMETER_TYPE)0, oc);
            }
            if constexpr(SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                for(TI oc = 0; oc < SPEC::OUTPUT_CHANNELS; oc++) {
                    set(device, layer.norm.running_mean.parameters, (PARAMETER_TYPE)0, oc);
                    set(device, layer.norm.running_var.parameters, (PARAMETER_TYPE)1, oc);
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::conv2d::LayerForward<SPEC>& layer, RNG& rng) {
        init_weights(device, layer, typename SPEC::INITIALIZER{}, rng);
    }

    // ======================== evaluate (LayerForward, no storage) ========================
#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr auto NORMALIZATION = LAYER_SPEC::NORMALIZATION;
        // Reshape to 4D: [INTERNAL_BATCH_SIZE, H, W, C]
        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);

        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::NONE) {
            // Fused conv + activation
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T acc = get(device, layer.biases.parameters, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            acc += get(device, layer.weights.parameters, oc, ic, kh, kw) * get(device, input_4d, bi, ih, iw, ic);
                                        }
                                    }
                                }
                            }
                            set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(acc), bi, oh, ow, oc);
                        }
                    }
                }
            }
        } else {
            // Phase 1: Compute conv output -> store temporarily in output
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T acc = get(device, layer.biases.parameters, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            acc += get(device, layer.weights.parameters, oc, ic, kh, kw) * get(device, input_4d, bi, ih, iw, ic);
                                        }
                                    }
                                }
                            }
                            set(device, output_4d, acc, bi, oh, ow, oc);
                        }
                    }
                }
            }

            constexpr T eps = (T)LAYER_SPEC::NORM_EPSILON;

            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                // Evaluate mode for BN: use running statistics
                for(TI bi = 0; bi < BATCH_SIZE; bi++){
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                T conv_out = get(device, output_4d, bi, oh, ow, oc);
                                T inv_std = (T)1 / math::sqrt(device.math, get(device, layer.norm.running_var.parameters, oc) + eps);
                                T z_hat = (conv_out - get(device, layer.norm.running_mean.parameters, oc)) * inv_std;
                                T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                                set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out), bi, oh, ow, oc);
                            }
                        }
                    }
                }
            } else { // LAYER_NORM
                // Compute per-sample statistics on the fly
                constexpr TI N_LN = LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS;
                for(TI bi = 0; bi < BATCH_SIZE; bi++){
                    // Compute mean
                    T mean_val = 0;
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                mean_val += get(device, output_4d, bi, oh, ow, oc);
                            }
                        }
                    }
                    mean_val /= (T)N_LN;
                    // Compute variance
                    T var_val = 0;
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                T diff = get(device, output_4d, bi, oh, ow, oc) - mean_val;
                                var_val += diff * diff;
                            }
                        }
                    }
                    var_val /= (T)N_LN;
                    T inv_std = (T)1 / math::sqrt(device.math, var_val + eps);
                    // Normalize + activate
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                T conv_out = get(device, output_4d, bi, oh, ow, oc);
                                T z_hat = (conv_out - mean_val) * inv_std;
                                T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                                set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out), bi, oh, ow, oc);
                            }
                        }
                    }
                }
            }
        }
    }

    // ======================== forward (LayerBackward, stores pre_activations and caches stats) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr auto NORMALIZATION = LAYER_SPEC::NORMALIZATION;

        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);

        // Phase 1: Compute conv output -> store in pre_activations
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                        T acc = get(device, layer.biases.parameters, oc);
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                        acc += get(device, layer.weights.parameters, oc, ic, kh, kw) * get(device, input_4d, bi, ih, iw, ic);
                                    }
                                }
                            }
                        }
                        set(device, layer.pre_activations, acc, bi, oh, ow, oc);
                    }
                }
            }
        }

        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::NONE) {
            // Apply activation directly
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T pre_act = get(device, layer.pre_activations, bi, oh, ow, oc);
                            set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(pre_act), bi, oh, ow, oc);
                        }
                    }
                }
            }
        } else if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
            constexpr T eps = (T)LAYER_SPEC::NORM_EPSILON;
            constexpr T momentum = (T)LAYER_SPEC::BN_MOMENTUM;
            constexpr TI N_BN = BATCH_SIZE * LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH;

            // Phase 2: Compute batch statistics per channel
            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                T mean_val = 0;
                for(TI bi = 0; bi < BATCH_SIZE; bi++){
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            mean_val += get(device, layer.pre_activations, bi, oh, ow, oc);
                        }
                    }
                }
                mean_val /= (T)N_BN;

                T var_val = 0;
                for(TI bi = 0; bi < BATCH_SIZE; bi++){
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            T diff = get(device, layer.pre_activations, bi, oh, ow, oc) - mean_val;
                            var_val += diff * diff;
                        }
                    }
                }
                var_val /= (T)N_BN;
                T inv_std = (T)1 / math::sqrt(device.math, var_val + eps);

                // Cache statistics
                set(device, layer.norm_cache.mean, mean_val, oc);
                set(device, layer.norm_cache.inv_std, inv_std, oc);

                // Update running statistics (EMA)
                T running_mean = get(device, layer.norm.running_mean.parameters, oc);
                T running_var = get(device, layer.norm.running_var.parameters, oc);
                set(device, layer.norm.running_mean.parameters, ((T)1 - momentum) * running_mean + momentum * mean_val, oc);
                set(device, layer.norm.running_var.parameters, ((T)1 - momentum) * running_var + momentum * var_val, oc);
            }

            // Phase 3: Normalize + activate -> output
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            T z_hat = (conv_out - get(device, layer.norm_cache.mean, oc)) * get(device, layer.norm_cache.inv_std, oc);
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out), bi, oh, ow, oc);
                        }
                    }
                }
            }
        } else { // LAYER_NORM
            constexpr T eps = (T)LAYER_SPEC::NORM_EPSILON;
            constexpr TI N_LN = LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS;

            // Phase 2: Compute per-sample statistics
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                T mean_val = 0;
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            mean_val += get(device, layer.pre_activations, bi, oh, ow, oc);
                        }
                    }
                }
                mean_val /= (T)N_LN;

                T var_val = 0;
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T diff = get(device, layer.pre_activations, bi, oh, ow, oc) - mean_val;
                            var_val += diff * diff;
                        }
                    }
                }
                var_val /= (T)N_LN;
                T inv_std = (T)1 / math::sqrt(device.math, var_val + eps);

                set(device, layer.norm_cache.mean, mean_val, bi);
                set(device, layer.norm_cache.inv_std, inv_std, bi);
            }

            // Phase 3: Normalize + activate -> output
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                T mean_val = get(device, layer.norm_cache.mean, bi);
                T inv_std = get(device, layer.norm_cache.inv_std, bi);
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            T z_hat = (conv_out - mean_val) * inv_std;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            set(device, output_4d, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out), bi, oh, ow, oc);
                        }
                    }
                }
            }
        }
    }
#endif

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::conv2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::conv2d::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== Normalization backward helper ========================
    // Computes d_conv_out from d_output given cached forward stats.
    // Also optionally accumulates d_gamma and d_beta.
    // Returns nothing; caller uses d_conv_out inline in phase 2.
    // This is integrated directly into the backward functions below.

    // ======================== backward_input ========================
#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename DEVICE::index_t;
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr auto NORMALIZATION = LAYER_SPEC::NORMALIZATION;

        using INTERNAL_D_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto d_input_4d = view_memory<INTERNAL_D_INPUT_SHAPE>(device, d_input);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        set_all(device, d_input_4d, (T)0);

        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::NONE) {
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(device, layer.pre_activations, bi, oh, ow, oc)) * get(device, d_output_4d, bi, oh, ow, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, d_input_4d, get(device, layer.weights.parameters, oc, ic, kh, kw) * d_pre_act, bi, ih, iw, ic);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Two-phase normalization backward
            constexpr TI NORM_DIM = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ? LAYER_SPEC::OUTPUT_CHANNELS : BATCH_SIZE;
            constexpr TI N = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ?
                (BATCH_SIZE * LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH) :
                (LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS);

            T sum_dz_hat[NORM_DIM];
            T sum_dz_hat_z_hat[NORM_DIM];
            for(TI i = 0; i < NORM_DIM; i++){
                sum_dz_hat[i] = 0;
                sum_dz_hat_z_hat[i] = 0;
            }

            // Phase 1: Compute sums for norm backward
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            sum_dz_hat[stat_idx] += d_z_hat_val;
                            sum_dz_hat_z_hat[stat_idx] += d_z_hat_val * z_hat;
                        }
                    }
                }
            }

            // Phase 2: Compute d_conv_out and propagate to d_input
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            T d_conv_out = inv_std_val * ((T)1 / (T)N) * ((T)N * d_z_hat_val - sum_dz_hat[stat_idx] - z_hat * sum_dz_hat_z_hat[stat_idx]);

                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, d_input_4d, get(device, layer.weights.parameters, oc, ic, kh, kw) * d_conv_out, bi, ih, iw, ic);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ======================== backward (gradient accumulation only) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::conv2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using TI = typename DEVICE::index_t;
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr auto NORMALIZATION = LAYER_SPEC::NORMALIZATION;

        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::NONE) {
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(device, layer.pre_activations, bi, oh, ow, oc)) * get(device, d_output_4d, bi, oh, ow, oc);
                            increment(device, layer.biases.gradient, d_pre_act, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, layer.weights.gradient, d_pre_act * get(device, input_4d, bi, ih, iw, ic), oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            constexpr TI NORM_DIM = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ? LAYER_SPEC::OUTPUT_CHANNELS : BATCH_SIZE;
            constexpr TI N = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ?
                (BATCH_SIZE * LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH) :
                (LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS);

            T sum_dz_hat[NORM_DIM];
            T sum_dz_hat_z_hat[NORM_DIM];
            for(TI i = 0; i < NORM_DIM; i++){
                sum_dz_hat[i] = 0;
                sum_dz_hat_z_hat[i] = 0;
            }

            // Phase 1: Compute sums + accumulate d_gamma, d_beta
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            // Accumulate d_gamma and d_beta
                            increment(device, layer.norm.gamma.gradient, d_norm_out * z_hat, oc);
                            increment(device, layer.norm.beta.gradient, d_norm_out, oc);
                            sum_dz_hat[stat_idx] += d_z_hat_val;
                            sum_dz_hat_z_hat[stat_idx] += d_z_hat_val * z_hat;
                        }
                    }
                }
            }

            // Phase 2: Compute d_conv_out and accumulate d_weights, d_biases
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            T d_conv_out = inv_std_val * ((T)1 / (T)N) * ((T)N * d_z_hat_val - sum_dz_hat[stat_idx] - z_hat * sum_dz_hat_z_hat[stat_idx]);

                            increment(device, layer.biases.gradient, d_conv_out, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, layer.weights.gradient, d_conv_out * get(device, input_4d, bi, ih, iw, ic), oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ======================== backward_full (d_input + gradient accumulation) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using TI = typename DEVICE::index_t;
        using T = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Gradient>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr auto NORMALIZATION = LAYER_SPEC::NORMALIZATION;

        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_D_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto d_input_4d = view_memory<INTERNAL_D_INPUT_SHAPE>(device, d_input);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        set_all(device, d_input_4d, (T)0);

        if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::NONE) {
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(device, layer.pre_activations, bi, oh, ow, oc)) * get(device, d_output_4d, bi, oh, ow, oc);
                            increment(device, layer.biases.gradient, d_pre_act, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, d_input_4d, get(device, layer.weights.parameters, oc, ic, kh, kw) * d_pre_act, bi, ih, iw, ic);
                                            increment(device, layer.weights.gradient, d_pre_act * get(device, input_4d, bi, ih, iw, ic), oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            constexpr TI NORM_DIM = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ? LAYER_SPEC::OUTPUT_CHANNELS : BATCH_SIZE;
            constexpr TI N = (NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) ?
                (BATCH_SIZE * LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH) :
                (LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS);

            T sum_dz_hat[NORM_DIM];
            T sum_dz_hat_z_hat[NORM_DIM];
            for(TI i = 0; i < NORM_DIM; i++){
                sum_dz_hat[i] = 0;
                sum_dz_hat_z_hat[i] = 0;
            }

            // Phase 1: Compute sums + accumulate d_gamma, d_beta
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            increment(device, layer.norm.gamma.gradient, d_norm_out * z_hat, oc);
                            increment(device, layer.norm.beta.gradient, d_norm_out, oc);
                            sum_dz_hat[stat_idx] += d_z_hat_val;
                            sum_dz_hat_z_hat[stat_idx] += d_z_hat_val * z_hat;
                        }
                    }
                }
            }

            // Phase 2: Compute d_conv_out, accumulate d_weights, d_biases, d_input
            for(TI bi = 0; bi < BATCH_SIZE; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            T conv_out = get(device, layer.pre_activations, bi, oh, ow, oc);
                            TI stat_idx;
                            T mean_val, inv_std_val;
                            if constexpr(NORMALIZATION == nn::layers::conv2d::Normalization::BATCH_NORM) {
                                stat_idx = oc;
                                mean_val = get(device, layer.norm_cache.mean, oc);
                                inv_std_val = get(device, layer.norm_cache.inv_std, oc);
                            } else {
                                stat_idx = bi;
                                mean_val = get(device, layer.norm_cache.mean, bi);
                                inv_std_val = get(device, layer.norm_cache.inv_std, bi);
                            }
                            T z_hat = (conv_out - mean_val) * inv_std_val;
                            T norm_out = get(device, layer.norm.gamma.parameters, oc) * z_hat + get(device, layer.norm.beta.parameters, oc);
                            T d_norm_out = d_activation_d_x<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(norm_out) * get(device, d_output_4d, bi, oh, ow, oc);
                            T d_z_hat_val = d_norm_out * get(device, layer.norm.gamma.parameters, oc);
                            T d_conv_out = inv_std_val * ((T)1 / (T)N) * ((T)N * d_z_hat_val - sum_dz_hat[stat_idx] - z_hat * sum_dz_hat_z_hat[stat_idx]);

                            increment(device, layer.biases.gradient, d_conv_out, oc);
                            for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                                for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                    TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                    TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                    if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                       iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                        TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                        TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                        for(TI ic = 0; ic < LAYER_SPEC::INPUT_CHANNELS; ic++){
                                            increment(device, d_input_4d, get(device, layer.weights.parameters, oc, ic, kh, kw) * d_conv_out, bi, ih, iw, ic);
                                            increment(device, layer.weights.gradient, d_conv_out * get(device, input_4d, bi, ih, iw, ic), oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    // ======================== zero_gradient / update / _reset_optimizer_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        zero_gradient(device, layer.weights);
        zero_gradient(device, layer.biases);
        zero_gradient(device, layer.norm);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer){
        update(device, layer.weights, optimizer);
        update(device, layer.biases, optimizer);
        update(device, layer.norm, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, layer.weights, optimizer);
        _reset_optimizer_state(device, layer.biases, optimizer);
        _reset_optimizer_state(device, layer.norm, optimizer);
    }

    // ======================== copy ========================
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::conv2d::LayerForward<SOURCE_SPEC>& source, nn::layers::conv2d::LayerForward<TARGET_SPEC>& target){
        static_assert(nn::layers::conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.weights, target.weights);
        copy(source_device, target_device, source.biases, target.biases);
        copy(source_device, target_device, source.norm, target.norm);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::conv2d::LayerBackward<SOURCE_SPEC>& source, nn::layers::conv2d::LayerBackward<TARGET_SPEC>& target){
        static_assert(nn::layers::conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::conv2d::LayerForward<SOURCE_SPEC>&>(source), static_cast<nn::layers::conv2d::LayerForward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.pre_activations, target.pre_activations);
        copy(source_device, target_device, source.norm_cache, target.norm_cache);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::conv2d::LayerGradient<SOURCE_SPEC>& source, nn::layers::conv2d::LayerGradient<TARGET_SPEC>& target){
        static_assert(nn::layers::conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::conv2d::LayerBackward<SOURCE_SPEC>&>(source), static_cast<nn::layers::conv2d::LayerBackward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerForward<SPEC_1>& l1, const rl_tools::nn::layers::conv2d::LayerForward<SPEC_2>& l2) {
        static_assert(nn::layers::conv2d::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, l1.weights, l2.weights);
        acc += abs_diff(device, l1.biases, l2.biases);
        acc += abs_diff(device, l1.norm, l2.norm);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerBackward<SPEC_1>& l1, const rl_tools::nn::layers::conv2d::LayerBackward<SPEC_2>& l2) {
        static_assert(nn::layers::conv2d::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = abs_diff(device, static_cast<const rl_tools::nn::layers::conv2d::LayerForward<SPEC_1>&>(l1), static_cast<const rl_tools::nn::layers::conv2d::LayerForward<SPEC_2>&>(l2));
        acc += abs_diff(device, l1.pre_activations, l2.pre_activations);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerGradient<SPEC_1>& l1, const rl_tools::nn::layers::conv2d::LayerGradient<SPEC_2>& l2) {
        static_assert(nn::layers::conv2d::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = abs_diff(device, static_cast<const rl_tools::nn::layers::conv2d::LayerBackward<SPEC_1>&>(l1), static_cast<const rl_tools::nn::layers::conv2d::LayerBackward<SPEC_2>&>(l2));
        acc += abs_diff(device, l1.output, l2.output);
        return acc;
    }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const rl_tools::nn::layers::conv2d::State& s1, const rl_tools::nn::layers::conv2d::State& s2) {
        return 0;
    }

    // ======================== reset_forward_state ========================
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, rl_tools::nn::layers::conv2d::LayerBackward<SPEC>& l) {
        set_all(device, l.pre_activations, 0);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, rl_tools::nn::layers::conv2d::LayerGradient<SPEC>& l) {
        reset_forward_state(device, static_cast<rl_tools::nn::layers::conv2d::LayerBackward<SPEC>&>(l));
        set_all(device, l.output, 0);
    }

    // ======================== is_nan ========================
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerForward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return is_nan(device, l.weights, mode) || is_nan(device, l.biases, mode) || is_nan(device, l.norm, mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerBackward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        bool upstream_nan = is_nan(device, static_cast<const rl_tools::nn::layers::conv2d::LayerForward<SPEC>&>(l), mode);
        if(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        return upstream_nan || is_nan(device, l.pre_activations, mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::conv2d::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        bool upstream_nan = is_nan(device, static_cast<const rl_tools::nn::layers::conv2d::LayerBackward<SPEC>&>(l), mode);
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        return upstream_nan || is_nan(device, l.output, mode);
    }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn::layers::conv2d::State& state, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return false;
    }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::conv2d::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }

    // ======================== gradient_norm ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto gradient_norm(DEVICE& device, const nn::layers::conv2d::LayerGradient<SPEC>& layer) {
        auto acc = gradient_norm(device, layer.weights) + gradient_norm(device, layer.biases);
        if constexpr(SPEC::NORMALIZATION != nn::layers::conv2d::Normalization::NONE) {
            acc += gradient_norm(device, layer.norm.gamma) + gradient_norm(device, layer.norm.beta);
        }
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
