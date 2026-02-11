#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_RESNET_BLOCK_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_RESNET_BLOCK_OPERATIONS_GENERIC_H
#include "../../../containers/tensor/tensor.h"
#include "../conv2d/operations_generic.h"
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== DownsampleStorage malloc / free / copy / etc ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&) {}
    template<typename SD, typename TD, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD&, TD&, const nn::layers::resnet_block::DownsampleStorage<false, S1>&, nn::layers::resnet_block::DownsampleStorage<false, S2>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, OPTIMIZER&) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, OPTIMIZER&) {}
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE&, const nn::layers::resnet_block::DownsampleStorage<false, S1>&, const nn::layers::resnet_block::DownsampleStorage<false, S2>&) { return 0; }
    template<typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, const Mode<MODE>&) { return false; }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&) {}
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE&, nn::layers::resnet_block::DownsampleStorage<false, SPEC>&, RNG&) {}

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds) { malloc(device, ds.conv); }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds) { free(device, ds.conv); }
    template<typename SD, typename TD, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::resnet_block::DownsampleStorage<true, S1>& src, nn::layers::resnet_block::DownsampleStorage<true, S2>& dst) { copy(sd, td, src.conv, dst.conv); }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds) { zero_gradient(device, ds.conv); }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, OPTIMIZER& opt) { update(device, ds.conv, opt); }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, OPTIMIZER& opt) { _reset_optimizer_state(device, ds.conv, opt); }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::resnet_block::DownsampleStorage<true, S1>& a, const nn::layers::resnet_block::DownsampleStorage<true, S2>& b) { return abs_diff(device, a.conv, b.conv); }
    template<typename DEVICE, typename SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, const Mode<MODE>& mode) { return is_nan(device, ds.conv, mode); }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds) { reset_forward_state(device, ds.conv); }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::resnet_block::DownsampleStorage<true, SPEC>& ds, RNG& rng) { init_weights(device, ds.conv, rng); }

    // ======================== Buffer malloc / free ========================
    template<typename DEVICE, bool DA, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::resnet_block::Buffer<DA, SPEC>& buffer) {
        malloc(device, buffer.intermediate);
        malloc(device, buffer.shortcut);
        malloc(device, buffer.conv1_buffer);
        malloc(device, buffer.conv2_buffer);
        malloc(device, buffer.downsample_buffer);
    }
    template<typename DEVICE, bool DA, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::resnet_block::Buffer<DA, SPEC>& buffer) {
        free(device, buffer.intermediate);
        free(device, buffer.shortcut);
        free(device, buffer.conv1_buffer);
        free(device, buffer.conv2_buffer);
        free(device, buffer.downsample_buffer);
    }

    // ======================== Layer malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::resnet_block::LayerForward<SPEC>& layer) {
        malloc(device, layer.conv1);
        malloc(device, layer.conv2);
        malloc(device, layer.downsample);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::resnet_block::LayerForward<SPEC>& layer) {
        free(device, layer.conv1);
        free(device, layer.conv2);
        free(device, layer.downsample);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::resnet_block::LayerBackward<SPEC>& layer) {
        malloc(device, (nn::layers::resnet_block::LayerForward<SPEC>&)layer);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::resnet_block::LayerBackward<SPEC>& layer) {
        free(device, (nn::layers::resnet_block::LayerForward<SPEC>&)layer);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer) {
        malloc(device, (nn::layers::resnet_block::LayerBackward<SPEC>&)layer);
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer) {
        free(device, (nn::layers::resnet_block::LayerBackward<SPEC>&)layer);
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, nn::layers::resnet_block::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, nn::layers::resnet_block::State&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::resnet_block::State&, nn::layers::resnet_block::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::resnet_block::LayerForward<SPEC>&, nn::layers::resnet_block::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::resnet_block::LayerForward<SPEC>& layer, RNG& rng) {
        init_weights(device, layer.conv1, rng);
        init_weights(device, layer.conv2, rng);
        init_weights(device, layer.downsample, rng);
    }

    // ======================== evaluate ========================
#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::resnet_block::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI BATCH = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI OC = LAYER_SPEC::OUTPUT_CHANNELS;

        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH, OH, OW, OC>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);

        // Conv1: input → intermediate
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV1_SPEC>&>(layer.conv1), input, buffer.intermediate, buffer.conv1_buffer, rng, mode);
        // Conv2: intermediate → output (temporary, before skip)
        evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::CONV2_SPEC>&>(layer.conv2), buffer.intermediate, output, buffer.conv2_buffer, rng, mode);
        // Downsample shortcut if needed
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE) {
            evaluate(device, static_cast<const nn::layers::conv2d::LayerForward<typename LAYER_SPEC::DOWNSAMPLE_SPEC>&>(layer.downsample.conv), input, buffer.shortcut, buffer.downsample_buffer, rng, mode);
        }
        // output = ReLU(conv2_out + shortcut)
        for(TI bi = 0; bi < BATCH; bi++){
            for(TI h = 0; h < OH; h++){
                for(TI w = 0; w < OW; w++){
                    for(TI c = 0; c < OC; c++){
                        T conv2_val = get(device, output_4d, bi, h, w, c);
                        T shortcut_val;
                        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE) {
                            shortcut_val = get(device, buffer.shortcut, bi, h, w, c);
                        } else {
                            shortcut_val = get(device, input_4d, bi, h, w, c);
                        }
                        set(device, output_4d, math::max(device.math, conv2_val + shortcut_val, (T)0), bi, h, w, c);
                    }
                }
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::resnet_block::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI BATCH = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI OC = LAYER_SPEC::OUTPUT_CHANNELS;

        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<tensor::Shape<TI, BATCH, OH, OW, OC>>(device, output);

        // Conv1: input → conv1.output
        forward(device, layer.conv1, input, buffer.conv1_buffer, rng, mode);
        // Conv2: conv1.output → conv2.output
        auto conv1_out = rl_tools::output(device, layer.conv1);
        forward(device, layer.conv2, conv1_out, buffer.conv2_buffer, rng, mode);
        // Downsample if needed
        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE) {
            forward(device, layer.downsample.conv, input, buffer.downsample_buffer, rng, mode);
        }
        // output = ReLU(conv2.output + shortcut)
        auto conv2_out = rl_tools::output(device, layer.conv2);
        for(TI bi = 0; bi < BATCH; bi++){
            for(TI h = 0; h < OH; h++){
                for(TI w = 0; w < OW; w++){
                    for(TI c = 0; c < OC; c++){
                        T conv2_val = get(device, conv2_out, bi, h, w, c);
                        T shortcut_val;
                        if constexpr(LAYER_SPEC::HAS_DOWNSAMPLE) {
                            auto ds_out = rl_tools::output(device, layer.downsample.conv);
                            shortcut_val = get(device, ds_out, bi, h, w, c);
                        } else {
                            shortcut_val = get(device, input_4d, bi, h, w, c);
                        }
                        set(device, output_4d, math::max(device.math, conv2_val + shortcut_val, (T)0), bi, h, w, c);
                    }
                }
            }
        }
    }
#endif

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::resnet_block::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::resnet_block::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::resnet_block::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::resnet_block::Buffer<true, LAYER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== zero_gradient / update / _reset_optimizer_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer) {
        zero_gradient(device, layer.conv1);
        zero_gradient(device, layer.conv2);
        zero_gradient(device, layer.downsample);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        update(device, layer.conv1, optimizer);
        update(device, layer.conv2, optimizer);
        update(device, layer.downsample, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, layer.conv1, optimizer);
        _reset_optimizer_state(device, layer.conv2, optimizer);
        _reset_optimizer_state(device, layer.downsample, optimizer);
    }

    // ======================== copy ========================
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::resnet_block::LayerForward<SS>& src, nn::layers::resnet_block::LayerForward<TS>& dst) {
        copy(sd, td, src.conv1, dst.conv1);
        copy(sd, td, src.conv2, dst.conv2);
        copy(sd, td, src.downsample, dst.downsample);
    }
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::resnet_block::LayerBackward<SS>& src, nn::layers::resnet_block::LayerBackward<TS>& dst) {
        copy(sd, td, static_cast<const nn::layers::resnet_block::LayerForward<SS>&>(src), static_cast<nn::layers::resnet_block::LayerForward<TS>&>(dst));
    }
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::resnet_block::LayerGradient<SS>& src, nn::layers::resnet_block::LayerGradient<TS>& dst) {
        copy(sd, td, static_cast<const nn::layers::resnet_block::LayerBackward<SS>&>(src), static_cast<nn::layers::resnet_block::LayerBackward<TS>&>(dst));
        copy(sd, td, src.output, dst.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::resnet_block::LayerForward<S1>& l1, const nn::layers::resnet_block::LayerForward<S2>& l2) {
        using T = typename S1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, l1.conv1, l2.conv1);
        acc += abs_diff(device, l1.conv2, l2.conv2);
        acc += abs_diff(device, l1.downsample, l2.downsample);
        return acc;
    }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::resnet_block::LayerBackward<S1>& l1, const nn::layers::resnet_block::LayerBackward<S2>& l2) {
        return abs_diff(device, static_cast<const nn::layers::resnet_block::LayerForward<S1>&>(l1), static_cast<const nn::layers::resnet_block::LayerForward<S2>&>(l2));
    }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::resnet_block::LayerGradient<S1>& l1, const nn::layers::resnet_block::LayerGradient<S2>& l2) {
        auto acc = abs_diff(device, static_cast<const nn::layers::resnet_block::LayerBackward<S1>&>(l1), static_cast<const nn::layers::resnet_block::LayerBackward<S2>&>(l2));
        acc += abs_diff(device, l1.output, l2.output);
        return acc;
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::resnet_block::State&, const nn::layers::resnet_block::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::resnet_block::LayerBackward<SPEC>& l) {
        reset_forward_state(device, l.conv1);
        reset_forward_state(device, l.conv2);
        reset_forward_state(device, l.downsample);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& l) {
        reset_forward_state(device, static_cast<nn::layers::resnet_block::LayerBackward<SPEC>&>(l));
        set_all(device, l.output, 0);
    }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::resnet_block::LayerForward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, l.conv1, mode) || is_nan(device, l.conv2, mode) || is_nan(device, l.downsample, mode);
    }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::resnet_block::LayerBackward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, static_cast<const nn::layers::resnet_block::LayerForward<SPEC>&>(l), mode);
    }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::resnet_block::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, static_cast<const nn::layers::resnet_block::LayerBackward<SPEC>&>(l), mode) || is_nan(device, l.output, mode);
    }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::resnet_block::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::resnet_block::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
