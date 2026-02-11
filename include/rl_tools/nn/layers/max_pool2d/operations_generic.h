#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_MAX_POOL2D_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_MAX_POOL2D_OPERATIONS_GENERIC_H
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::max_pool2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::max_pool2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::max_pool2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::max_pool2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer) {
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer) {
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::max_pool2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::max_pool2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::max_pool2d::Buffer&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::max_pool2d::Buffer&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::max_pool2d::State&, nn::layers::max_pool2d::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::max_pool2d::LayerForward<SPEC>&, nn::layers::max_pool2d::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights (no-op) ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::max_pool2d::LayerForward<SPEC>& layer, RNG& rng) {}

    // ======================== evaluate ========================
#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::max_pool2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::max_pool2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < LAYER_SPEC::OUTPUT_CHANNELS; c++){
                        bool first = true;
                        T max_val = 0;
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    T val = get(device, input_4d, bi, ih, iw, c);
                                    if(first || val > max_val){
                                        max_val = val;
                                        first = false;
                                    }
                                }
                            }
                        }
                        set(device, output_4d, max_val, bi, oh, ow, c);
                    }
                }
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::max_pool2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::max_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::max_pool2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
#endif
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::max_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::max_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::max_pool2d::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::max_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::max_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== zero_gradient / update / _reset_optimizer_state (no-ops) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}

    // ======================== copy ========================
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::max_pool2d::LayerForward<SS>& src, nn::layers::max_pool2d::LayerForward<TS>& dst) {}
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::max_pool2d::LayerBackward<SS>& src, nn::layers::max_pool2d::LayerBackward<TS>& dst) {}
    template<typename SD, typename TD, typename SS, typename TS>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SD& sd, TD& td, const nn::layers::max_pool2d::LayerGradient<SS>& src, nn::layers::max_pool2d::LayerGradient<TS>& dst) {
        copy(sd, td, src.output, dst.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::max_pool2d::LayerForward<S1>&, const nn::layers::max_pool2d::LayerForward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::max_pool2d::LayerBackward<S1>&, const nn::layers::max_pool2d::LayerBackward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::max_pool2d::LayerGradient<S1>& l1, const nn::layers::max_pool2d::LayerGradient<S2>& l2) { return abs_diff(device, l1.output, l2.output); }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::max_pool2d::State&, const nn::layers::max_pool2d::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::max_pool2d::LayerBackward<SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& l) { set_all(device, l.output, 0); }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::max_pool2d::LayerForward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::max_pool2d::LayerBackward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::max_pool2d::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) { return is_nan(device, l.output, mode); }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::max_pool2d::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::max_pool2d::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
