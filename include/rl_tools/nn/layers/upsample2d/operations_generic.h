#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_UPSAMPLE2D_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_UPSAMPLE2D_OPERATIONS_GENERIC_H
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::upsample2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::upsample2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::upsample2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::upsample2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& layer) {
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& layer) {
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::upsample2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::upsample2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::upsample2d::Buffer&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::upsample2d::Buffer&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::upsample2d::State&, nn::layers::upsample2d::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::upsample2d::LayerForward<SPEC>&, nn::layers::upsample2d::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights (no-op) ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::upsample2d::LayerForward<SPEC>& layer, RNG& rng) {}

    // ======================== evaluate (bilinear upsample) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::upsample2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::upsample2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI BATCH_SIZE = product(typename INPUT_SPEC::SHAPE{}) / (IH * IW * C);
        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, IH, IW, C>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, OH, OW, C>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oy = 0; oy < OH; oy++){
                for(TI ox = 0; ox < OW; ox++){
                    // Source coordinate (align_corners=false, half-pixel offset)
                    ACCUMULATOR_TYPE sy = ((ACCUMULATOR_TYPE)oy + (ACCUMULATOR_TYPE)0.5) * (ACCUMULATOR_TYPE)IH / (ACCUMULATOR_TYPE)OH - (ACCUMULATOR_TYPE)0.5;
                    ACCUMULATOR_TYPE sx = ((ACCUMULATOR_TYPE)ox + (ACCUMULATOR_TYPE)0.5) * (ACCUMULATOR_TYPE)IW / (ACCUMULATOR_TYPE)OW - (ACCUMULATOR_TYPE)0.5;
                    TI y0 = sy < 0 ? 0 : (TI)sy;
                    TI x0 = sx < 0 ? 0 : (TI)sx;
                    TI y1 = y0 + 1 < IH ? y0 + 1 : IH - 1;
                    TI x1 = x0 + 1 < IW ? x0 + 1 : IW - 1;
                    if(y0 >= IH) y0 = IH - 1;
                    if(x0 >= IW) x0 = IW - 1;
                    ACCUMULATOR_TYPE fy = sy - (ACCUMULATOR_TYPE)y0;
                    ACCUMULATOR_TYPE fx = sx - (ACCUMULATOR_TYPE)x0;
                    if(fy < 0) fy = 0;
                    if(fx < 0) fx = 0;
                    ACCUMULATOR_TYPE w00 = ((ACCUMULATOR_TYPE)1 - fy) * ((ACCUMULATOR_TYPE)1 - fx);
                    ACCUMULATOR_TYPE w01 = ((ACCUMULATOR_TYPE)1 - fy) * fx;
                    ACCUMULATOR_TYPE w10 = fy * ((ACCUMULATOR_TYPE)1 - fx);
                    ACCUMULATOR_TYPE w11 = fy * fx;
                    for(TI c_i = 0; c_i < C; c_i++){
                        ACCUMULATOR_TYPE v = w00 * (ACCUMULATOR_TYPE)get(device, input_4d, bi, y0, x0, c_i)
                                           + w01 * (ACCUMULATOR_TYPE)get(device, input_4d, bi, y0, x1, c_i)
                                           + w10 * (ACCUMULATOR_TYPE)get(device, input_4d, bi, y1, x0, c_i)
                                           + w11 * (ACCUMULATOR_TYPE)get(device, input_4d, bi, y1, x1, c_i);
                        set(device, output_4d, (T)v, bi, oy, ox, c_i);
                    }
                }
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn::layers::upsample2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::upsample2d::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::upsample2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::upsample2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::upsample2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::upsample2d::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::upsample2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward ========================
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::upsample2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::upsample2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::upsample2d::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename D_OUTPUT_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI OH = LAYER_SPEC::OUTPUT_HEIGHT;
        constexpr TI OW = LAYER_SPEC::OUTPUT_WIDTH;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, OH, OW, C>;
        using INTERNAL_D_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, IH, IW, C>;
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);
        auto d_input_4d = view_memory<INTERNAL_D_INPUT_SHAPE>(device, d_input);
        // Zero d_input first (scatter-add pattern)
        set_all(device, d_input_4d, (T)0);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oy = 0; oy < OH; oy++){
                for(TI ox = 0; ox < OW; ox++){
                    ACCUMULATOR_TYPE sy = ((ACCUMULATOR_TYPE)oy + (ACCUMULATOR_TYPE)0.5) * (ACCUMULATOR_TYPE)IH / (ACCUMULATOR_TYPE)OH - (ACCUMULATOR_TYPE)0.5;
                    ACCUMULATOR_TYPE sx = ((ACCUMULATOR_TYPE)ox + (ACCUMULATOR_TYPE)0.5) * (ACCUMULATOR_TYPE)IW / (ACCUMULATOR_TYPE)OW - (ACCUMULATOR_TYPE)0.5;
                    TI y0 = sy < 0 ? 0 : (TI)sy;
                    TI x0 = sx < 0 ? 0 : (TI)sx;
                    TI y1 = y0 + 1 < IH ? y0 + 1 : IH - 1;
                    TI x1 = x0 + 1 < IW ? x0 + 1 : IW - 1;
                    if(y0 >= IH) y0 = IH - 1;
                    if(x0 >= IW) x0 = IW - 1;
                    ACCUMULATOR_TYPE fy = sy - (ACCUMULATOR_TYPE)y0;
                    ACCUMULATOR_TYPE fx = sx - (ACCUMULATOR_TYPE)x0;
                    if(fy < 0) fy = 0;
                    if(fx < 0) fx = 0;
                    ACCUMULATOR_TYPE w00 = ((ACCUMULATOR_TYPE)1 - fy) * ((ACCUMULATOR_TYPE)1 - fx);
                    ACCUMULATOR_TYPE w01 = ((ACCUMULATOR_TYPE)1 - fy) * fx;
                    ACCUMULATOR_TYPE w10 = fy * ((ACCUMULATOR_TYPE)1 - fx);
                    ACCUMULATOR_TYPE w11 = fy * fx;
                    for(TI c_i = 0; c_i < C; c_i++){
                        ACCUMULATOR_TYPE grad = (ACCUMULATOR_TYPE)get(device, d_output_4d, bi, oy, ox, c_i);
                        set(device, d_input_4d, (T)((ACCUMULATOR_TYPE)get(device, d_input_4d, bi, y0, x0, c_i) + w00 * grad), bi, y0, x0, c_i);
                        set(device, d_input_4d, (T)((ACCUMULATOR_TYPE)get(device, d_input_4d, bi, y0, x1, c_i) + w01 * grad), bi, y0, x1, c_i);
                        set(device, d_input_4d, (T)((ACCUMULATOR_TYPE)get(device, d_input_4d, bi, y1, x0, c_i) + w10 * grad), bi, y1, x0, c_i);
                        set(device, d_input_4d, (T)((ACCUMULATOR_TYPE)get(device, d_input_4d, bi, y1, x1, c_i) + w11 * grad), bi, y1, x1, c_i);
                    }
                }
            }
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::upsample2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        // No parameters to accumulate gradients for
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::upsample2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::upsample2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        backward_input(device, static_cast<const nn::layers::upsample2d::LayerBackward<LAYER_SPEC>&>(layer), d_output, d_input, buffer, mode);
    }

    // ======================== zero_gradient / update / _reset_optimizer_state (no-ops) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}

    // ======================== copy ========================
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::upsample2d::LayerForward<SOURCE_SPEC>& source, nn::layers::upsample2d::LayerForward<TARGET_SPEC>& target) {
        static_assert(nn::layers::upsample2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::upsample2d::LayerBackward<SOURCE_SPEC>& source, nn::layers::upsample2d::LayerBackward<TARGET_SPEC>& target) {
        static_assert(nn::layers::upsample2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::upsample2d::LayerGradient<SOURCE_SPEC>& source, nn::layers::upsample2d::LayerGradient<TARGET_SPEC>& target) {
        static_assert(nn::layers::upsample2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::upsample2d::LayerForward<S1>&, const nn::layers::upsample2d::LayerForward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::upsample2d::LayerBackward<S1>&, const nn::layers::upsample2d::LayerBackward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::upsample2d::LayerGradient<S1>& l1, const nn::layers::upsample2d::LayerGradient<S2>& l2) { return abs_diff(device, l1.output, l2.output); }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::upsample2d::State&, const nn::layers::upsample2d::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::upsample2d::LayerBackward<SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& l) { set_all(device, l.output, 0); }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::upsample2d::LayerForward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::upsample2d::LayerBackward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::upsample2d::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) { return is_nan(device, l.output, mode); }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::upsample2d::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::upsample2d::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
