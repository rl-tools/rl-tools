#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_AVG_POOL2D_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_AVG_POOL2D_OPERATIONS_GENERIC_H
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::avg_pool2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::avg_pool2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::avg_pool2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::avg_pool2d::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& layer) {
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& layer) {
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::avg_pool2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::avg_pool2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::avg_pool2d::Buffer&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::avg_pool2d::Buffer&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::avg_pool2d::State&, nn::layers::avg_pool2d::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::avg_pool2d::LayerForward<SPEC>&, nn::layers::avg_pool2d::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights (no-op) ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::avg_pool2d::LayerForward<SPEC>& layer, RNG& rng) {}

    // ======================== evaluate ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::avg_pool2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;
        constexpr TI BATCH_SIZE = product(typename INPUT_SPEC::SHAPE{}) / (IH * IW * C);
        using INTERNAL_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, IH, IW, C>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, C>;
        auto input_4d = view_memory<INTERNAL_INPUT_SHAPE>(device, input);
        auto output_2d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);
        constexpr T scale = (T)1 / (T)(IH * IW);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI c_i = 0; c_i < C; c_i++){
                T acc = 0;
                for(TI h = 0; h < IH; h++){
                    for(TI w = 0; w < IW; w++){
                        acc += get(device, input_4d, bi, h, w, c_i);
                    }
                }
                set(device, output_2d, acc * scale, bi, c_i);
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::avg_pool2d::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::avg_pool2d::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::avg_pool2d::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward ========================
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::avg_pool2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::avg_pool2d::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename D_OUTPUT_SPEC::T;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI IH = LAYER_SPEC::INPUT_HEIGHT;
        constexpr TI IW = LAYER_SPEC::INPUT_WIDTH;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, C>;
        using INTERNAL_D_INPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, IH, IW, C>;
        auto d_output_2d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);
        auto d_input_4d = view_memory<INTERNAL_D_INPUT_SHAPE>(device, d_input);
        constexpr T scale = (T)1 / (T)(IH * IW);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI c_i = 0; c_i < C; c_i++){
                T grad = get(device, d_output_2d, bi, c_i) * scale;
                for(TI h = 0; h < IH; h++){
                    for(TI w = 0; w < IW; w++){
                        set(device, d_input_4d, grad, bi, h, w, c_i);
                    }
                }
            }
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::avg_pool2d::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        // No parameters to accumulate gradients for
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::avg_pool2d::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        backward_input(device, static_cast<const nn::layers::avg_pool2d::LayerBackward<LAYER_SPEC>&>(layer), d_output, d_input, buffer, mode);
    }

    // ======================== zero_gradient / update / _reset_optimizer_state (no-ops) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}

    // ======================== copy ========================
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::avg_pool2d::LayerForward<SOURCE_SPEC>& source, nn::layers::avg_pool2d::LayerForward<TARGET_SPEC>& target) {
        static_assert(nn::layers::avg_pool2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::avg_pool2d::LayerBackward<SOURCE_SPEC>& source, nn::layers::avg_pool2d::LayerBackward<TARGET_SPEC>& target) {
        static_assert(nn::layers::avg_pool2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::avg_pool2d::LayerGradient<SOURCE_SPEC>& source, nn::layers::avg_pool2d::LayerGradient<TARGET_SPEC>& target) {
        static_assert(nn::layers::avg_pool2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::avg_pool2d::LayerForward<S1>&, const nn::layers::avg_pool2d::LayerForward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::avg_pool2d::LayerBackward<S1>&, const nn::layers::avg_pool2d::LayerBackward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::avg_pool2d::LayerGradient<S1>& l1, const nn::layers::avg_pool2d::LayerGradient<S2>& l2) { return abs_diff(device, l1.output, l2.output); }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::avg_pool2d::State&, const nn::layers::avg_pool2d::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::avg_pool2d::LayerBackward<SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& l) { set_all(device, l.output, 0); }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::avg_pool2d::LayerForward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::avg_pool2d::LayerBackward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::avg_pool2d::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) { return is_nan(device, l.output, mode); }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::avg_pool2d::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::avg_pool2d::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
