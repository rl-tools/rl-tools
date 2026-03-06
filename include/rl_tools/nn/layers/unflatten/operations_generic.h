#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_UNFLATTEN_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_UNFLATTEN_OPERATIONS_GENERIC_H
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::unflatten::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::unflatten::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::unflatten::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::unflatten::LayerBackward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& layer) {
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& layer) {
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::unflatten::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::unflatten::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::unflatten::Buffer&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::unflatten::Buffer&) {}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::unflatten::State&, nn::layers::unflatten::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::unflatten::LayerForward<SPEC>&, nn::layers::unflatten::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights (no-op) ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::unflatten::LayerForward<SPEC>& layer, RNG& rng) {}

    // ======================== evaluate ========================
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::unflatten::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::unflatten::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::unflatten::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(tensor::dense_row_major_layout<INPUT_SPEC>(), "Unflatten requires contiguous row-major input");
        static_assert(tensor::dense_row_major_layout<OUTPUT_SPEC>(), "Unflatten requires contiguous row-major output");
        using TI = typename DEVICE::index_t;
        constexpr TI TOTAL = product(typename INPUT_SPEC::SHAPE{});
        auto* src = data(input);
        auto* dst = data(output);
        for(TI i = 0; i < TOTAL; i++){
            dst[i] = src[i];
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn::layers::unflatten::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::unflatten::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::unflatten::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::unflatten::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::unflatten::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        evaluate(device, static_cast<const nn::layers::unflatten::LayerForward<LAYER_SPEC>&>(layer), input, output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::unflatten::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::unflatten::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::unflatten::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::unflatten::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::unflatten::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward ========================
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::unflatten::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::unflatten::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::unflatten::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(tensor::dense_row_major_layout<D_OUTPUT_SPEC>(), "Unflatten requires contiguous row-major d_output");
        static_assert(tensor::dense_row_major_layout<D_INPUT_SPEC>(), "Unflatten requires contiguous row-major d_input");
        using TI = typename DEVICE::index_t;
        constexpr TI TOTAL = product(typename D_INPUT_SPEC::SHAPE{});
        auto* src = data(d_output);
        auto* dst = data(d_input);
        for(TI i = 0; i < TOTAL; i++){
            dst[i] = src[i];
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::unflatten::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::unflatten::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::unflatten::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::unflatten::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        backward_input(device, static_cast<const nn::layers::unflatten::LayerBackward<LAYER_SPEC>&>(layer), d_output, d_input, buffer, mode);
    }

    // ======================== zero_gradient / update / _reset_optimizer_state (no-ops) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}

    // ======================== copy ========================
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::unflatten::LayerForward<SOURCE_SPEC>& source, nn::layers::unflatten::LayerForward<TARGET_SPEC>& target) {
        static_assert(nn::layers::unflatten::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::unflatten::LayerBackward<SOURCE_SPEC>& source, nn::layers::unflatten::LayerBackward<TARGET_SPEC>& target) {
        static_assert(nn::layers::unflatten::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::unflatten::LayerGradient<SOURCE_SPEC>& source, nn::layers::unflatten::LayerGradient<TARGET_SPEC>& target) {
        static_assert(nn::layers::unflatten::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::unflatten::LayerForward<S1>&, const nn::layers::unflatten::LayerForward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::unflatten::LayerBackward<S1>&, const nn::layers::unflatten::LayerBackward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::unflatten::LayerGradient<S1>& l1, const nn::layers::unflatten::LayerGradient<S2>& l2) { return abs_diff(device, l1.output, l2.output); }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::unflatten::State&, const nn::layers::unflatten::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::unflatten::LayerBackward<SPEC>&) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& l) { set_all(device, l.output, 0); }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::unflatten::LayerForward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::unflatten::LayerBackward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::unflatten::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) { return is_nan(device, l.output, mode); }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::unflatten::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::unflatten::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
