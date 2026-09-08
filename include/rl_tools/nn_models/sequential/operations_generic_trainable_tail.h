#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_TRAINABLE_TAIL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_TRAINABLE_TAIL_H

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<auto FIRST, typename DEVICE, typename MODEL_SPEC, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_trainable_tail(DEVICE& device, nn_models::sequential::ModuleGradient<MODEL_SPEC>& model, D_OUTPUT& d_output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(FIRST > 0 && FIRST + 1 < MODEL_SPEC::NUM_LAYERS);
        using nn_models::sequential::layer_output;
        using nn_models::sequential::backward_full_impl;
        using nn_models::sequential::layer;
        using nn_models::sequential::content_buffer;
        using OUTPUT_SHAPE = typename nn_models::sequential::layer_spec_t<FIRST, MODEL_SPEC>::OUTPUT_SHAPE;
        auto d_first_output = view_memory<OUTPUT_SHAPE>(device, buffer.tick);
        auto first_output = layer_output<FIRST>(device, model);
        backward_full_impl<false, FIRST + 1>(device, model, first_output, d_output, d_first_output, buffer, buffer.content_buffer, mode);
        auto first_input = layer_output<FIRST - 1>(device, model);
        backward(device, layer<FIRST>(model), first_input, d_first_output, content_buffer<FIRST>(buffer.content_buffer), mode);
    }
}

RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
