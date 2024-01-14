#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H

#include "model.h"
#include "../../utils/generic/typing.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename MODULE_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        malloc(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            malloc(device, module.next_module);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void free(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        free(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            free(device, module.next_module);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module, RNG& rng){
        using namespace nn_models::sequential;
        init_weights(device, module.content, rng);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            init_weights(device, module.next_module, rng);
        }
    }
    template <typename SPEC>
    constexpr typename SPEC::TI num_layers(nn_models::sequential::Module<SPEC>){
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            return num_layers(typename SPEC::NEXT_MODULE{}) + 1;
        }
        else{
            return 1;
        }
    }
    template <typename SPEC>
    constexpr auto& output(nn_models::sequential::Module<SPEC>& m){
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            return m.content.output;
        } else {
            return output(m.next_module);
        }
    }
    // Evaluate is like a forward pass but without saving intermediate activations (so a backward pass is not possible). Hence we can reuse the memory of the intermediate outputs and just require a double buffer where each buffer has to be able to contain the maximum hidden dimension of the module
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void evaluate(DEVICE& device, const nn_models::sequential::Module<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        static_assert(BUFFER_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            evaluate(device, model.content, input, output);
        }
        else{
            DOUBLE_BUFFER_TYPE& output_buffer = TICK ? buffers.tick : buffers.tock;
            auto output_buffer_view = view(device, output_buffer, matrix::ViewSpec<BATCH_SIZE, MODULE_SPEC::CONTENT::OUTPUT_DIM>{});
            evaluate(device, model.content, input, output_buffer_view);
            evaluate<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename decltype(output_buffer_view)::SPEC, OUTPUT_SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, output_buffer_view, output, buffers);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT>
    void forward(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module, INPUT& input){
        forward(device, module.content, input);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            forward(device, module.next_module, module.content.output);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT>
    void forward(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module, INPUT& input, OUTPUT& output){
        forward(device, module, input);
        copy(device, device, rl_tools::output(module), output);
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void zero_gradient(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& module){
        zero_gradient(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            zero_gradient(device, module.next_module);
        }
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::sequential::Module<SPEC>& module, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, module.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            _reset_optimizer_state(device, module.next_module, optimizer);
        }
    }
    template<typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, nn_models::sequential::Module<SPEC>& module) {
        reset_forward_state(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            reset_forward_state(device, module.next_module);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void backward_full(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers) {
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward(device, model.content, input, d_output, d_input);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            backward_full<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename decltype(model.content.output)::SPEC, D_OUTPUT_SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, model.content.output, d_output, current_d_output_buffer, buffers);
            backward(device, model.content, input, current_d_output_buffer, d_input);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void backward_input(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& model, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers) {
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward_input(device, model.content, d_output, d_input);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            backward_input<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, D_OUTPUT_SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, d_output, current_d_output_buffer, buffers);
            backward_input(device, model.content, current_d_output_buffer, d_input);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC>
    void backward(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers) {
        static_assert(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>);
        backward_full<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename decltype(model.content.output)::SPEC, D_OUTPUT_SPEC, typename decltype(buffers.tick)::SPEC, BUFFER_SPEC, false>(device, model.next_module, model.content.output, d_output, buffers.tick, buffers);
        backward_param(device, model.content, input, buffers.tick);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn_models::sequential::Module<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            update(device, model.next_module, optimizer);
        }
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::sequential::Module<SOURCE_SPEC>& source, nn_models::sequential::Module<TARGET_SPEC>& target){
        copy(source_device, target_device, source.content, target.content);
        if constexpr(!utils::typing::is_same_v<typename TARGET_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            copy(source_device, target_device, source.next_module, target.next_module);
        }
    }

    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    typename SPEC_A::T abs_diff(DEVICE& device, nn_models::sequential::Module<SPEC_A>& a, const nn_models::sequential::Module<SPEC_B>& b){
        auto diff = abs_diff(device, a.content, b.content);
        if constexpr(!utils::typing::is_same_v<typename SPEC_A::NEXT_MODULE, nn_models::sequential::OutputModule>){
            diff += abs_diff(device, a.next_module, b.next_module);
        }
        return diff;
    }

    template<typename DEVICE, typename MODULE_SPEC, auto LAYER_I>
    constexpr auto& get_layer(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& model, Constant<LAYER_I>){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < num_layers(nn_models::sequential::Module<MODULE_SPEC>{}));
        if constexpr(LAYER_I == 0){
            return model.content;
        }
        else{
            return get_layer(device, model.next_module, Constant<LAYER_I - 1>{});
        }
    }

    template<typename DEVICE, typename MODULE_SPEC>
    bool is_nan(DEVICE& device, nn_models::sequential::Module<MODULE_SPEC>& model){
        bool current_module_nan = is_nan(device, model.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            current_module_nan = current_module_nan || is_nan(device, model.next_module);
        }
        return current_module_nan;
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
