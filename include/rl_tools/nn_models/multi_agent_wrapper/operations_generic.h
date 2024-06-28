#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_OPERATIONS_GENERIC_H

#include "model.h"
#include "../../utils/generic/typing.h"
#include "../../nn/mode.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename MODULE_SPEC>
    void malloc(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& module){
        malloc(device, module.content);
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void free(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& module){
        free(device, module.content);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffer){
        malloc(device, buffer.input);
        malloc(device, buffer.d_input);
        malloc(device, buffer.output);
        malloc(device, buffer.buffer);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffer){
        free(device, buffer.input);
        free(device, buffer.d_input);
        free(device, buffer.output);
        free(device, buffer.buffer);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_BUFFER_SPEC, typename TARGET_BUFFER_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn_models::multi_agent_wrapper::ModuleBuffer<SOURCE_BUFFER_SPEC>& source, nn_models::multi_agent_wrapper::ModuleBuffer<TARGET_BUFFER_SPEC>& target){
        copy(source_device, target_device, source.buffer, target.buffer);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& module, RNG& rng){
        init_weights(device, module.content, rng);
    }

    template <typename MODULE_SPEC> // non-const
    constexpr auto& get_first_layer(nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model){
        return get_first_layer(model.content);
    }
    template <typename MODULE_SPEC> // const
    constexpr auto& get_first_layer(const nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model){
        return get_first_layer(model.content);
    }

    template <typename MODULE_SPEC> // non-const
    constexpr auto& get_last_layer(nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model){
        return get_last_layer(model.content);
    }
    template <typename MODULE_SPEC> // const
    constexpr auto& get_last_layer(const nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model){
        return get_last_layer(model.content);
    }

    template <typename BUFFER_SPEC> // non-const
    constexpr auto& get_last_buffer(nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffer){
        return get_last_buffer(buffer.buffer);
    }
    template <typename BUFFER_SPEC> // const
    constexpr auto& get_last_buffer(const nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffer){
        return get_last_buffer(buffer.buffer);
    }

    template <typename SPEC> // non-const
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& output(nn_models::multi_agent_wrapper::ModuleGradient<SPEC>& m){
        return output(m.content);
    }
    template <typename SPEC> // const
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& output(const nn_models::multi_agent_wrapper::ModuleGradient<SPEC>& m){
        return output(m.content);
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void evaluate(DEVICE& device, const nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        copy(device, device, input, buffers.input);
        auto input_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, INPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.input);
        auto output_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, OUTPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.output);
        evaluate(device, model.content, input_reshaped, output_reshaped, buffers.buffer, rng, mode);
        copy(device, device, buffers.output, output);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<MODULE_SPEC>& module, INPUT& input, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        using TI = typename DEVICE::index_t;
        using INPUT_SPEC = typename INPUT::SPEC;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        copy(device, device, input, buffers.input);
        auto input_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, INPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.input);
        forward(device, module.content, input_reshaped, buffers.buffer, rng, mode);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<MODULE_SPEC>& module, INPUT& input, OUTPUT& output, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        using TI = typename DEVICE::index_t;
        using INPUT_SPEC = typename INPUT::SPEC;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        forward(device, module, input, buffers, rng, mode);
        auto module_output = rl_tools::output(module);
        auto output_reshaped = reshape<BATCH_SIZE, decltype(module_output)::SPEC::COLS*MODULE_SPEC::N_AGENTS>(device, module_output);
        copy(device, device, output_reshaped, output);
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void zero_gradient(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<MODULE_SPEC>& module){
        zero_gradient(device, module.content);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, module.content, optimizer);
    }
    template<typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<SPEC>& module) {
        reset_forward_state(device, module.content);
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward_full(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC> buffers, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        copy(device, device, input, buffers.input);
        copy(device, device, d_output, buffers.output);
        auto input_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, INPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.input);
        auto d_output_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, D_OUTPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.output);
        backward_full(device, model.content, input_reshaped, d_output_reshaped, buffers.d_inpu, buffers.buffer, mode);
        copy(device, device, buffers.d_input, d_input);
    }
    template<typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward_input(DEVICE& device, nn_models::multi_agent_wrapper::ModuleBackward<MODULE_SPEC>& model, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC> buffers, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_INPUT_SPEC::ROWS;
        copy(device, device, d_output, buffers.output);
        auto d_output_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, D_OUTPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.output);
        backward_input(device, model.content, d_output_reshaped, buffers.d_input, buffers.buffer, mode);
        copy(device, device, buffers.d_input, d_input);
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC> buffers, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        copy(device, device, input, buffers.input);
        copy(device, device, d_output, buffers.output);
        auto input_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, INPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.input);
        auto d_output_reshaped = reshape<BATCH_SIZE*MODULE_SPEC::N_AGENTS, D_OUTPUT_SPEC::COLS/MODULE_SPEC::N_AGENTS>(device, buffers.output);
        backward(device, model.content, input_reshaped, d_output_reshaped, buffers.buffer, mode);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn_models::multi_agent_wrapper::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.content, optimizer);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::multi_agent_wrapper::ModuleForward<SOURCE_SPEC>& source, nn_models::multi_agent_wrapper::ModuleForward<TARGET_SPEC>& target){
        copy(source_device, target_device, source.content, target.content);
    }

    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    typename SPEC_A::T abs_diff(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<SPEC_A>& a, const nn_models::multi_agent_wrapper::ModuleForward<SPEC_B>& b){
        auto diff = abs_diff(device, a.content, b.content);
        return diff;
    }


    template<typename DEVICE, typename MODULE_SPEC>
    bool is_nan(DEVICE& device, nn_models::multi_agent_wrapper::ModuleForward<MODULE_SPEC>& model){
        bool current_module_nan = is_nan(device, model.content);
        return current_module_nan;
    }

    template <typename DEVICE, typename BUFFER_SPEC, typename RNG>
    void sample(DEVICE& device, nn_models::multi_agent_wrapper::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng){
        sample(device, buffer.buffer, rng);
    }
    template <typename DEVICE, typename SPEC>
    void print(DEVICE& device, const nn_models::multi_agent_wrapper::ModuleForward<SPEC>& model, typename DEVICE::index_t layer_i = 0){
        print(device, model.content, layer_i);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
