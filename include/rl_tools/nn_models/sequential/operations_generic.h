#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H

#include "model.h"
#include "../../utils/generic/typing.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename MODULE_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        malloc(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            malloc(device, module.next_module);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        free(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            free(device, module.next_module);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& content_buffer){
        using namespace nn_models::sequential;
        malloc(device, content_buffer.buffer);
        if constexpr(!utils::typing::is_same_v<typename BUFFER_SPEC::NEXT_SPEC, OutputModule>){
            malloc(device, content_buffer.next_content_buffer);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& content_buffer){
        using namespace nn_models::sequential;
        free(device, content_buffer.content_buffer);
        if constexpr(!utils::typing::is_same_v<typename BUFFER_SPEC::NEXT_SPEC, OutputModule>){
            free(device, content_buffer.next_content_buffer);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
        malloc(device, buffers.content_buffer);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module, RNG& rng){
        using namespace nn_models::sequential;
        init_weights(device, module.content, rng);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            init_weights(device, module.next_module, rng);
        }
    }
    namespace nn_models::sequential{
        template <typename SPEC>
        constexpr typename SPEC::TI num_layers(){
            if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
                return num_layers<typename SPEC::NEXT_MODULE::SPEC>() + 1;
            }
            else{
                return 1;
            }
        }
    }
    template <typename SPEC>
    constexpr typename SPEC::TI num_layers(nn_models::sequential::ModuleForward<SPEC>){
        return nn_models::sequential::num_layers<SPEC>();
    }

    template<auto LAYER_I, typename DEVICE, typename MODULE_SPEC>
    constexpr auto& get_layer(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < nn_models::sequential::num_layers<MODULE_SPEC>());
        if constexpr(LAYER_I == 0){
            return model.content;
        }
        else{
            return get_layer<LAYER_I - 1>(device, model.next_module);
        }
    }
    template<auto LAYER_I, typename DEVICE, typename MODULE_SPEC>
    constexpr auto& get_layer(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < nn_models::sequential::num_layers<MODULE_SPEC>());
        if constexpr(LAYER_I == 0){
            return model.content;
        }
        else{
            return get_layer<LAYER_I - 1>(device, model.next_module);
        }
    }
    template <typename SPEC>
    constexpr auto& output(nn_models::sequential::ModuleGradient<SPEC>& m){
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            return output(m.content);
        } else {
            return output(m.next_module);
        }
    }
    // Evaluate is like a forward pass but without saving intermediate activations (so a backward pass is not possible). Hence we can reuse the memory of the intermediate outputs and just require a double buffer where each buffer has to be able to contain the maximum hidden dimension of the module
    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename RNG>
    void _evaluate(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer, RNG& rng){
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        static_assert(BUFFER_SPEC::BATCH_SIZE >= OUTPUT_SPEC::ROWS);
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            evaluate(device, model.content, input, output, content_buffer.buffer, rng);
        }
        else{
            DOUBLE_BUFFER_TYPE& output_buffer = TICK ? buffers.tick : buffers.tock;
            auto output_buffer_view = view(device, output_buffer, matrix::ViewSpec<BATCH_SIZE, MODULE_SPEC::CONTENT::OUTPUT_DIM>{});
            evaluate(device, model.content, input, output_buffer_view, content_buffer.buffer, rng);
            _evaluate<!TICK>(device, model.next_module, output_buffer_view, output, buffers, content_buffer.next_content_buffer, rng);
        }
    }
    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG>
    void evaluate(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng){
        _evaluate<TICK>(device, model, input, output, buffers, buffers.content_buffer, rng);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename RNG>
    void forward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module, INPUT& input, RNG& rng){
        forward(device, module.content, input, rng);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            forward(device, module.next_module, module.content.output, rng);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename RNG>
    void forward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module, INPUT& input, OUTPUT& output, RNG& rng){
        forward(device, module, input, rng);
        copy(device, device, rl_tools::output(module), output);
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void zero_gradient(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module){
        zero_gradient(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            zero_gradient(device, module.next_module);
        }
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, module.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            _reset_optimizer_state(device, module.next_module, optimizer);
        }
    }
    template<typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& module) {
        reset_forward_state(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            reset_forward_state(device, module.next_module);
        }
    }
    // the _xxx are unrolling the content_buffers (which should not be exposed to the user)
    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC>
    void _backward_full(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer) {
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using TI = typename DEVICE::index_t;
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward_full(device, model.content, input, d_output, d_input, content_buffer.buffer);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            auto current_d_output_buffer_view = view(device, current_d_output_buffer, matrix::ViewSpec<BATCH_SIZE, MODULE_SPEC::CONTENT::OUTPUT_DIM>{});
            _backward_full<!TICK>(device, model.next_module, model.content.output, d_output, current_d_output_buffer_view, buffers, content_buffer.next_content_buffer);
            backward_full(device, model.content, input, current_d_output_buffer, d_input, content_buffer.buffer);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC>
    void backward_full(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers) {
        _backward_full(device, model, input, d_output, d_input, buffers, buffers.content_buffer);
    }
    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC>
    void _backward_input(DEVICE& device, nn_models::sequential::ModuleBackward<MODULE_SPEC>& model, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer){
        static_assert(nn_models::sequential::check_input_output<MODULE_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using TI = typename DEVICE::index_t;
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward_input(device, model.content, d_output, d_input, content_buffer.buffer);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            auto current_d_output_buffer_view = view(device, current_d_output_buffer, matrix::ViewSpec<BATCH_SIZE, MODULE_SPEC::CONTENT::OUTPUT_DIM>{});
            _backward_input<!TICK>(device, model.next_module, d_output, current_d_output_buffer_view, buffers, content_buffer.next_content_buffer);
            backward_input(device, model.content, current_d_output_buffer, d_input, content_buffer.buffer);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC>
    void backward_input(DEVICE& device, nn_models::sequential::ModuleBackward<MODULE_SPEC>& model, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers){
        _backward_input(device, model, d_output, d_input, buffers, buffers.content_buffer);
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC>
    void backward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers) {
        constexpr bool NEXT_IS_FINAL = utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        // This backward function is called on the final, complete module, the following are called for each submodule, hence the full backward only for the next module (to save the calc for d_input)
        if constexpr(!NEXT_IS_FINAL){
            auto current_d_input_buffer_view = view(device, buffers.tick, matrix::ViewSpec<BATCH_SIZE, MODULE_SPEC::CONTENT::OUTPUT_DIM>{});
            _backward_full<false>(device, model.next_module, model.content.output, d_output, current_d_input_buffer_view, buffers, buffers.content_buffer.next_content_buffer);
            backward(device, model.content, input, current_d_input_buffer_view, buffers.content_buffer.buffer);
        }
        else{
            backward(device, model.content, input, d_output, buffers.content_buffer.buffer);
        }
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            update(device, model.next_module, optimizer);
        }
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::sequential::ModuleForward<SOURCE_SPEC>& source, nn_models::sequential::ModuleForward<TARGET_SPEC>& target){
        copy(source_device, target_device, source.content, target.content);
        if constexpr(!utils::typing::is_same_v<typename TARGET_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            copy(source_device, target_device, source.next_module, target.next_module);
        }
    }

    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    typename SPEC_A::T abs_diff(DEVICE& device, nn_models::sequential::ModuleForward<SPEC_A>& a, const nn_models::sequential::ModuleForward<SPEC_B>& b){
        auto diff = abs_diff(device, a.content, b.content);
        if constexpr(!utils::typing::is_same_v<typename SPEC_A::NEXT_MODULE, nn_models::sequential::OutputModule>){
            diff += abs_diff(device, a.next_module, b.next_module);
        }
        return diff;
    }


    template<typename DEVICE, typename MODULE_SPEC>
    bool is_nan(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        bool current_module_nan = is_nan(device, model.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            current_module_nan = current_module_nan || is_nan(device, model.next_module);
        }
        return current_module_nan;
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
