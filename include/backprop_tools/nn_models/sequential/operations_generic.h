#ifndef BACKPROP_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H

#include "model.h"
namespace backprop_tools{
    template <typename DEVICE, typename MODULE_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        malloc(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            malloc(device, module.next_module);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module){
        using namespace nn_models::sequential;
        free(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            free(device, module.next_module, module.content.output);
        }
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
    }
    template <typename DEVICE, typename BUFFER_SPEC>
    void free(DEVICE& device, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
        using namespace nn_models::sequential;
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template <typename DEVICE, typename MODULE_SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module, RNG& rng){
        using namespace nn_models::sequential;
        init_weights(device, module.content, rng);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, OutputModule>){
            init_weights(device, module.next_module, rng);
        }
    }
    // Evaluate is like a forward pass but without saving intermediate activations (so a backward pass is not possible). Hence we can reuse the memory of the intermediate outputs and just require a double buffer where each buffer has to be able to contain the maximum hidden dimension of the module
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void evaluate(DEVICE& device, const nn_models::sequential::ModuleInternal<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC>& buffers){
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
            evaluate(device, model.content, input, output_buffer);
            evaluate<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, OUTPUT_SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, output_buffer, output, buffers);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT>
    void forward(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module, INPUT& input, OUTPUT& output){
        forward(device, module.content, input);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            forward(device, module.next_module, module.content.output, output);
        }
        else{
            copy(device, device, output, module.content.output);
        }
    }
    template <typename DEVICE, typename MODULE_SPEC>
    void zero_gradient(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& module){
        zero_gradient(device, module.content);
        if constexpr(!utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            zero_gradient(device, module.next_module);
        }
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::sequential::ModuleInternal<SPEC>& module, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, module.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            _reset_optimizer_state(device, module.next_module, optimizer);
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, bool TICK = true>
    void backward(DEVICE& device, nn_models::sequential::ModuleInternal<MODULE_SPEC>& model, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::sequential::ModuleDoubleBuffer<BUFFER_SPEC> buffers) {
        static_assert(nn_models::sequential::buffer_compatible<BUFFER_SPEC, MODULE_SPEC>);
        using DOUBLE_BUFFER_TYPE = decltype(buffers.tick);

        if constexpr(utils::typing::is_same_v<typename MODULE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            backward(device, model.content, input, d_output, d_input);
        }
        else{
            DOUBLE_BUFFER_TYPE& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
            backward<DEVICE, typename MODULE_SPEC::NEXT_MODULE::SPEC, typename decltype(model.content.output)::SPEC, D_OUTPUT_SPEC, typename DOUBLE_BUFFER_TYPE::SPEC, BUFFER_SPEC, !TICK>(device, model.next_module, model.content.output, d_output, current_d_output_buffer, buffers);
            backward(device, model.content, input, current_d_output_buffer, d_input);
        }

    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn_models::sequential::ModuleInternal<SPEC>& model, OPTIMIZER& optimizer) {
        update(device, model.content, optimizer);
        if constexpr(!utils::typing::is_same_v<typename SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
            update(device, model.next_module, optimizer);
        }
    }
}

#endif
