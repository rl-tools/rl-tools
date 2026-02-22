#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_OPERATIONS_GENERIC_H

#include "model.h"
#include "../../utils/generic/typing.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, Tensor<SPEC>& tensor);
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, const Tensor<SPEC>& tensor);
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, Matrix<SPEC>& matrix);
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, const Matrix<SPEC>& matrix);

    namespace nn_models::sequential {
        template <auto LAYER_I, typename SPEC>
        using layer_spec_t = typename tuple_element<LAYER_I, typename SPEC::LAYER_SPECS>::type;

        template <auto LAYER_I, typename MODULE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& layer(ModuleForward<MODULE_SPEC>& model) {
            return get<LAYER_I>(model.content);
        }
        template <auto LAYER_I, typename MODULE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& layer(const ModuleForward<MODULE_SPEC>& model) {
            return get<LAYER_I>(model.content);
        }
        template <auto LAYER_I, typename CONTENT_BUFFER_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& content_buffer(ContentBuffer<CONTENT_BUFFER_SPEC>& buffer) {
            return get<LAYER_I>(buffer.buffers);
        }
        template <auto LAYER_I, typename CONTENT_BUFFER_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& content_buffer(const ContentBuffer<CONTENT_BUFFER_SPEC>& buffer) {
            return get<LAYER_I>(buffer.buffers);
        }
        template <auto LAYER_I, typename CONTENT_STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& content_state(ContentState<CONTENT_STATE_SPEC>& state) {
            return get<LAYER_I>(state.states);
        }
        template <auto LAYER_I, typename CONTENT_STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& content_state(const ContentState<CONTENT_STATE_SPEC>& state) {
            return get<LAYER_I>(state.states);
        }

        template <auto LAYER_I = 0, typename TARGET_SPEC, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE>
        RL_TOOLS_FUNCTION_PLACEMENT void copy_from_generic_layers(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const SOURCE& source, ModuleForward<TARGET_SPEC>& target){
            if constexpr(LAYER_I < TARGET_SPEC::NUM_LAYERS){
                copy_from_generic(source_device, target_device, source.content, layer<LAYER_I>(target));
                if constexpr(LAYER_I + 1 < TARGET_SPEC::NUM_LAYERS){
                    copy_from_generic_layers<LAYER_I + 1>(source_device, target_device, source.next_module, target);
                }
            }
        }

        template <auto LAYER_I, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT auto layer_output(DEVICE& device, ModuleGradient<SPEC>& m){
            auto output_matrix = output(device, layer<LAYER_I>(m));
            static_assert(sizeof(output_matrix) <= sizeof(void*));
            return _content_output_helper<typename layer_spec_t<LAYER_I, SPEC>::OUTPUT_SHAPE>(device, output_matrix);
        }
        template <auto LAYER_I, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT auto layer_output(DEVICE& device, const ModuleGradient<SPEC>& m){
            auto output_matrix = output(device, layer<LAYER_I>(m));
            static_assert(sizeof(output_matrix) <= sizeof(void*));
            return _content_output_helper<typename layer_spec_t<LAYER_I, SPEC>::OUTPUT_SHAPE>(device, output_matrix);
        }

        template<bool TICK = true, auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_impl(DEVICE& device, const ModuleForward<MODULE_SPEC>& model, const INPUT& input, OUTPUT& output, ModuleBuffer<BUFFER_SPEC>& buffers, ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffers, RNG& rng, const Mode<MODE>& mode){
            constexpr auto LAST = MODULE_SPEC::NUM_LAYERS - 1;
            if constexpr(LAYER_I == LAST){
                evaluate(device, layer<LAYER_I>(model), input, output, content_buffer<LAYER_I>(content_buffers), rng, mode);
            }
            else{
                auto& output_buffer = TICK ? buffers.tick : buffers.tock;
                using LAYER_TYPE = utils::typing::remove_reference_t<decltype(layer<LAYER_I>(model))>;
                using OUTPUT_SHAPE = typename LAYER_TYPE::template OUTPUT_SHAPE_FACTORY<typename INPUT::SPEC::SHAPE>;
                auto output_buffer_view = view_memory<OUTPUT_SHAPE>(device, output_buffer);
                evaluate(device, layer<LAYER_I>(model), input, output_buffer_view, content_buffer<LAYER_I>(content_buffers), rng, mode);
                evaluate_impl<!TICK, LAYER_I + 1>(device, model, output_buffer_view, output, buffers, content_buffers, rng, mode);
            }
        }

        template<bool TICK = true, auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename STATE_SPEC, typename CONTENT_STATE_SPEC, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_impl(DEVICE& device, const ModuleForward<MODULE_SPEC>& model, const INPUT& input, ModuleState<STATE_SPEC>& state, ContentState<CONTENT_STATE_SPEC>& content_state_container, OUTPUT& output, ModuleBuffer<BUFFER_SPEC>& buffers, ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffers, RNG& rng, const Mode<MODE>& mode){
            constexpr auto LAST = MODULE_SPEC::NUM_LAYERS - 1;
            if constexpr(LAYER_I == LAST){
                evaluate_step(device, layer<LAYER_I>(model), input, content_state<LAYER_I>(content_state_container), output, content_buffer<LAYER_I>(content_buffers), rng, mode);
            }
            else{
                auto& output_buffer = TICK ? buffers.tick : buffers.tock;
                using LAYER_TYPE = utils::typing::remove_reference_t<decltype(layer<LAYER_I>(model))>;
                using OUTPUT_SHAPE = typename LAYER_TYPE::template OUTPUT_SHAPE_FACTORY<typename INPUT::SPEC::SHAPE>;
                auto output_buffer_view = view_memory<OUTPUT_SHAPE>(device, output_buffer);
                evaluate_step(device, layer<LAYER_I>(model), input, content_state<LAYER_I>(content_state_container), output_buffer_view, content_buffer<LAYER_I>(content_buffers), rng, mode);
                evaluate_step_impl<!TICK, LAYER_I + 1>(device, model, output_buffer_view, state, content_state_container, output, buffers, content_buffers, rng, mode);
            }
        }

        template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename BUFFER_SPEC, typename RNG, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void forward_impl(DEVICE& device, ModuleGradient<MODULE_SPEC>& module, INPUT& input, ContentBuffer<BUFFER_SPEC>& buffers, RNG& rng, const Mode<MODE>& mode){
            forward(device, layer<LAYER_I>(module), input, content_buffer<LAYER_I>(buffers), rng, mode);
            if constexpr(LAYER_I + 1 < MODULE_SPEC::NUM_LAYERS){
                auto output = layer_output<LAYER_I>(device, module);
                forward_impl<LAYER_I + 1>(device, module, output, buffers, rng, mode);
            }
        }

        template<bool TICK = true, auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void backward_full_impl(DEVICE& device, ModuleGradient<MODULE_SPEC>& model, const INPUT& input, D_OUTPUT& d_output, D_INPUT& d_input, ModuleBuffer<BUFFER_SPEC>& buffers, ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffers, const Mode<MODE>& mode){
            constexpr auto LAST = MODULE_SPEC::NUM_LAYERS - 1;
            if constexpr(LAYER_I == LAST){
                backward_full(device, layer<LAYER_I>(model), input, d_output, d_input, content_buffer<LAYER_I>(content_buffers), mode);
            }
            else{
                auto& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
                using OUTPUT_SHAPE = typename layer_spec_t<LAYER_I, MODULE_SPEC>::OUTPUT_SHAPE;
                auto current_d_output_buffer_view = view_memory<OUTPUT_SHAPE>(device, current_d_output_buffer);
                auto current_output = output(device, layer<LAYER_I>(model));
                auto current_output_tensor = to_tensor(device, current_output);
                backward_full_impl<!TICK, LAYER_I + 1>(device, model, current_output_tensor, d_output, current_d_output_buffer_view, buffers, content_buffers, mode);
                backward_full(device, layer<LAYER_I>(model), input, current_d_output_buffer_view, d_input, content_buffer<LAYER_I>(content_buffers), mode);
            }
        }

        template<bool TICK = true, auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename MODE>
        RL_TOOLS_FUNCTION_PLACEMENT void backward_input_impl(DEVICE& device, ModuleBackward<MODULE_SPEC>& model, D_OUTPUT& d_output, D_INPUT& d_input, ModuleBuffer<BUFFER_SPEC>& buffers, ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffers, const Mode<MODE>& mode){
            constexpr auto LAST = MODULE_SPEC::NUM_LAYERS - 1;
            if constexpr(LAYER_I == LAST){
                backward_input(device, layer<LAYER_I>(model), d_output, d_input, content_buffer<LAYER_I>(content_buffers), mode);
            }
            else{
                auto& current_d_output_buffer = TICK ? buffers.tick : buffers.tock;
                using OUTPUT_SHAPE = typename layer_spec_t<LAYER_I, MODULE_SPEC>::OUTPUT_SHAPE;
                auto current_d_output_buffer_view = view_memory<OUTPUT_SHAPE>(device, current_d_output_buffer);
                backward_input_impl<!TICK, LAYER_I + 1>(device, model, d_output, current_d_output_buffer_view, buffers, content_buffers, mode);
                backward_input(device, layer<LAYER_I>(model), current_d_output_buffer_view, d_input, content_buffer<LAYER_I>(content_buffers), mode);
            }
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            malloc(device, nn_models::sequential::layer<LAYER_I>(module));
            malloc<LAYER_I + 1>(device, module);
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            free(device, nn_models::sequential::layer<LAYER_I>(module));
            free<LAYER_I + 1>(device, module);
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::sequential::ContentState<STATE_SPEC>& state){
        if constexpr(LAYER_I < STATE_SPEC::SPEC::NUM_LAYERS){
            malloc(device, nn_models::sequential::content_state<LAYER_I>(state));
            malloc<LAYER_I + 1>(device, state);
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename STATE_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, nn_models::sequential::ContentState<STATE_SPEC>& state, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            reset(device, nn_models::sequential::layer<LAYER_I>(model), nn_models::sequential::content_state<LAYER_I>(state), rng, mode);
            reset<LAYER_I + 1>(device, model, state, rng, mode);
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::sequential::ContentState<STATE_SPEC>& state){
        if constexpr(LAYER_I < STATE_SPEC::SPEC::NUM_LAYERS){
            free(device, nn_models::sequential::content_state<LAYER_I>(state));
            free<LAYER_I + 1>(device, state);
        }
    }

    template <auto LAYER_I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_STATE_SPEC, typename TARGET_STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn_models::sequential::ContentState<SOURCE_STATE_SPEC>& source, nn_models::sequential::ContentState<TARGET_STATE_SPEC>& target){
        if constexpr(LAYER_I < SOURCE_STATE_SPEC::SPEC::NUM_LAYERS){
            copy(source_device, target_device, get<LAYER_I>(source.states), get<LAYER_I>(target.states));
            copy<LAYER_I + 1>(source_device, target_device, source, target);
        }
    }

    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::sequential::ModuleState<STATE_SPEC>& state){
        malloc(device, state.content_state);
    }

    template <typename DEVICE, typename MODULE_SPEC, typename STATE_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, nn_models::sequential::ModuleState<STATE_SPEC>& state, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        reset(device, model, state.content_state, rng, mode);
    }

    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn_models::sequential::ModuleState<SOURCE_SPEC>& source, nn_models::sequential::ModuleState<TARGET_SPEC>& target){
        copy(source_device, target_device, source.content_state, target.content_state);
    }

    template <typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::sequential::ModuleState<STATE_SPEC>& state){
        free(device, state.content_state);
    }

    template <auto LAYER_I = 0, typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& buffer){
        if constexpr(LAYER_I < BUFFER_SPEC::SPEC::NUM_LAYERS){
            malloc(device, nn_models::sequential::content_buffer<LAYER_I>(buffer));
            malloc<LAYER_I + 1>(device, buffer);
        }
    }

    template <auto LAYER_I = 0, typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& buffer){
        if constexpr(LAYER_I < BUFFER_SPEC::SPEC::NUM_LAYERS){
            free(device, nn_models::sequential::content_buffer<LAYER_I>(buffer));
            free<LAYER_I + 1>(device, buffer);
        }
    }

    template <auto LAYER_I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_BUFFER_SPEC, typename TARGET_BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn_models::sequential::ContentBuffer<SOURCE_BUFFER_SPEC>& source, nn_models::sequential::ContentBuffer<TARGET_BUFFER_SPEC>& target){
        if constexpr(LAYER_I < SOURCE_BUFFER_SPEC::SPEC::NUM_LAYERS){
            copy(source_device, target_device, get<LAYER_I>(source.buffers), get<LAYER_I>(target.buffers));
            copy<LAYER_I + 1>(source_device, target_device, source, target);
        }
    }

    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
        malloc(device, buffers.content_buffer);
    }

    template <typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers){
        free(device, buffers.tick);
        free(device, buffers.tock);
        free(device, buffers.content_buffer);
    }

    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_BUFFER_SPEC, typename TARGET_BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn_models::sequential::ModuleBuffer<SOURCE_BUFFER_SPEC>& source, nn_models::sequential::ModuleBuffer<TARGET_BUFFER_SPEC>& target){
        copy(source_device, target_device, source.tick, target.tick);
        copy(source_device, target_device, source.tock, target.tock);
        copy(source_device, target_device, source.content_buffer, target.content_buffer);
    }

    template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& module, RNG& rng){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            init_weights(device, nn_models::sequential::layer<LAYER_I>(module), rng);
            init_weights<LAYER_I + 1>(device, module, rng);
        }
    }

    namespace nn_models::sequential{
        template <typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI num_layers(){
            return SPEC::NUM_LAYERS;
        }
    }

    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI num_layers(const nn_models::sequential::ModuleForward<SPEC>&){
        return nn_models::sequential::num_layers<SPEC>();
    }
    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI num_layers(const nn_models::sequential::ContentBuffer<SPEC>&){
        return SPEC::SPEC::NUM_LAYERS;
    }
    template <typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr typename SPEC::TI num_layers(const nn_models::sequential::ModuleBuffer<SPEC>&){
        return SPEC::SPEC::NUM_LAYERS;
    }

    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_layer(nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < MODULE_SPEC::NUM_LAYERS);
        return nn_models::sequential::layer<LAYER_I>(model);
    }
    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_layer(const nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < MODULE_SPEC::NUM_LAYERS);
        return nn_models::sequential::layer<LAYER_I>(model);
    }

    template <typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_first_layer(nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        return get_layer<0>(model);
    }
    template <typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_first_layer(const nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        return get_layer<0>(model);
    }
    template <typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_last_layer(nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        return get_layer<MODULE_SPEC::NUM_LAYERS-1>(model);
    }
    template <typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_last_layer(const nn_models::sequential::ModuleForward<MODULE_SPEC>& model){
        return get_layer<MODULE_SPEC::NUM_LAYERS-1>(model);
    }

    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_buffer(nn_models::sequential::ContentBuffer<MODULE_SPEC>& buffer){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < MODULE_SPEC::SPEC::NUM_LAYERS);
        return nn_models::sequential::content_buffer<LAYER_I>(buffer);
    }
    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_buffer(const nn_models::sequential::ContentBuffer<MODULE_SPEC>& buffer){
        static_assert(LAYER_I >= 0);
        static_assert(LAYER_I < MODULE_SPEC::SPEC::NUM_LAYERS);
        return nn_models::sequential::content_buffer<LAYER_I>(buffer);
    }
    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_buffer(nn_models::sequential::ModuleBuffer<MODULE_SPEC>& buffer){
        return get_buffer<LAYER_I>(buffer.content_buffer);
    }
    template<auto LAYER_I, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_buffer(const nn_models::sequential::ModuleBuffer<MODULE_SPEC>& buffer){
        return get_buffer<LAYER_I>(buffer.content_buffer);
    }
    template <typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& get_last_buffer(nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffer){
        return get_buffer<BUFFER_SPEC::SPEC::NUM_LAYERS - 1>(buffer);
    }
    template <typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto& get_last_buffer(const nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffer){
        return get_buffer<BUFFER_SPEC::SPEC::NUM_LAYERS - 1>(buffer);
    }

    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE&, Tensor<SPEC>& tensor){
        return tensor;
    }
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE&, const Tensor<SPEC>& tensor){
        return tensor;
    }
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, Matrix<SPEC>& matrix){
        auto output_tensor = to_tensor(device, matrix);
        auto output_tensor_reshaped = reshape_row_major(device, output_tensor, TARGET_SHAPE{});
        return output_tensor_reshaped;
    }
    template <typename TARGET_SHAPE, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto _content_output_helper(DEVICE& device, const Matrix<SPEC>& matrix){
        auto output_tensor = to_tensor(device, matrix);
        auto output_tensor_reshaped = reshape_row_major(device, output_tensor, TARGET_SHAPE{});
        return output_tensor_reshaped;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto content_output(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& m){
        return nn_models::sequential::layer_output<SPEC::NUM_LAYERS - 1>(device, m);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto content_output(DEVICE& device, const nn_models::sequential::ModuleGradient<SPEC>& m){
        return nn_models::sequential::layer_output<SPEC::NUM_LAYERS - 1>(device, m);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& m){
        return nn_models::sequential::layer_output<SPEC::NUM_LAYERS - 1>(device, m);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, const nn_models::sequential::ModuleGradient<SPEC>& m){
        return nn_models::sequential::layer_output<SPEC::NUM_LAYERS - 1>(device, m);
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void _evaluate(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const INPUT& input, OUTPUT& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::sequential::evaluate_impl<TICK, 0>(device, model, input, output, buffers, content_buffer, rng, mode);
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const INPUT& input, OUTPUT& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        _evaluate<TICK>(device, model, input, output, buffers, buffers.content_buffer, rng, mode);
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename STATE_SPEC, typename CONTENT_STATE_SPEC, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void _evaluate_step(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const INPUT& input, nn_models::sequential::ModuleState<STATE_SPEC>& state, nn_models::sequential::ContentState<CONTENT_STATE_SPEC>& content_state, OUTPUT& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::sequential::evaluate_step_impl<TICK, 0>(device, model, input, state, content_state, output, buffers, content_buffer, rng, mode);
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename STATE_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const INPUT& input, nn_models::sequential::ModuleState<STATE_SPEC>& state, OUTPUT& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(length(typename INPUT::SPEC::SHAPE{}) == 2, "evaluate_step input must be rank 2 (batch x features)");
        static_assert(length(typename OUTPUT::SPEC::SHAPE{}) == 2, "evaluate_step output must be rank 2 (batch x features)");
        _evaluate_step<TICK>(device, model, input, state, state.content_state, output, buffers, buffers.content_buffer, rng, mode);
    }

    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void _forward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module, INPUT& input, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::sequential::forward_impl<0>(device, module, input, buffer, rng, mode);
    }

    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module, INPUT& input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        _forward(device, module, input, buffers.content_buffer, rng, mode);
    }

    template <typename DEVICE, typename MODULE_SPEC, typename INPUT, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module, INPUT& input, OUTPUT& output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        forward(device, module, input, buffers, rng, mode);
        auto output_tensor = rl_tools::output(device, module);
        copy(device, device, output_tensor, output);
    }

    template <auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& module){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            zero_gradient(device, nn_models::sequential::layer<LAYER_I>(module));
            zero_gradient<LAYER_I + 1>(device, module);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& module, OPTIMIZER& optimizer) {
        if constexpr(LAYER_I < SPEC::NUM_LAYERS){
            _reset_optimizer_state(device, nn_models::sequential::layer<LAYER_I>(module), optimizer);
            _reset_optimizer_state<LAYER_I + 1>(device, module, optimizer);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& module) {
        if constexpr(LAYER_I < SPEC::NUM_LAYERS){
            reset_forward_state(device, nn_models::sequential::layer<LAYER_I>(module));
            reset_forward_state<LAYER_I + 1>(device, module);
        }
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename INPUT, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void _backward_full(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const INPUT& input, D_OUTPUT& d_output, D_INPUT& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        nn_models::sequential::backward_full_impl<TICK, 0>(device, model, input, d_output, d_input, buffers, content_buffer, mode);
    }

    template<typename DEVICE, typename MODULE_SPEC, typename INPUT, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const INPUT& input, D_OUTPUT& d_output, D_INPUT& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        _backward_full(device, model, input, d_output, d_input, buffers, buffers.content_buffer, mode);
    }

    template<bool TICK = true, typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename CONTENT_BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void _backward_input(DEVICE& device, nn_models::sequential::ModuleBackward<MODULE_SPEC>& model, D_OUTPUT& d_output, D_INPUT& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, nn_models::sequential::ContentBuffer<CONTENT_BUFFER_SPEC>& content_buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn_models::sequential::backward_input_impl<TICK, 0>(device, model, d_output, d_input, buffers, content_buffer, mode);
    }

    template<typename DEVICE, typename MODULE_SPEC, typename D_OUTPUT, typename D_INPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, nn_models::sequential::ModuleBackward<MODULE_SPEC>& model, D_OUTPUT& d_output, D_INPUT& d_input, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        _backward_input(device, model, d_output, d_input, buffers, buffers.content_buffer, mode);
    }

    template<typename DEVICE, typename MODULE_SPEC, typename INPUT, typename D_OUTPUT, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn_models::sequential::ModuleGradient<MODULE_SPEC>& model, const INPUT& input, D_OUTPUT& d_output, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        if constexpr(MODULE_SPEC::NUM_LAYERS > 1){
            using FIRST_OUTPUT_SHAPE = typename nn_models::sequential::layer_spec_t<0, MODULE_SPEC>::OUTPUT_SHAPE;
            auto current_d_input_buffer_view = view_memory<FIRST_OUTPUT_SHAPE>(device, buffers.tick);
            auto first_output = nn_models::sequential::layer_output<0>(device, model);
            nn_models::sequential::backward_full_impl<false, 1>(device, model, first_output, d_output, current_d_input_buffer_view, buffers, buffers.content_buffer, mode);
            backward(device, nn_models::sequential::layer<0>(model), input, current_d_input_buffer_view, nn_models::sequential::content_buffer<0>(buffers.content_buffer), mode);
        }
        else{
            backward(device, nn_models::sequential::layer<0>(model), input, d_output, nn_models::sequential::content_buffer<0>(buffers.content_buffer), mode);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn_models::sequential::ModuleGradient<SPEC>& model, OPTIMIZER& optimizer) {
        if constexpr(LAYER_I < SPEC::NUM_LAYERS){
            update(device, nn_models::sequential::layer<LAYER_I>(model), optimizer);
            update<LAYER_I + 1>(device, model, optimizer);
        }
    }

    template<auto LAYER_I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn_models::sequential::ModuleForward<SOURCE_SPEC>& source, nn_models::sequential::ModuleForward<TARGET_SPEC>& target){
        if constexpr(LAYER_I < SOURCE_SPEC::NUM_LAYERS){
            copy(source_device, target_device, nn_models::sequential::layer<LAYER_I>(source), nn_models::sequential::layer<LAYER_I>(target));
            copy<LAYER_I + 1>(source_device, target_device, source, target);
        }
    }

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy_from_generic(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const SOURCE& source, nn_models::sequential::ModuleForward<TARGET_SPEC>& target){
        nn_models::sequential::copy_from_generic_layers(source_device, target_device, source, target);
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, nn_models::sequential::ModuleForward<SPEC_A>& a, const nn_models::sequential::ModuleForward<SPEC_B>& b){
        using T = typename SPEC_A::TYPE_POLICY::DEFAULT;
        if constexpr(LAYER_I < SPEC_A::NUM_LAYERS){
            return static_cast<T>(abs_diff(device, nn_models::sequential::layer<LAYER_I>(a), nn_models::sequential::layer<LAYER_I>(b))) + abs_diff<LAYER_I + 1>(device, a, b);
        } else {
            return T(0);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn_models::sequential::ModuleForward<MODULE_SPEC>& model, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(LAYER_I < MODULE_SPEC::NUM_LAYERS){
            return is_nan(device, nn_models::sequential::layer<LAYER_I>(model), mode) || is_nan<LAYER_I + 1>(device, model, mode);
        } else {
            return false;
        }
    }

    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn_models::sequential::OutputModule&, const Mode<MODE>& = Mode<mode::Default<>>{}){
        return false;
    }

    template<auto LAYER_I = 0, typename DEVICE, typename MODULE_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn_models::sequential::ContentState<MODULE_SPEC>& state, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        if constexpr(LAYER_I < MODULE_SPEC::SPEC::NUM_LAYERS){
            return is_nan(device, nn_models::sequential::content_state<LAYER_I>(state), mode) || is_nan<LAYER_I + 1>(device, state, mode);
        } else {
            return false;
        }
    }

    template<typename DEVICE, typename MODULE_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn_models::sequential::ModuleState<MODULE_SPEC>& state, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return is_nan(device, state.content_state, mode);
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, nn_models::sequential::ContentState<SPEC_A>& a, nn_models::sequential::ContentState<SPEC_B>& b){
        using T = typename SPEC_A::SPEC::TYPE_POLICY::DEFAULT;
        if constexpr(LAYER_I < SPEC_A::SPEC::NUM_LAYERS){
            return static_cast<T>(abs_diff(device, nn_models::sequential::content_state<LAYER_I>(a), nn_models::sequential::content_state<LAYER_I>(b))) + abs_diff<LAYER_I + 1>(device, a, b);
        } else {
            return T(0);
        }
    }

    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, nn_models::sequential::ModuleState<SPEC_A>& a, nn_models::sequential::ModuleState<SPEC_B>& b){
        return abs_diff(device, a.content_state, b.content_state);
    }

    template <auto LAYER_I = 0, typename DEVICE, typename BUFFER_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample(DEVICE& device, nn_models::sequential::ContentBuffer<BUFFER_SPEC>& buffers, RNG& rng){
        if constexpr(LAYER_I < BUFFER_SPEC::SPEC::NUM_LAYERS){
            sample(device, nn_models::sequential::content_buffer<LAYER_I>(buffers), rng);
            sample<LAYER_I + 1>(device, buffers, rng);
        }
    }

    template <typename DEVICE, typename BUFFER_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample(DEVICE& device, nn_models::sequential::ModuleBuffer<BUFFER_SPEC>& buffers, RNG& rng){
        sample(device, buffers.content_buffer, rng);
    }

    template <auto LAYER_I = 0, typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void print(DEVICE& device, const nn_models::sequential::ModuleForward<SPEC>& model){
        if constexpr(LAYER_I < SPEC::NUM_LAYERS){
            using TI = typename DEVICE::index_t;
            using LAYER_TYPE = utils::typing::remove_reference_t<decltype(nn_models::sequential::layer<LAYER_I>(model))>;
            log(device, device.logger, "Layer ", static_cast<TI>(LAYER_I), ": ", LAYER_TYPE::INPUT_DIM, " => ", LAYER_TYPE::OUTPUT_DIM);
            print<LAYER_I + 1>(device, model);
        }
    }

    namespace nn_models::sequential{
        template <auto LAYER_I = 0, typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT auto gradient_norm_sum(DEVICE& device, const ModuleForward<SPEC>& model){
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            if constexpr(LAYER_I < SPEC::NUM_LAYERS){
                return static_cast<T>(gradient_norm(device, layer<LAYER_I>(model))) + gradient_norm_sum<LAYER_I + 1>(device, model);
            } else {
                return T(0);
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto gradient_norm(DEVICE& device, const nn_models::sequential::ModuleForward<SPEC>& model, bool initial = true){
        auto return_value = nn_models::sequential::gradient_norm_sum(device, model);
        if(initial) {
            return_value = math::sqrt(device.math, return_value);
        }
        return return_value;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
