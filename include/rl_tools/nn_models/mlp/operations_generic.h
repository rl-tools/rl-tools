#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H

#include "network.h"
#include "../../nn/operations_generic.h"
#include "../../nn/parameters/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& network) {
        malloc(device, network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            malloc(device, network.hidden_layers[layer_i]);
        }
        malloc(device, network.output_layer);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>& buffers) {
        malloc(device, buffers.tick);
        malloc(device, buffers.tock);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& network) {
        free(device, network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            free(device, network.hidden_layers[layer_i]);
        }
        free(device, network.output_layer);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetworkBuffers<SPEC>& buffers) {
        free(device, buffers.tick);
        free(device, buffers.tock);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& network, RNG& rng) {
        init_weights(device, network.input_layer, rng);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            init_weights(device, network.hidden_layers[layer_i], rng);
        }
        init_weights(device, network.output_layer, rng);
    }

    template <typename SPEC>
    constexpr auto& output(nn_models::mlp::NeuralNetworkForward<SPEC>& m){
        return m.output_layer.output;
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers

    namespace nn_models::mlp{
        template <typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        constexpr bool check_input_output_f(){
            static_assert(INPUT_SPEC::COLS == MODEL_SPEC::INPUT_DIM);
            static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
            static_assert(OUTPUT_SPEC::COLS == MODEL_SPEC::OUTPUT_DIM);
            static_assert(!MODEL_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename MODEL_SPEC::T, typename INPUT_SPEC::T>);
            static_assert(!MODEL_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename INPUT_SPEC::T, typename OUTPUT_SPEC::T>);
            return true;
        }
        template <typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        constexpr bool check_input_output = check_input_output_f<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>();
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetworkForward<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE >= BATCH_SIZE);
        static_assert(BUFFER_MODEL_SPEC::DIM >= MODEL_SPEC::HIDDEN_DIM);
        matrix::ViewSpec<BATCH_SIZE, MODEL_SPEC::HIDDEN_DIM> hidden_vs;
        {
            auto output_buffer_view = view(device, buffers.tick, matrix::ViewSpec<BATCH_SIZE, MODEL_SPEC::HIDDEN_DIM>{});
            evaluate(device, network.input_layer, input, output_buffer_view, buffers.layer_buffer, rng);
        }
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            auto& input_buffer = (layer_i % 2 == 0) ? buffers.tick : buffers.tock;
            auto input_buffer_view = view(device, input_buffer, hidden_vs);
            auto& output_buffer = (layer_i % 2 == 0) ? buffers.tock : buffers.tick;
            auto output_buffer_view = view(device, output_buffer, hidden_vs);
            evaluate(device, network.hidden_layers[layer_i], input_buffer_view, output_buffer_view, buffers.layer_buffer, rng);
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            auto input_buffer_view = view(device, buffers.tick, hidden_vs);
            evaluate(device, network.output_layer, input_buffer_view, output, buffers.layer_buffer, rng);
        } else {
            auto input_buffer_view = view(device, buffers.tock, hidden_vs);
            evaluate(device, network.output_layer, input_buffer_view, output, buffers.layer_buffer, rng);
        }
    }

//    // forward modifies intermediate outputs and pre activations to facilitate backward pass
//    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TEMP_SPEC, typename RNG, typename MODE = nn::mode::Default>
//    void forward_memless(DEVICE& device, const nn_models::mlp::NeuralNetworkForward<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<TEMP_SPEC>& layer_output_tick, Matrix<TEMP_SPEC>& layer_output_tock, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
//        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
//        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
//        static_assert(TEMP_SPEC::ROWS == BATCH_SIZE);
//        static_assert(TEMP_SPEC::COLS == MODEL_SPEC::HIDDEN_DIM);
//
//        forward(network.input_layer, input, layer_output_tick, rng);
//        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
//            if(layer_i % 2 == 0){
//                forward(network.hidden_layers[layer_i], layer_output_tick, layer_output_tock, rng);
//            } else {
//                forward(network.hidden_layers[layer_i], layer_output_tock, layer_output_tick, rng);
//            }
//        }
//        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
//            forward(network.output_layer, layer_output_tick, output, rng);
//        } else {
//            forward(network.output_layer, layer_output_tock, output, rng);
//        }
//    }
//    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC, typename RNG, typename MODE = nn::mode::Default>
//    void forward(DEVICE& device, const nn_models::mlp::NeuralNetworkForward<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
//        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
//        forward_memless(device, network, input, output, buffers.tick, buffers.tock, rng);
//    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        nn::layers::dense::Buffer layer_buffer;
        forward(device, network.input_layer, input, layer_buffer, rng);

        auto current_output = network.input_layer.output;
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            forward(device, network.hidden_layers[layer_i], current_output, layer_buffer, rng);
            current_output = network.hidden_layers[layer_i].output;
        }
        forward(device, network.output_layer, current_output, layer_buffer, rng);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>& buffers, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, network, input, buffers, rng);
        copy(device, device, network.output_layer.output, output);
    }

    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<SPEC>& network) {
        zero_gradient(device, network.input_layer);
        for(typename DEVICE::index_t i = 0; i < SPEC::NUM_HIDDEN_LAYERS; i++){
            zero_gradient(device, network.hidden_layers[i]);
        }
        zero_gradient(device, network.output_layer);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_MODEL_SPEC, typename MODE = nn::mode::Default>
    void backward_input(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<MODEL_SPEC>& network, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffer, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        // ATTENTION: this modifies d_output (uses it as a buffer for the d_pre_activations
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == BATCH_SIZE);
        static_assert(BUFFER_MODEL_SPEC::DIM >= MODEL_SPEC::HIDDEN_DIM);
        using T = typename MODEL_SPEC::T;
        using TI = typename DEVICE::index_t;

        backward_input(device, network.output_layer, d_output, buffer.tick, buffer.layer_buffer);
        for (typename DEVICE::index_t layer_i_plus_one = MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            if(layer_i % 2 == (MODEL_SPEC::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward_input(device, network.hidden_layers[layer_i], buffer.tick, buffer.tock, buffer.layer_buffer);
            } else {
                backward_input(device, network.hidden_layers[layer_i], buffer.tock, buffer.tick, buffer.layer_buffer);
            }
        }
        auto& target_d_output_buffer = (MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0) ? buffer.tick : buffer.tock;
        backward_input(device, network.input_layer, target_d_output_buffer, d_input, buffer.layer_buffer);
    }

    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_MODEL_SPEC, typename D_INPUT_SPEC, typename MODE = nn::mode::Default>
    void backward_full(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffer, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        // ATTENTION: this modifies d_output (uses it as a buffer for the d_pre_activations
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == BATCH_SIZE);
        static_assert(BUFFER_MODEL_SPEC::DIM >= MODEL_SPEC::HIDDEN_DIM);

        auto previous_output = MODEL_SPEC::NUM_HIDDEN_LAYERS > 0 ? network.hidden_layers[MODEL_SPEC::NUM_HIDDEN_LAYERS - 1].output : network.input_layer.output;
        backward_full(device, network.output_layer, previous_output, d_output, buffer.tick, buffer.layer_buffer);
        for (typename DEVICE::index_t layer_i_plus_one = MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            previous_output = layer_i > 0 ? network.hidden_layers[layer_i - 1].output : network.input_layer.output;
            if(layer_i % 2 == (MODEL_SPEC::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward_full(device, network.hidden_layers[layer_i], previous_output, buffer.tick, buffer.tock, buffer.layer_buffer);
            } else {
                backward_full(device, network.hidden_layers[layer_i], previous_output, buffer.tock, buffer.tick, buffer.layer_buffer);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            backward_full(device, network.input_layer, input, buffer.tick, d_input, buffer.layer_buffer);
        } else {
            backward_full(device, network.input_layer, input, buffer.tock, d_input, buffer.layer_buffer);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_MODEL_SPEC, typename MODE = nn::mode::Default>
    void backward(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffer, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        // ATTENTION: this modifies d_output (uses it as a buffer for the d_pre_activations
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == BATCH_SIZE);
        static_assert(BUFFER_MODEL_SPEC::DIM >= MODEL_SPEC::HIDDEN_DIM);

        auto previous_output = MODEL_SPEC::NUM_HIDDEN_LAYERS > 0 ? network.hidden_layers[MODEL_SPEC::NUM_HIDDEN_LAYERS - 1].output : network.input_layer.output;
        backward_full(device, network.output_layer, previous_output, d_output, buffer.tick, buffer.layer_buffer);
        for (typename DEVICE::index_t layer_i_plus_one = MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            previous_output = layer_i > 0 ? network.hidden_layers[layer_i - 1].output : network.input_layer.output;
            if(layer_i % 2 == (MODEL_SPEC::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward_full(device, network.hidden_layers[layer_i], previous_output, buffer.tick, buffer.tock, buffer.layer_buffer);
            } else {
                backward_full(device, network.hidden_layers[layer_i], previous_output, buffer.tock, buffer.tick, buffer.layer_buffer);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(device, network.input_layer, input, buffer.tick, buffer.layer_buffer);
        } else {
            backward(device, network.input_layer, input, buffer.tock, buffer.layer_buffer);
        }
    }
//    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_MODEL_SPEC>
//    void backward_param(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffer) {
//        backward(device, network, input, d_output, buffer);
//    }

    template<typename DEVICE, typename SPEC, typename ADAM_PARAMETERS>
    void update(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<SPEC>& network, nn::optimizers::Adam<ADAM_PARAMETERS>& optimizer) {
        using T = typename SPEC::T;
        update(device, network.input_layer, optimizer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            update(device, network.hidden_layers[layer_i], optimizer);
        }
        update(device, network.output_layer, optimizer);
    }

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::mlp::NeuralNetworkGradient<SPEC>& network, OPTIMIZER& optimizer) {
        // this function is marked with a underscore because it should usually be called from the reset_optimizer_state function of the optimizer to have one coherent entrypoint for resetting the optimizer state in the optimizer and in the model
        _reset_optimizer_state(device, network.input_layer, optimizer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            _reset_optimizer_state(device, network.hidden_layers[layer_i], optimizer);
        }
        _reset_optimizer_state(device, network.output_layer, optimizer);
    }

    // The following copy operators are more powerful than the default copy assignment operator in that they can e.g. copy between networks with different activation functions
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::mlp::NeuralNetworkForward<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetworkForward<TARGET_SPEC>& target){
        static_assert(rl_tools::nn_models::mlp::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>, "The source and target network must have the same structure");
        copy(source_device, target_device, source.input_layer, target.input_layer);
        for(typename SOURCE_SPEC::TI layer_i = 0; layer_i <  SOURCE_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            copy(source_device, target_device, source.hidden_layers[layer_i], target.hidden_layers[layer_i]);
        }
        copy(source_device, target_device, source.output_layer, target.output_layer);
    }

//    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
//    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::mlp::NeuralNetworkAdam<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetworkAdam<TARGET_SPEC>& target){
//        static_assert(rl_tools::nn_models::mlp::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>, "The source and target network must have the same structure");
//        copy(source_device, target_device, (nn_models::mlp::NeuralNetworkForward<SOURCE_SPEC>&)source, (nn_models::mlp::NeuralNetworkForward<TARGET_SPEC>&)target);
//    }

    template<typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC>& n){
        reset_forward_state(device, n.input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i <  SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            reset_forward_state(device, n.hidden_layers[layer_i]);
        }
        reset_forward_state(device, n.output_layer);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, nn_models::mlp::NeuralNetworkForward<SPEC_1>& n1, const nn_models::mlp::NeuralNetworkForward<SPEC_2>& n2){
        static_assert(rl_tools::nn_models::mlp::check_spec_memory<SPEC_1, SPEC_2>, "The source and target network must have the same structure");
        typename SPEC_1::T acc = 0;

        acc += abs_diff(device, n1.output_layer, n2.output_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC_1::NUM_HIDDEN_LAYERS; layer_i++){
            acc += abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
        }
        acc += abs_diff(device, n1.input_layer, n2.input_layer);
        return acc;
    }
    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const rl_tools::nn_models::mlp::NeuralNetworkForward<SPEC>& n) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, n.input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            found_nan = found_nan || is_nan(device, n.hidden_layers[layer_i]);
        }
        found_nan = found_nan || is_nan(device, n.output_layer);
        return found_nan;
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::mlp::NeuralNetworkBuffers<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetworkBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.tick, target.tick);
        copy(source_device, target_device, source.tock, target.tock);
    }
    template<typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto& output(nn_models::mlp::NeuralNetworkGradient<SPEC>& nn){
        return nn.output_layer.output;
    }
    template <typename DEVICE, typename BUFFER_SPEC, typename RNG>
    void sample(DEVICE& device, nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>& buffers, RNG& rng){ }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
