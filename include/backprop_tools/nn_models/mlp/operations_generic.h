#ifndef BACKPROP_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_NN_MODELS_MLP_OPERATIONS_GENERIC_H

#include <backprop_tools/nn_models/mlp/network.h>
#include <backprop_tools/nn/operations_generic.h>
#include <backprop_tools/nn/parameters/operations_generic.h>
#include <backprop_tools/nn/optimizers/adam/operations_generic.h>

namespace backprop_tools {
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
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
    template<typename DEVICE, typename BUFFER_SPEC>
    void malloc(DEVICE& device, nn_models::mlp::NeuralNetworkBuffersForwardBackward<BUFFER_SPEC>& buffers) {
        malloc(device, (nn_models::mlp::NeuralNetworkBuffers<BUFFER_SPEC>&) buffers);
        malloc(device, buffers.d_input);
        malloc(device, buffers.d_output);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
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
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn_models::mlp::NeuralNetworkBuffersForwardBackward<SPEC>& buffers) {
        free(device, (nn_models::mlp::NeuralNetworkBuffers<SPEC>&) buffers);
        free(device, buffers.d_input);
        free(device, buffers.d_output);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, RNG& rng) {
        init_weights(device, network.input_layer, rng);
        for (typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            init_weights(device, network.hidden_layers[layer_i], rng);
        }
        init_weights(device, network.output_layer, rng);
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers

    namespace nn_models::mlp{
        template <typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        constexpr bool check_input_output =
                INPUT_SPEC::COLS == MODEL_SPEC::INPUT_DIM &&
                INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS &&
                OUTPUT_SPEC::COLS == MODEL_SPEC::OUTPUT_DIM &&
                (!MODEL_SPEC::ENFORCE_FLOATING_POINT_TYPE || ( utils::typing::is_same_v<typename MODEL_SPEC::T, typename INPUT_SPEC::T> && utils::typing::is_same_v<typename INPUT_SPEC::T, typename OUTPUT_SPEC::T>));
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TEMP_SPEC>
    void evaluate_memless(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<TEMP_SPEC>& layer_output_tick, Matrix<TEMP_SPEC>& layer_output_tock){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS >= BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS >= MODEL_SPEC::HIDDEN_DIM);
        evaluate(device, network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                evaluate(device, network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                evaluate(device, network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            evaluate(device, network.output_layer, layer_output_tick, output);
        } else {
            evaluate(device, network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC>& buffers){
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        evaluate_memless(device, network, input, output, buffers.tick, buffers.tock);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = typename MODEL_SPEC::T;
        using TICK_TOCK_SPEC = matrix::Specification<T, typename DEVICE::index_t, BATCH_SIZE, MODEL_SPEC::HIDDEN_DIM>;
#ifndef BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
        MatrixDynamic<TICK_TOCK_SPEC> layer_output_tick;
        MatrixDynamic<TICK_TOCK_SPEC> layer_output_tock;
#else
        MatrixStatic<TICK_TOCK_SPEC> layer_output_tick;
        MatrixStatic<TICK_TOCK_SPEC> layer_output_tock;
#endif
        malloc(device, layer_output_tick);
        malloc(device, layer_output_tock);
        evaluate_memless(device, network, input, output, layer_output_tick, layer_output_tock);
        free(device, layer_output_tick);
        free(device, layer_output_tock);
    }

    // forward modifies intermediate outputs and pre activations to facilitate backward pass
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TEMP_SPEC>
    void forward_memless(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<TEMP_SPEC>& layer_output_tick, Matrix<TEMP_SPEC>& layer_output_tock){
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS == BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS == MODEL_SPEC::HIDDEN_DIM);

        forward(network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                forward(network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                forward(network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            forward(network.output_layer, layer_output_tick, output);
        } else {
            forward(network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void forward(DEVICE& device, const nn_models::mlp::NeuralNetwork<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffers){
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == OUTPUT_SPEC::ROWS);
        forward_memless(device, network, input, output, buffers.tick, buffers.tock);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC>
    void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input) {
        forward(device, network.input_layer, input);

        auto current_output = network.input_layer.output;
        for (typename DEVICE::index_t layer_i = 0; layer_i < MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            forward(device, network.hidden_layers[layer_i], current_output);
            current_output = network.hidden_layers[layer_i].output;
        }
        forward(device, network.output_layer, current_output);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output) {
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, network, input);
        copy(device, device, output, network.output_layer.output);
    }

    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
        zero_gradient(device, network.input_layer);
        for(typename DEVICE::index_t i = 0; i < SPEC::NUM_HIDDEN_LAYERS; i++){
            zero_gradient(device, network.hidden_layers[i]);
        }
        zero_gradient(device, network.output_layer);
    }
    template<typename DEVICE, typename MODEL_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename TEMP_SPEC>
    void backward_memless(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<MODEL_SPEC>& network, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, Matrix<TEMP_SPEC>& d_layer_input_tick, Matrix<TEMP_SPEC>& d_layer_input_tock) {
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS == BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS == MODEL_SPEC::HIDDEN_DIM);

        backward(device, network.output_layer, d_output, d_layer_input_tick);
        for (typename DEVICE::index_t layer_i_plus_one = MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            if(layer_i % 2 == (MODEL_SPEC::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward(device, network.hidden_layers[layer_i], d_layer_input_tick, d_layer_input_tock);
            } else {
                backward(device, network.hidden_layers[layer_i], d_layer_input_tock, d_layer_input_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(device, network.input_layer, d_layer_input_tick, d_input);
        } else {
            backward(device, network.input_layer, d_layer_input_tock, d_input);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void backward(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<MODEL_SPEC>& network, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffers) {
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == D_INPUT_SPEC::ROWS);
        backward_memless(device, network, d_output, d_input, buffers.tick, buffers.tock);
    }

    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename TEMP_SPEC>
    void backward_memless(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, Matrix<TEMP_SPEC>& d_layer_input_tick, Matrix<TEMP_SPEC>& d_layer_input_tock) {
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = D_INPUT_SPEC::ROWS;
        static_assert(TEMP_SPEC::ROWS == BATCH_SIZE);
        static_assert(TEMP_SPEC::COLS == MODEL_SPEC::HIDDEN_DIM);

        auto previous_output = MODEL_SPEC::NUM_HIDDEN_LAYERS > 0 ? network.hidden_layers[MODEL_SPEC::NUM_HIDDEN_LAYERS - 1].output : network.input_layer.output;
        backward(device, network.output_layer, previous_output, d_output, d_layer_input_tick);
        for (typename DEVICE::index_t layer_i_plus_one = MODEL_SPEC::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            previous_output = layer_i > 0 ? network.hidden_layers[layer_i - 1].output : network.input_layer.output;
            if(layer_i % 2 == (MODEL_SPEC::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward(device, network.hidden_layers[layer_i], previous_output, d_layer_input_tick, d_layer_input_tock);
            } else {
                backward(device, network.hidden_layers[layer_i], previous_output, d_layer_input_tock, d_layer_input_tick);
            }
        }
        if constexpr(MODEL_SPEC::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(device, network.input_layer, input, d_layer_input_tick, d_input);
        } else {
            backward(device, network.input_layer, input, d_layer_input_tock, d_input);
        }
    }
    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_MODEL_SPEC>
    void backward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn_models::mlp::NeuralNetworkBuffers<BUFFER_MODEL_SPEC> buffers) {
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == D_INPUT_SPEC::ROWS);
        backward_memless(device, network, input, d_output, d_input, buffers.tick, buffers.tock);
    }

    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename TARGET_SPEC, typename BUFFER_MODEL_SPEC>
    void forward_backward_mse(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<MODEL_SPEC>& network, const Matrix<INPUT_SPEC>& input, const Matrix<TARGET_SPEC>& target, nn_models::mlp::NeuralNetworkBuffersForwardBackward<BUFFER_MODEL_SPEC> buffers, typename MODEL_SPEC::T loss_weight = 1) {
        static_assert(BUFFER_MODEL_SPEC::BATCH_SIZE == INPUT_SPEC::ROWS);
        static_assert(nn_models::mlp::check_input_output<MODEL_SPEC, INPUT_SPEC, TARGET_SPEC>);
        forward(device, network, input);
        nn::loss_functions::mse::gradient(device, network.output_layer.output, target, buffers.d_output, loss_weight);
        backward(device, network, input, buffers.d_output, buffers.d_input, buffers);
    }

    template<typename DEVICE, typename SPEC, typename ADAM_PARAMETERS>
    void update(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, nn::optimizers::Adam<ADAM_PARAMETERS>& optimizer) {
        using T = typename SPEC::T;
        update(device, network.input_layer, optimizer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            update(device, network.hidden_layers[layer_i], optimizer);
        }
        update(device, network.output_layer, optimizer);
    }

    template<typename DEVICE, typename SPEC>
    void _reset_optimizer_state(DEVICE& device, nn_models::mlp::NeuralNetworkSGD<SPEC>& network) {
    }

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, OPTIMIZER& optimizer) {
        // this function is marked with a underscore because it should usually be called from the reset_optimizer_state function of the optimizer to have one coherent entrypoint for resetting the optimizer state in the optimizer and in the model
        _reset_optimizer_state(device, network.input_layer, optimizer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            _reset_optimizer_state(device, network.hidden_layers[layer_i], optimizer);
        }
        _reset_optimizer_state(device, network.output_layer, optimizer);
    }

    // The following copy operators are more powerful than the default copy assignment operator in that they can e.g. copy between networks with different activation functions
    template<typename TARGET_DEVICE, typename SOURCE_DEVICE,  typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn_models::mlp::NeuralNetwork<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetwork<SOURCE_SPEC>& source){
        static_assert(backprop_tools::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(target_device, source_device, target.input_layer, source.input_layer);
        for(typename TARGET_SPEC::TI layer_i = 0; layer_i <  TARGET_SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            copy(target_device, source_device, target.hidden_layers[layer_i], source.hidden_layers[layer_i]);
        }
        copy(target_device, source_device, target.output_layer, source.output_layer);
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn_models::mlp::NeuralNetworkAdam<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetworkAdam<SOURCE_SPEC>& source){
        static_assert(backprop_tools::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(target_device, source_device, (nn_models::mlp::NeuralNetwork<TARGET_SPEC>&)target, (nn_models::mlp::NeuralNetwork<SOURCE_SPEC>&)source);
    }

    template<typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& n){
        reset_forward_state(device, n.input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i <  SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            reset_forward_state(device, n.hidden_layers[layer_i]);
        }
        reset_forward_state(device, n.output_layer);
    }

    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC_1>& n1, const nn_models::mlp::NeuralNetwork<SPEC_2>& n2){
        static_assert(backprop_tools::nn_models::mlp::check_spec_memory<typename SPEC_1::STRUCTURE_SPEC, typename SPEC_2::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        typename SPEC_1::T acc = 0;

        acc += abs_diff(device, n1.output_layer, n2.output_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC_1::NUM_HIDDEN_LAYERS; layer_i++){
            acc += abs_diff(device, n1.hidden_layers[layer_i], n2.hidden_layers[layer_i]);
        }
        acc += abs_diff(device, n1.input_layer, n2.input_layer);
        return acc;
    }
    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, const backprop_tools::nn_models::mlp::NeuralNetwork<SPEC>& n) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, n.input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < SPEC::NUM_HIDDEN_LAYERS; layer_i++){
            found_nan = found_nan || is_nan(device, n.hidden_layers[layer_i]);
        }
        found_nan = found_nan || is_nan(device, n.output_layer);
        return found_nan;
    }
    template<typename TARGET_DEVICE, typename SOURCE_DEVICE,  typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn_models::mlp::NeuralNetworkBuffers<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetworkBuffers<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.tick, source.tick);
        copy(target_device, source_device, target.tock, source.tock);
    }
    template<typename TARGET_DEVICE, typename SOURCE_DEVICE,  typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn_models::mlp::NeuralNetworkBuffersForwardBackward<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetworkBuffersForwardBackward<SOURCE_SPEC>& source){
        copy(target_device, source_device, (nn_models::mlp::NeuralNetworkBuffers<TARGET_SPEC>&)target, (nn_models::mlp::NeuralNetworkBuffers<SOURCE_SPEC>&)source);
        copy(target_device, source_device, target.d_input, source.d_input);
        copy(target_device, source_device, target.d_output, source.d_output);
    }
}

#endif
