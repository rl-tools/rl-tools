#ifndef LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_GENERIC_H

#include <layer_in_c/nn_models/mlp/network.h>
#include <layer_in_c/nn/operations_generic.h>

namespace layer_in_c {

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], typename SPEC::T output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM]){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T layer_output_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T layer_output_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        evaluate(network.input_layer, input, layer_output_tick);
        for (index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                evaluate(network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                evaluate(network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            evaluate(network.output_layer, layer_output_tick, output);
        } else {
            evaluate(network.output_layer, layer_output_tock, output);
        }
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T evaluate(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        static_assert(NetworkType::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[NetworkType::OUTPUT_DIM];
        evaluate(network, input, output);
        return output[0];
    }
    // forward modifies intermediate outputs and pre activations to facilitate backward pass
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::mlp::NeuralNetworkBackward<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], typename SPEC::T output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM]){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T layer_output_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T layer_output_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        forward(network.input_layer, input, layer_output_tick);
        for (index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                forward(network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                forward(network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            forward(network.output_layer, layer_output_tick, output);
        } else {
            forward(network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        forward(network.input_layer, input);

        auto current_output = network.input_layer.output;
        for (index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            forward(network.hidden_layers[layer_i], current_output);
            current_output = network.hidden_layers[layer_i].output;
        }
        forward(network.output_layer, current_output);
    }
    template<typename DEVICE, typename SPEC>
    [[deprecated("Calling forward with an output buffer on a layer requiring the gradient is not recommended. Consider using forward without an output buffer to avoid unecessary copies instead.")]]
    FUNCTION_PLACEMENT void forward(nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], typename SPEC::T output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM]) {
        forward(network, input);
        for(index_t i=0; i < utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM; i++){
            output[i] = network.output_layer.output[i];
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(nn_models::mlp::NeuralNetworkBackward<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        static_assert(utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[1];
        forward(network, input, output);
        return output[0];
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        static_assert(utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        forward(network, input);
        return network.output_layer.output[0];
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void zero_gradient(nn_models::mlp::NeuralNetwork<DEVICE, SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        zero_gradient(network.input_layer);
        for(index_t i = 0; i < NetworkType::NUM_HIDDEN_LAYERS; i++){
            zero_gradient(network.hidden_layers[i]);
        }
        zero_gradient(network.output_layer);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(nn_models::mlp::NeuralNetworkBackward<DEVICE, SPEC>& network, const typename SPEC::T d_output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM], typename SPEC::T d_input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T d_layer_input_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T d_layer_input_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        backward(network.output_layer, d_output, d_layer_input_tick);
        for (index_t layer_i_plus_one = NetworkType::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            index_t layer_i = layer_i_plus_one - 1;
            if(layer_i % 2 == (NetworkType::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward(network.hidden_layers[layer_i], d_layer_input_tick, d_layer_input_tock);
            } else {
                backward(network.hidden_layers[layer_i], d_layer_input_tock, d_layer_input_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(network.input_layer, d_layer_input_tick, d_input);
        } else {
            backward(network.input_layer, d_layer_input_tock, d_input);
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], const typename SPEC::T d_output[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM], typename SPEC::T d_input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;

        typename SPEC::T d_layer_input_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T d_layer_input_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        auto previous_output = NetworkType::NUM_HIDDEN_LAYERS > 0 ? network.hidden_layers[NetworkType::NUM_HIDDEN_LAYERS - 1].output : network.input_layer.output;
        backward(network.output_layer, previous_output, d_output, d_layer_input_tick);
        for (index_t layer_i_plus_one = NetworkType::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            index_t layer_i = layer_i_plus_one - 1;
            previous_output = layer_i > 0 ? network.hidden_layers[layer_i - 1].output : network.input_layer.output;
            if(layer_i % 2 == (NetworkType::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward(network.hidden_layers[layer_i], previous_output, d_layer_input_tick, d_layer_input_tock);
            } else {
                backward(network.hidden_layers[layer_i], previous_output, d_layer_input_tock, d_layer_input_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(network.input_layer, input, d_layer_input_tick, d_input);
        } else {
            backward(network.input_layer, input, d_layer_input_tock, d_input);
        }
    }
    template<typename DEVICE, typename SPEC, index_t BATCH_SIZE>
    FUNCTION_PLACEMENT void forward_backward_mse(nn_models::mlp::NeuralNetworkBackwardGradient<DEVICE, SPEC>& network, const typename SPEC::T input[utils::typing::remove_reference<decltype(network)>::type::INPUT_DIM], typename SPEC::T target[utils::typing::remove_reference<decltype(network)>::type::OUTPUT_DIM]) {
        typename SPEC::T d_input[SPEC::STRUCTURE_SPEC::INPUT_DIM];
        forward(network, input);
        typename SPEC::T d_loss_d_output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<typename SPEC::T, SPEC::STRUCTURE_SPEC::OUTPUT_DIM, BATCH_SIZE>(network.output_layer.output, target, d_loss_d_output);
        backward(network, input, d_loss_d_output, d_input);
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::mlp::NeuralNetworkSGD<DEVICE, SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        update_layer(network.input_layer);
        for (index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_layer(network.hidden_layers[layer_i]);
        }
        update_layer(network.output_layer);
    }


    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::mlp::NeuralNetworkAdam<DEVICE, SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T  first_order_moment_bias_correction = 1/(1 - math::pow(SPEC::ADAM_PARAMETERS::BETA_1, network.age));
        typename SPEC::T second_order_moment_bias_correction = 1/(1 - math::pow(SPEC::ADAM_PARAMETERS::BETA_2, network.age));

        update_layer(network.input_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        for(index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_layer(network.hidden_layers[layer_i], first_order_moment_bias_correction, second_order_moment_bias_correction);
        }
        update_layer(network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::mlp::NeuralNetworkSGD<DEVICE, SPEC>& network) {
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::mlp::NeuralNetworkAdam<DEVICE, SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        reset_optimizer_state(network.input_layer);
        for(index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            reset_optimizer_state(network.hidden_layers[layer_i]);
        }
        reset_optimizer_state(network.output_layer);
        network.age = 1;
    }


}

#endif
