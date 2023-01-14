#ifndef LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_MODELS_MLP_OPERATIONS_GENERIC_H

#include <layer_in_c/nn_models/mlp/network.h>
#include <layer_in_c/nn/operations_generic.h>

namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        malloc(device, network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            malloc(device, network.hidden_layers[layer_i]);
        }
        malloc(device, network.output_layer);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network, RNG& rng) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        init_kaiming(device, network.input_layer, rng);
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            init_kaiming(device, network.hidden_layers[layer_i], rng);
        }
        init_kaiming(device, network.output_layer, rng);
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void evaluate_memless(DEVICE& device, const nn_models::mlp::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], typename SPEC::T output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM], typename SPEC::T layer_output_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM], typename SPEC::T layer_output_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM]){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        evaluate(device, network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            if(layer_i % 2 == 0){
                evaluate(device, network.hidden_layers[layer_i], layer_output_tick, layer_output_tock);
            } else {
                evaluate(device, network.hidden_layers[layer_i], layer_output_tock, layer_output_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            evaluate(device, network.output_layer, layer_output_tick, output);
        } else {
            evaluate(device, network.output_layer, layer_output_tock, output);
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], typename SPEC::T output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM]){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T layer_output_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T layer_output_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        evaluate_memless(device, network, input, output, layer_output_tick, layer_output_tock);
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T evaluate(DEVICE& device, const nn_models::mlp::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        static_assert(NetworkType::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[NetworkType::OUTPUT_DIM];
        evaluate(device, network, input, output);
        return output[0];
    }
    // forward modifies intermediate outputs and pre activations to facilitate backward pass
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], typename SPEC::T output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM]){
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T layer_output_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T layer_output_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        forward(network.input_layer, input, layer_output_tick);
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
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
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        forward(device, network.input_layer, input);

        auto current_output = network.input_layer.output;
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            forward(device, network.hidden_layers[layer_i], current_output);
            current_output = network.hidden_layers[layer_i].output;
        }
        forward(device, network.output_layer, current_output);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], typename SPEC::T output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM]) {
        forward(device, network, input);
        for(typename DEVICE::index_t i=0; i < SPEC::STRUCTURE_SPEC::OUTPUT_DIM; i++){
            output[i] = network.output_layer.output[i];
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        static_assert(SPEC::STRUCTURE_SPEC::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[1];
        forward(device, network, input, output);
        return output[0];
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        static_assert(SPEC::STRUCTURE_SPEC::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        forward(device, network, input);
        return network.output_layer.output[0];
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        zero_gradient(device, network.input_layer);
        for(typename DEVICE::index_t i = 0; i < NetworkType::NUM_HIDDEN_LAYERS; i++){
            zero_gradient(device, network.hidden_layers[i]);
        }
        zero_gradient(device, network.output_layer);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(DEVICE& device, nn_models::mlp::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T d_output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T d_layer_input_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T d_layer_input_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        backward(network.output_layer, d_output, d_layer_input_tick);
        for (typename DEVICE::index_t layer_i_plus_one = NetworkType::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
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
    FUNCTION_PLACEMENT void backward(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::STRUCTURE_SPEC::INPUT_DIM]) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;

        typename SPEC::T d_layer_input_tick[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        typename SPEC::T d_layer_input_tock[SPEC::STRUCTURE_SPEC::HIDDEN_DIM];
        auto previous_output = NetworkType::NUM_HIDDEN_LAYERS > 0 ? network.hidden_layers[NetworkType::NUM_HIDDEN_LAYERS - 1].output : network.input_layer.output;
        backward(device, network.output_layer, previous_output, d_output, d_layer_input_tick);
        for (typename DEVICE::index_t layer_i_plus_one = NetworkType::NUM_HIDDEN_LAYERS; layer_i_plus_one > 0; layer_i_plus_one--){
            typename DEVICE::index_t layer_i = layer_i_plus_one - 1;
            previous_output = layer_i > 0 ? network.hidden_layers[layer_i - 1].output : network.input_layer.output;
            if(layer_i % 2 == (NetworkType::NUM_HIDDEN_LAYERS - 1) % 2){ // we are starting with the last hidden layer where the result should go to tock
                backward(device, network.hidden_layers[layer_i], previous_output, d_layer_input_tick, d_layer_input_tock);
            } else {
                backward(device, network.hidden_layers[layer_i], previous_output, d_layer_input_tock, d_layer_input_tick);
            }
        }
        if constexpr(NetworkType::NUM_HIDDEN_LAYERS % 2 == 0){
            backward(device, network.input_layer, input, d_layer_input_tick, d_input);
        } else {
            backward(device, network.input_layer, input, d_layer_input_tock, d_input);
        }
    }
    template<typename DEVICE, typename SPEC, auto BATCH_SIZE>
    FUNCTION_PLACEMENT void forward_backward_mse(DEVICE& device, nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>& network, const typename SPEC::T input[SPEC::STRUCTURE_SPEC::INPUT_DIM], typename SPEC::T target[SPEC::STRUCTURE_SPEC::OUTPUT_DIM]) {
        typename SPEC::T d_input[SPEC::STRUCTURE_SPEC::INPUT_DIM];
        forward(device, network, input);
        typename SPEC::T d_loss_d_output[SPEC::STRUCTURE_SPEC::OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<DEVICE, typename SPEC::T, SPEC::STRUCTURE_SPEC::OUTPUT_DIM, BATCH_SIZE>(device, network.output_layer.output, target, d_loss_d_output);
        backward(device, network, input, d_loss_d_output, d_input);
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update(DEVICE& device, nn_models::mlp::NeuralNetworkSGD<SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        update_layer(network.input_layer);
        for (typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_layer(network.hidden_layers[layer_i]);
        }
        update_layer(network.output_layer);
    }


    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update(DEVICE& device, nn_models::mlp::NeuralNetworkAdam<SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        typename SPEC::T  first_order_moment_bias_correction = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), SPEC::ADAM_PARAMETERS::BETA_1, network.age));
        typename SPEC::T second_order_moment_bias_correction = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), SPEC::ADAM_PARAMETERS::BETA_2, network.age));

        update_layer(device, network.input_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_layer(device, network.hidden_layers[layer_i], first_order_moment_bias_correction, second_order_moment_bias_correction);
        }
        update_layer(device, network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(DEVICE& device, nn_models::mlp::NeuralNetworkSGD<SPEC>& network) {
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(DEVICE& device, nn_models::mlp::NeuralNetworkAdam<SPEC>& network) {
        using NetworkType = typename utils::typing::remove_reference<decltype(network)>::type;
        reset_optimizer_state(device, network.input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i < NetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            reset_optimizer_state(device, network.hidden_layers[layer_i]);
        }
        reset_optimizer_state(device, network.output_layer);
        network.age = 1;
    }

    // The following copy operators are more powerful than the default copy assignment operator in that they can e.g. copy between networks with different activation functions
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetwork<TARGET_SPEC>* target, const nn_models::mlp::NeuralNetwork<SOURCE_SPEC>* source){
        static_assert(layer_in_c::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(target->input_layer, source->input_layer);
        for(typename TARGET_SPEC::TI layer_i = 0; layer_i <  utils::typing::remove_pointer<decltype(target)>::type::NUM_HIDDEN_LAYERS; layer_i++){
            copy(target->hidden_layers[layer_i], source->hidden_layers[layer_i]);
        }
        copy(target->output_layer, source->output_layer);
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetwork<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetwork<SOURCE_SPEC>& source){
        static_assert(layer_in_c::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(&target, &source);
    }

    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetworkAdam<TARGET_SPEC>* target, const nn_models::mlp::NeuralNetworkAdam<SOURCE_SPEC>* source){
        static_assert(layer_in_c::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy((nn_models::mlp::NeuralNetwork<TARGET_SPEC>*)target, (nn_models::mlp::NeuralNetwork<SOURCE_SPEC>*)source);
        target->age = source->age;
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn_models::mlp::NeuralNetworkAdam<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetworkAdam<SOURCE_SPEC>& source){
        static_assert(layer_in_c::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(&target, &source);
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>* n){
        reset_forward_state(device, n->input_layer);
        for(typename DEVICE::index_t layer_i = 0; layer_i <  utils::typing::remove_pointer<decltype(n)>::type::NUM_HIDDEN_LAYERS; layer_i++){
            reset_forward_state(device, n->hidden_layers[layer_i]);
        }
        reset_forward_state(device, n->output_layer);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetwork<SPEC>& n){
        reset_forward_state(device, &n);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetworkAdam<SPEC>* n){
        reset_forward_state(device, (nn_models::mlp::NeuralNetwork<SPEC>*)n);
        n->age = 1; // not technically forward state but fits the same category from a usage point of view
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn_models::mlp::NeuralNetworkAdam<SPEC>& n){
        reset_forward_state(device, &n);
    }


}

#endif
