#ifndef LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_MODELS_THREE_LAYER_FC_OPERATIONS_GENERIC_H
#include <layer_in_c/nn_models/three_layer_fc/network.h>
#include <layer_in_c/nn/operations_generic.h>

namespace layer_in_c {
    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers
    template<typename SPEC>
    FUNCTION_PLACEMENT void evaluate(nn_models::three_layer_fc::NeuralNetwork<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM]){
        typename SPEC::T input_layer_output[SPEC::LAYER_1::OUTPUT_DIM];
        evaluate(network.input_layer     , input, input_layer_output);
        typename SPEC::T hidden_layer_0_output[SPEC::LAYER_2::OUTPUT_DIM];
        evaluate(network.hidden_layer_0     , input_layer_output, hidden_layer_0_output);
        evaluate(network.output_layer, hidden_layer_0_output, output);
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T evaluate(nn_models::three_layer_fc::NeuralNetwork<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        evaluate(network, input, output);
        return output[0];
    }
    // forward modifies intermediate outputs and pre activations to facilitate backward pass
    template<typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::three_layer_fc::NeuralNetworkBackward<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM]){
        typename SPEC::T input_layer_output[SPEC::LAYER_1::OUTPUT_DIM];
        typename SPEC::T hidden_layer_0_output[SPEC::LAYER_2::OUTPUT_DIM];
        forward(network.input_layer     ,          input, input_layer_output);
        forward(network.hidden_layer_0     , input_layer_output, hidden_layer_0_output);
        forward(network.output_layer, hidden_layer_0_output,         output);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::three_layer_fc::NeuralNetworkBackwardGradient<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        forward(network.input_layer     , input);
        forward(network.hidden_layer_0     , network.input_layer.output);
        forward(network.output_layer, network.hidden_layer_0.output);
    }
    template<typename SPEC>
    [[deprecated("Calling forward with an output buffer on a layer requiring the gradient is not recommended. Consider using forward without an output buffer to avoid unecessary copies instead.")]]
    FUNCTION_PLACEMENT void forward(nn_models::three_layer_fc::NeuralNetworkBackwardGradient<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        forward(network, input);
        for(int i=0; i < SPEC::OUTPUT_LAYER::OUTPUT_DIM; i++){
            output[i] = network.output_layer.output[i];
        }
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(nn_models::three_layer_fc::NeuralNetworkBackward<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[1];
        forward(network, input, output);
        return output[0];
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(nn_models::three_layer_fc::NeuralNetworkBackwardGradient<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        forward(network, input);
        return network.output_layer.output[0];
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void zero_gradient(nn_models::three_layer_fc::NeuralNetwork<devices::Generic, SPEC>& network) {
        zero_gradient(network.input_layer);
        zero_gradient(network.hidden_layer_0);
        zero_gradient(network.output_layer);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void backward(nn_models::three_layer_fc::NeuralNetworkBackward<devices::Generic, SPEC>& network, const typename SPEC::T d_output[SPEC::OUTPUT_LAYER::OUTPUT_DIM], typename SPEC::T d_input[SPEC::LAYER_1::INPUT_DIM]) {
        typename SPEC::T d_hidden_layer_0_output[SPEC::LAYER_2::SPEC::OUTPUT_DIM];
        backward(network.output_layer, d_output, d_hidden_layer_0_output);
        typename SPEC::T d_input_layer_output[SPEC::LAYER_1::SPEC::OUTPUT_DIM];
        backward(network.hidden_layer_0     , d_hidden_layer_0_output, d_input_layer_output);
        backward(network.input_layer     , d_input_layer_output, d_input);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void backward(nn_models::three_layer_fc::NeuralNetworkBackwardGradient<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_LAYER::OUTPUT_DIM], typename SPEC::T d_input[SPEC::LAYER_1::INPUT_DIM]) {
        typename SPEC::T d_hidden_layer_0_output[SPEC::LAYER_2::SPEC::OUTPUT_DIM];
        backward(network.output_layer, network.hidden_layer_0.output, d_output, d_hidden_layer_0_output);
        typename SPEC::T d_input_layer_output[SPEC::LAYER_1::SPEC::OUTPUT_DIM];
        backward(network.hidden_layer_0     , network.input_layer.output, d_hidden_layer_0_output, d_input_layer_output);
        backward(network.input_layer     , input                 , d_input_layer_output, d_input);
    }
    template<typename SPEC, int BATCH_SIZE>
    FUNCTION_PLACEMENT void forward_backward_mse(nn_models::three_layer_fc::NeuralNetworkBackwardGradient<devices::Generic, SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T target[SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        typename SPEC::T d_input[SPEC::LAYER_1::INPUT_DIM];
        forward(network, input);
        typename SPEC::T d_loss_d_output[SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<typename SPEC::T, SPEC::OUTPUT_LAYER::OUTPUT_DIM, BATCH_SIZE>(network.output_layer.output, target, d_loss_d_output);
        backward(network, input, d_loss_d_output, d_input);
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::three_layer_fc::NeuralNetworkSGD<devices::Generic, SPEC>& network) {
        update_layer(network.input_layer     );
        update_layer(network.hidden_layer_0     );
        update_layer(network.output_layer);
    }


    template<typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::three_layer_fc::NeuralNetworkAdam<devices::Generic, SPEC>& network) {
        typename SPEC::T  first_order_moment_bias_correction = 1/(1 - pow(SPEC::ADAM_PARAMETERS::BETA_1, network.age));
        typename SPEC::T second_order_moment_bias_correction = 1/(1 - pow(SPEC::ADAM_PARAMETERS::BETA_2, network.age));
        update_layer(network.input_layer     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.hidden_layer_0     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::three_layer_fc::NeuralNetworkSGD<devices::Generic, SPEC>& network) {
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::three_layer_fc::NeuralNetworkAdam<devices::Generic, SPEC>& network) {
        reset_optimizer_state(network.input_layer);
        reset_optimizer_state(network.hidden_layer_0);
        reset_optimizer_state(network.output_layer);
        network.age = 1;
    }


    template<typename SPEC, auto RANDOM_UNIFORM, typename RNG>
    FUNCTION_PLACEMENT void init_weights(nn_models::three_layer_fc::NeuralNetwork<devices::Generic, SPEC>& network, RNG& rng) {
        init_kaiming<typename SPEC::LAYER_1::SPEC, RANDOM_UNIFORM, RNG>(network.input_layer, rng);
        init_kaiming<typename SPEC::LAYER_2::SPEC, RANDOM_UNIFORM, RNG>(network.hidden_layer_0, rng);
        init_kaiming<typename SPEC::OUTPUT_LAYER::SPEC, RANDOM_UNIFORM, RNG>(network.output_layer, rng);
    }
}

#endif