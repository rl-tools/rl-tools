#ifndef NEURAL_NETWORK_MODELS_H
#define NEURAL_NETWORK_MODELS_H
#include <nn/nn.h>
namespace layer_in_c::nn_models {
    using namespace nn;
    using namespace nn::activation_functions;

    namespace three_layer_fc {
        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::layers::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::layers::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::layers::ActivationFunction OUTPUT_LAYER_FN>
        struct InferenceSpecification{
            typedef T_T T;
            typedef layers::Layer<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> LAYER_1;
            typedef layers::Layer<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> LAYER_2;
            typedef layers::Layer<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::layers::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::layers::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::layers::ActivationFunction OUTPUT_LAYER_FN>
        struct InferenceBackwardSpecification{
            typedef T_T T;
            typedef layers::LayerBackward<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> LAYER_1;
            typedef layers::LayerBackward<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> LAYER_2;
            typedef layers::LayerBackward<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::layers::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::layers::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::layers::ActivationFunction OUTPUT_LAYER_FN, typename T_SGD_PARAMETERS>
        struct SGDSpecification{
            typedef T_T T;
            typedef T_SGD_PARAMETERS SGD_PARAMETERS;
            typedef layers::LayerBackwardSGD<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN, SGD_PARAMETERS> LAYER_1;
            typedef layers::LayerBackwardSGD<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN, SGD_PARAMETERS> LAYER_2;
            typedef layers::LayerBackwardSGD<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN, SGD_PARAMETERS> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::layers::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::layers::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::layers::ActivationFunction OUTPUT_LAYER_FN, typename T_ADAM_PARAMETERS>
        struct AdamSpecification{
            typedef T_T T;
            typedef T_ADAM_PARAMETERS ADAM_PARAMETERS;
            typedef layers::LayerBackwardAdam<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN, ADAM_PARAMETERS> LAYER_1;
            typedef layers::LayerBackwardAdam<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN, ADAM_PARAMETERS> LAYER_2;
            typedef layers::LayerBackwardAdam<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN, ADAM_PARAMETERS> OUTPUT_LAYER;
        };

        template<typename T_NETWORK_SPEC>
        struct NeuralNetwork{
            typedef T_NETWORK_SPEC NETWORK_SPEC;
            typename NETWORK_SPEC::LAYER_1 layer_1;
            typename NETWORK_SPEC::LAYER_2 layer_2;
            typename NETWORK_SPEC::OUTPUT_LAYER output_layer;
        };

        template<typename NETWORK_SPEC>
        struct NeuralNetworkBackward: public three_layer_fc::NeuralNetwork<NETWORK_SPEC>{
        };

        template<typename NETWORK_SPEC>
        struct NeuralNetworkSGD: public three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>{
        };

        template<typename NETWORK_SPEC>
        struct NeuralNetworkAdam: public three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>{
            uint32_t age = 1;
        };


    }

    // forward modifies intermediate outputs to facilitate backward pass
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void forward(three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
    }
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void forward(three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM], typename NETWORK_SPEC::T output[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, output);
    }
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT typename NETWORK_SPEC::T forward_univariate(three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
        return network.output_layer.output[0];
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void evaluate(three_layer_fc::NeuralNetwork<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM], typename NETWORK_SPEC::T output[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM]){
        typename NETWORK_SPEC::T layer_1_output[NETWORK_SPEC::LAYER_1::OUTPUT_DIM];
        evaluate(network.layer_1     , input, layer_1_output);
        typename NETWORK_SPEC::T layer_2_output[NETWORK_SPEC::LAYER_2::OUTPUT_DIM];
        evaluate(network.layer_2     , layer_1_output, layer_2_output);
        evaluate(network.output_layer, layer_2_output, output);
    }

    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT typename NETWORK_SPEC::T evaluate(three_layer_fc::NeuralNetwork<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename NETWORK_SPEC::T output[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        evaluate(network, input, output);
        return output[0];
    }
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void zero_gradient(three_layer_fc::NeuralNetwork<NETWORK_SPEC>& network) {
        layers::zero_gradient(network.layer_1);
        layers::zero_gradient(network.layer_2);
        layers::zero_gradient(network.output_layer);
    }
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void backward(three_layer_fc::NeuralNetwork<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM], const typename NETWORK_SPEC::T d_output[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM], typename NETWORK_SPEC::T d_input[NETWORK_SPEC::LAYER_1::INPUT_DIM]) {
        typename NETWORK_SPEC::T d_layer_2_output[NETWORK_SPEC::LAYER_2::OUTPUT_DIM];
        backward(network.output_layer, network.layer_2.output, d_output, d_layer_2_output);
        typename NETWORK_SPEC::T d_layer_1_output[NETWORK_SPEC::LAYER_1::OUTPUT_DIM];
        backward(network.layer_2     , network.layer_1.output, d_layer_2_output, d_layer_1_output);
        backward(network.layer_1     , input                 , d_layer_1_output, d_input);
    }
    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void forward_backward_mse(three_layer_fc::NeuralNetworkBackward<NETWORK_SPEC>& network, const typename NETWORK_SPEC::T input[NETWORK_SPEC::LAYER_1::INPUT_DIM], typename NETWORK_SPEC::T target[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        typename NETWORK_SPEC::T d_input[NETWORK_SPEC::LAYER_1::INPUT_DIM];
        forward(network, input);
        typename NETWORK_SPEC::T d_loss_d_output[NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<typename NETWORK_SPEC::T, NETWORK_SPEC::OUTPUT_LAYER::OUTPUT_DIM>(network.output_layer.output, target, d_loss_d_output);
        backward(network, input, d_loss_d_output, d_input);
    }

    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void update(three_layer_fc::NeuralNetworkSGD<NETWORK_SPEC>& network) {
        update_layer(network.layer_1     );
        update_layer(network.layer_2     );
        update_layer(network.output_layer);
    }


    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void update(three_layer_fc::NeuralNetworkAdam<NETWORK_SPEC>& network) {
        typename NETWORK_SPEC::T  first_order_moment_bias_correction = 1/(1 - pow(NETWORK_SPEC::ADAM_PARAMETERS::BETA_1, network.age));
        typename NETWORK_SPEC::T second_order_moment_bias_correction = 1/(1 - pow(NETWORK_SPEC::ADAM_PARAMETERS::BETA_2, network.age));
        update_layer(network.layer_1     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.layer_2     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(three_layer_fc::NeuralNetworkSGD<NETWORK_SPEC>& network) {
    }

    template<typename NETWORK_SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(three_layer_fc::NeuralNetworkAdam<NETWORK_SPEC>& network) {
        reset_optimizer_state(network.layer_1);
        reset_optimizer_state(network.layer_2);
        reset_optimizer_state(network.output_layer);
        network.age = 1;
    }


    template<typename NETWORK_SPEC, typename RNG>
    FUNCTION_PLACEMENT void init_weights(three_layer_fc::NeuralNetwork<NETWORK_SPEC>& network, RNG& rng) {
        init_layer_kaiming(network.layer_1, rng);
        init_layer_kaiming(network.layer_2, rng);
        init_layer_kaiming(network.output_layer, rng);
    }
}

#endif