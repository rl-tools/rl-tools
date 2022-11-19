#ifndef NEURAL_NETWORK_MODELS_H
#define NEURAL_NETWORK_MODELS_H
#include <nn/nn.h>
namespace layer_in_c::nn_models {
    using namespace nn;
    using namespace nn::activation_functions;

    template<typename T>
    struct DefaultAdamParameters{
    public:
        static constexpr T ALPHA   = 0.001;
        static constexpr T BETA_1  = 0.9;
        static constexpr T BETA_2  = 0.999;
        static constexpr T EPSILON = 1e-7;

    };
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    struct ThreeLayerNeuralNetworkTrainingAdam{
        layers::LayerBackwardAdam<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN, PARAMETERS> layer_1;
        layers::LayerBackwardAdam<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN, PARAMETERS> layer_2;
        layers::LayerBackwardAdam<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS> output_layer;
        uint32_t age = 1;
    };

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void forward(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void evaluate(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], T output[OUTPUT_DIM]) {
        forward(network, input);
        for(int i = 0; i < OUTPUT_DIM; i++) {
            output[i] = network.output_layer.output[i];
        }
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT T evaluate(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM]) {
        static_assert(OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        forward(network, input);
        return network.output_layer.output[0];
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void zero_gradient(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network) {
        zero_gradient(network.layer_1);
        zero_gradient(network.layer_2);
        zero_gradient(network.output_layer);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void backward(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], const T d_output[OUTPUT_DIM], T d_input[INPUT_DIM]) {
        T d_layer_2_output[LAYER_2_DIM];
        backward(network.output_layer, network.layer_2.output, d_output, d_layer_2_output);
        T d_layer_1_output[LAYER_1_DIM];
        backward(network.layer_2     , network.layer_1.output, d_layer_2_output, d_layer_1_output);
        backward(network.layer_1     , input                 , d_layer_1_output, d_input);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void forward_backward_mse(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], T target[OUTPUT_DIM]) {
        T d_input[INPUT_DIM];
        forward(network, input);
        T d_loss_d_output[OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<T, OUTPUT_DIM>(network.output_layer.output, target, d_loss_d_output);
        backward(network, input, d_loss_d_output, d_input);
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void update(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network) {
        T  first_order_moment_bias_correction = 1/(1 - pow(PARAMETERS::BETA_1, network.age));
        T second_order_moment_bias_correction = 1/(1 - pow(PARAMETERS::BETA_2, network.age));
        update_layer(network.layer_1     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.layer_2     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network) {
        reset_optimizer_state(network.layer_1);
        reset_optimizer_state(network.layer_2);
        reset_optimizer_state(network.output_layer);
        network.age = 1;
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS, typename RNG>
    FUNCTION_PLACEMENT void init_weights(ThreeLayerNeuralNetworkTrainingAdam<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, RNG& rng) {
        init_layer_kaiming(network.layer_1, rng);
        init_layer_kaiming(network.layer_2, rng);
        init_layer_kaiming(network.output_layer, rng);
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    struct ThreeLayerNeuralNetworkInferenceBackward{
        layers::LayerBackward<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> layer_1;
        layers::LayerBackward<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> layer_2;
        layers::LayerBackward<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> output_layer;
    };
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS, typename RNG>
    FUNCTION_PLACEMENT void init_weights(ThreeLayerNeuralNetworkInferenceBackward<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, RNG& rng) {
        init_layer_kaiming(network.layer_1, rng);
        init_layer_kaiming(network.layer_2, rng);
        init_layer_kaiming(network.output_layer, rng);
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void forward(ThreeLayerNeuralNetworkInferenceBackward<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void evaluate(ThreeLayerNeuralNetworkInferenceBackward<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], T output[OUTPUT_DIM]) {
        forward(network, input);
        for(int i = 0; i < OUTPUT_DIM; i++) {
            output[i] = network.output_layer.output[i];
        }
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT T evaluate(ThreeLayerNeuralNetworkInferenceBackward<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM]) {
        static_assert(OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        forward(network, input);
        return network.output_layer.output[0];
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void backward(ThreeLayerNeuralNetworkInferenceBackward<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], const T d_output[OUTPUT_DIM], T d_input[INPUT_DIM]) {
        T d_layer_2_output[LAYER_2_DIM];
        backward(network.output_layer, network.layer_2.output, d_output, d_layer_2_output);
        T d_layer_1_output[LAYER_1_DIM];
        backward(network.layer_2     , network.layer_1.output, d_layer_2_output, d_layer_1_output);
        backward(network.layer_1     , input                 , d_layer_1_output, d_input);
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    struct ThreeLayerNeuralNetworkInference{
        layers::Layer<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN> layer_1;
        layers::Layer<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN> layer_2;
        layers::Layer<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN> output_layer;
    };

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS, typename RNG>
    FUNCTION_PLACEMENT void init_weights(ThreeLayerNeuralNetworkInference<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, RNG& rng) {
        init_layer_kaiming(network.layer_1, rng);
        init_layer_kaiming(network.layer_2, rng);
        init_layer_kaiming(network.output_layer, rng);
    }

    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void forward(ThreeLayerNeuralNetworkInference<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], T output[INPUT_DIM]) {
        T layer_1_output[LAYER_1_DIM];
        evaluate(network.layer_1     , input, layer_1_output);
        T layer_2_output[LAYER_2_DIM];
        evaluate(network.layer_2     , layer_1_output, layer_2_output);
        evaluate(network.output_layer, layer_2_output, output);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void evaluate(ThreeLayerNeuralNetworkInference<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM], T output[OUTPUT_DIM]) {
        forward(network, input, output);
    }
    template<typename T, int INPUT_DIM, int LAYER_1_DIM, ActivationFunction LAYER_1_FN, int LAYER_2_DIM, ActivationFunction LAYER_2_FN, int OUTPUT_DIM, ActivationFunction OUTPUT_LAYER_FN, typename PARAMETERS>
    FUNCTION_PLACEMENT T evaluate(ThreeLayerNeuralNetworkInference<T, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_LAYER_FN, PARAMETERS>& network, const T input[INPUT_DIM]) {
        static_assert(OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        T output[1];
        forward(network, input, output);
        return output[0];
    }
}

#endif