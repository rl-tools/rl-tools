#ifndef NEURAL_NETWORK_MODELS_H
#define NEURAL_NETWORK_MODELS_H
#include <layer_in_c/nn/nn.h>
namespace layer_in_c {

    namespace nn_models::three_layer_fc {
        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::activation_functions::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::activation_functions::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN>
        struct InferenceSpecification{
            typedef T_T T;
            typedef nn::layers::dense::Layer<nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN>> LAYER_1;
            typedef nn::layers::dense::Layer<nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN>> LAYER_2;
            typedef nn::layers::dense::Layer<nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN>> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::activation_functions::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::activation_functions::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN>
        struct InferenceBackwardSpecification{
            typedef T_T T;
            typedef nn::layers::dense::LayerBackward<nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN>> LAYER_1;
            typedef nn::layers::dense::LayerBackward<nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN>> LAYER_2;
            typedef nn::layers::dense::LayerBackward<nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN>> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::activation_functions::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::activation_functions::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN, typename T_SGD_PARAMETERS>
        struct SGDSpecification{
            typedef T_T T;
            typedef T_SGD_PARAMETERS SGD_PARAMETERS;
            typedef nn::layers::dense::LayerBackwardSGD<nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN>, SGD_PARAMETERS> LAYER_1;
            typedef nn::layers::dense::LayerBackwardSGD<nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN>, SGD_PARAMETERS> LAYER_2;
            typedef nn::layers::dense::LayerBackwardSGD<nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN>, SGD_PARAMETERS> OUTPUT_LAYER;
        };

        template<typename T_T, int INPUT_DIM, int LAYER_1_DIM, nn::activation_functions::ActivationFunction LAYER_1_FN, int LAYER_2_DIM, nn::activation_functions::ActivationFunction LAYER_2_FN, int OUTPUT_DIM, nn::activation_functions::ActivationFunction OUTPUT_LAYER_FN, typename T_ADAM_PARAMETERS>
        struct AdamSpecification{
            typedef T_T T;
            typedef T_ADAM_PARAMETERS ADAM_PARAMETERS;
            typedef nn::layers::dense::LayerBackwardAdam<nn::layers::dense::LayerSpec<T,   INPUT_DIM, LAYER_1_DIM,      LAYER_1_FN>, ADAM_PARAMETERS> LAYER_1;
            typedef nn::layers::dense::LayerBackwardAdam<nn::layers::dense::LayerSpec<T, LAYER_1_DIM, LAYER_2_DIM,      LAYER_2_FN>, ADAM_PARAMETERS> LAYER_2;
            typedef nn::layers::dense::LayerBackwardAdam<nn::layers::dense::LayerSpec<T, LAYER_2_DIM,  OUTPUT_DIM, OUTPUT_LAYER_FN>, ADAM_PARAMETERS> OUTPUT_LAYER;
        };

        template<typename T_SPEC>
        struct NeuralNetwork{
            static constexpr int INPUT_DIM = T_SPEC::LAYER_1::SPEC::INPUT_DIM;
            static constexpr int OUTPUT_DIM = T_SPEC::OUTPUT_LAYER::SPEC::OUTPUT_DIM;
            typedef T_SPEC SPEC;
            typename SPEC::LAYER_1 layer_1;
            typename SPEC::LAYER_2 layer_2;
            typename SPEC::OUTPUT_LAYER output_layer;
        };

        template<typename SPEC>
        struct NeuralNetworkBackward: public NeuralNetwork<SPEC>{
        };

        template<typename SPEC>
        struct NeuralNetworkSGD: public NeuralNetworkBackward<SPEC>{
        };

        template<typename SPEC>
        struct NeuralNetworkAdam: public NeuralNetworkBackward<SPEC>{
            uint32_t age = 1;
        };


    }

    // forward modifies intermediate outputs to facilitate backward pass
    template<typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::three_layer_fc::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void forward(nn_models::three_layer_fc::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, output);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T forward_univariate(nn_models::three_layer_fc::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        evaluate(network.layer_1     , input);
        evaluate(network.layer_2     , network.layer_1.output);
        evaluate(network.output_layer, network.layer_2.output);
        return network.output_layer.output[0];
    }

    // evaluate does not set intermediate outputs and hence can also be called from stateless layers, for register efficiency use forward when working with "Backward" compatible layers
    template<typename SPEC>
    FUNCTION_PLACEMENT void evaluate(nn_models::three_layer_fc::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM]){
        typename SPEC::T layer_1_output[SPEC::LAYER_1::OUTPUT_DIM];
        evaluate(network.layer_1     , input, layer_1_output);
        typename SPEC::T layer_2_output[SPEC::LAYER_2::OUTPUT_DIM];
        evaluate(network.layer_2     , layer_1_output, layer_2_output);
        evaluate(network.output_layer, layer_2_output, output);
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT typename SPEC::T evaluate(nn_models::three_layer_fc::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM]) {
        static_assert(SPEC::OUTPUT_LAYER::OUTPUT_DIM == 1, "OUTPUT_DIM has to be 1 for return based evaluation");
        typename SPEC::T output[SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        evaluate(network, input, output);
        return output[0];
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void zero_gradient(nn_models::three_layer_fc::NeuralNetwork<SPEC>& network) {
        zero_gradient(network.layer_1);
        zero_gradient(network.layer_2);
        zero_gradient(network.output_layer);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void backward(nn_models::three_layer_fc::NeuralNetwork<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_LAYER::OUTPUT_DIM], typename SPEC::T d_input[SPEC::LAYER_1::INPUT_DIM]) {
        typename SPEC::T d_layer_2_output[SPEC::LAYER_2::SPEC::OUTPUT_DIM];
        backward(network.output_layer, network.layer_2.output, d_output, d_layer_2_output);
        typename SPEC::T d_layer_1_output[SPEC::LAYER_1::SPEC::OUTPUT_DIM];
        backward(network.layer_2     , network.layer_1.output, d_layer_2_output, d_layer_1_output);
        backward(network.layer_1     , input                 , d_layer_1_output, d_input);
    }
    template<typename SPEC>
    FUNCTION_PLACEMENT void forward_backward_mse(nn_models::three_layer_fc::NeuralNetworkBackward<SPEC>& network, const typename SPEC::T input[SPEC::LAYER_1::INPUT_DIM], typename SPEC::T target[SPEC::OUTPUT_LAYER::OUTPUT_DIM]) {
        typename SPEC::T d_input[SPEC::LAYER_1::INPUT_DIM];
        forward(network, input);
        typename SPEC::T d_loss_d_output[SPEC::OUTPUT_LAYER::OUTPUT_DIM];
        nn::loss_functions::d_mse_d_x<typename SPEC::T, SPEC::OUTPUT_LAYER::OUTPUT_DIM>(network.output_layer.output, target, d_loss_d_output);
        backward(network, input, d_loss_d_output, d_input);
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::three_layer_fc::NeuralNetworkSGD<SPEC>& network) {
        update_layer(network.layer_1     );
        update_layer(network.layer_2     );
        update_layer(network.output_layer);
    }


    template<typename SPEC>
    FUNCTION_PLACEMENT void update(nn_models::three_layer_fc::NeuralNetworkAdam<SPEC>& network) {
        typename SPEC::T  first_order_moment_bias_correction = 1/(1 - pow(SPEC::ADAM_PARAMETERS::BETA_1, network.age));
        typename SPEC::T second_order_moment_bias_correction = 1/(1 - pow(SPEC::ADAM_PARAMETERS::BETA_2, network.age));
        update_layer(network.layer_1     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.layer_2     , first_order_moment_bias_correction, second_order_moment_bias_correction);
        update_layer(network.output_layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
        network.age += 1;
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::three_layer_fc::NeuralNetworkSGD<SPEC>& network) {
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn_models::three_layer_fc::NeuralNetworkAdam<SPEC>& network) {
        reset_optimizer_state(network.layer_1);
        reset_optimizer_state(network.layer_2);
        reset_optimizer_state(network.output_layer);
        network.age = 1;
    }


    template<typename SPEC, auto RANDOM_UNIFORM, typename RNG>
    FUNCTION_PLACEMENT void init_weights(nn_models::three_layer_fc::NeuralNetwork<SPEC>& network, RNG& rng) {
        init_kaiming<typename SPEC::LAYER_1::SPEC, RANDOM_UNIFORM, RNG>(network.layer_1, rng);
        init_kaiming<typename SPEC::LAYER_2::SPEC, RANDOM_UNIFORM, RNG>(network.layer_2, rng);
        init_kaiming<typename SPEC::OUTPUT_LAYER::SPEC, RANDOM_UNIFORM, RNG>(network.output_layer, rng);
    }
}

#endif