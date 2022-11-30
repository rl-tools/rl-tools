#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H

#include <layer_in_c/nn/layers/dense/layer.h>

namespace layer_in_c{
    template<typename T, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(const nn::layers::dense::Layer<devices::Generic, SPEC>& layer, const T input[SPEC::INPUT_DIM], T output[SPEC::OUTPUT_DIM]) {
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.biases[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                output[i] += layer.weights[i][j] * input[j];
            }
            output[i] = nn::activation_functions::activation<T, SPEC::ACTIVATION_FUNCTION>(output[i]);
        }
    }

    template<typename T, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(nn::layers::dense::LayerBackward<devices::Generic, SPEC>& layer, const T input[SPEC::INPUT_DIM]) {
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.output[i] = layer.biases[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.output[i] += layer.weights[i][j] * input[j];
            }
            layer.output[i] = nn::activation_functions::activation<T, SPEC::ACTIVATION_FUNCTION>(layer.output[i]);
        }
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void backward(nn::layers::dense::LayerBackward<devices::Generic, SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            typename SPEC::T d_pre_activation = nn::activation_functions::d_activation_d_x<typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.output[i]) * d_output[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
            }
        }
    }

    template<typename LS>
    FUNCTION_PLACEMENT void backward(nn::layers::dense::LayerBackwardGradient<devices::Generic, LS>& layer, const typename LS::T input[LS::INPUT_DIM], const typename LS::T d_output[LS::OUTPUT_DIM], typename LS::T d_input[LS::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            typename LS::T d_pre_activation = nn::activation_functions::d_activation_d_x<typename LS::T, LS::ACTIVATION_FUNCTION>(layer.output[i]) * d_output[i];
            layer.d_biases[i] += d_pre_activation;
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
                layer.d_weights[i][j] += d_pre_activation * input[j];
            }
        }
    }
    template<typename LS>
    FUNCTION_PLACEMENT void zero_gradient(nn::layers::dense::LayerBackwardGradient<devices::Generic, LS>& layer) {
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.d_biases[i] = 0;
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.d_weights[i][j] = 0;
            }
        }
    }
    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(nn::layers::dense::LayerBackwardSGD<devices::Generic, LS, PARAMETERS>& layer){
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] -= PARAMETERS::ALPHA * layer.d_biases[i];
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] -= PARAMETERS::ALPHA * layer.d_weights[i][j];
            }
        }
    }

    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn::layers::dense::LayerBackwardAdam<devices::Generic, LS, PARAMETERS>& layer) {
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.d_biases_first_order_moment [i] = 0;
            layer.d_biases_second_order_moment[i] = 0;
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.d_weights_first_order_moment [i][j] = 0;
                layer.d_weights_second_order_moment[i][j] = 0;
            }
        }
    }
    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void gradient_descent(nn::layers::dense::LayerBackwardAdam<devices::Generic, LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction){
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            typename LS::T bias_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_biases_first_order_moment[i] / (std::sqrt(layer.d_biases_second_order_moment[i] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
            layer.biases[i] -= bias_update;
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                typename LS::T weight_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_weights_first_order_moment[i][j] / (std::sqrt(layer.d_weights_second_order_moment[i][j] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
                layer.weights[i][j] -= weight_update;
            }
        }
    }

    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(nn::layers::dense::LayerBackwardAdam<devices::Generic, LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction) {
        utils::polyak::update_matrix<typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak::update       <typename LS::T, LS::OUTPUT_DIM>               (layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak::update_squared_matrix<typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak::update_squared       <typename LS::T, LS::OUTPUT_DIM>               (layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(layer, first_order_moment_bias_correction, second_order_moment_bias_correction);

    }
    template<typename LS, auto RANDOM_UNIFORM, typename RNG>
    FUNCTION_PLACEMENT void init_kaiming(nn::layers::dense::Layer<devices::Generic, LS>& layer, RNG& rng) {
        typedef typename LS::T T;
        T negative_slope = std::sqrt((T)5);
        T gain = std::sqrt((T)2.0 / (1 + negative_slope * negative_slope));
        T fan = LS::INPUT_DIM;
        T std = gain / std::sqrt(fan);
        T weight_bound = std::sqrt((T)3.0) * std;
//        auto weight_distribution = std::uniform_real_distribution<T>(-weight_bound, weight_bound);
        T bias_bound = 1/std::sqrt((T)LS::INPUT_DIM);
//        auto bias_distribution = std::uniform_real_distribution<T>(-bias_bound, bias_bound);

        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
//            layer.biases[i] = bias_distribution(rng);
            layer.biases[i] = RANDOM_UNIFORM(-bias_bound, bias_bound, rng);
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] = RANDOM_UNIFORM(-weight_bound, weight_bound, rng);
            }
        }
    }
}

#endif