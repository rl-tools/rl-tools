#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <random>
#include "../utils/polyak.h"

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c::nn {
    template<typename T, int DIM>
    T mse(const T a[DIM], const T b[DIM]) {
        T acc = 0;
        for(int i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            acc += diff * diff;
        }
        return acc / DIM;
    }

    template<typename T, int DIM>
    void d_mse_d_x(const T a[DIM], const T b[DIM], T d_a[DIM]) {
        for(int i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            d_a[i] = 2*diff/DIM;
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction>
    struct Layer{
        T weights[OUTPUT_DIM][INPUT_DIM];
        T biases [OUTPUT_DIM];
    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION>
    struct LayerBackward: public Layer<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{
        T output   [OUTPUT_DIM];
        T d_weights[OUTPUT_DIM][INPUT_DIM];
        T d_biases [OUTPUT_DIM];
    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION, typename PARAMETERS>
    struct LayerBackwardAdam: public LayerBackward<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{
        T d_weights_first_order_moment [OUTPUT_DIM][INPUT_DIM];
        T d_weights_second_order_moment[OUTPUT_DIM][INPUT_DIM];
        T d_biases_first_order_moment  [OUTPUT_DIM];
        T d_biases_second_order_moment [OUTPUT_DIM];
    };


    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN>
    FUNCTION_PLACEMENT void evaluate(const Layer<T, INPUT_DIM, OUTPUT_DIM, FN>& layer, const T input[INPUT_DIM], T output[OUTPUT_DIM]) {
        for(int i = 0; i < OUTPUT_DIM; i++) {
            output[i] = layer.biases[i];
            for(int j = 0; j < INPUT_DIM; j++) {
                output[i] += layer.weights[i][j] * input[j];
            }
            output[i] = activation<T, FN>(output[i]);
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN>
    FUNCTION_PLACEMENT void evaluate(LayerBackward<T, INPUT_DIM, OUTPUT_DIM, FN>& layer, const T input[INPUT_DIM]) {
        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.output[i] = layer.biases[i];
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.output[i] += layer.weights[i][j] * input[j];
            }
            layer.output[i] = activation<T, FN>(layer.output[i]);
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN>
    FUNCTION_PLACEMENT void backward(LayerBackward<T, INPUT_DIM, OUTPUT_DIM, FN>& layer, const T input[INPUT_DIM], const T d_output[OUTPUT_DIM], T d_input[INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(int i = 0; i < OUTPUT_DIM; i++) {
            T d_pre_activation = d_activation_d_x<T, FN>(layer.output[i]) * d_output[i];
            layer.d_biases[i] += d_pre_activation;
            for(int j = 0; j < INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
                layer.d_weights[i][j] += d_pre_activation * input[j];
            }
        }
    }
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN>
    FUNCTION_PLACEMENT void zero_gradient(LayerBackward<T, INPUT_DIM, OUTPUT_DIM, FN>& layer) {
        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.d_biases[i] = 0;
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.d_weights[i][j] = 0;
            }
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer) {
        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.d_biases_first_order_moment [i] = 0;
            layer.d_biases_second_order_moment[i] = 0;
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.d_weights_first_order_moment [i][j] = 0;
                layer.d_weights_second_order_moment[i][j] = 0;
            }
        }
    }
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void gradient_descent(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer, T first_order_moment_bias_correction, T second_order_moment_bias_correction, const uint32_t batch_size){
        for(int i = 0; i < OUTPUT_DIM; i++) {
            T bias_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_biases_first_order_moment[i] / (std::sqrt(layer.d_biases_second_order_moment[i] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
            layer.biases[i] -= bias_update;
            for(int j = 0; j < INPUT_DIM; j++) {
                T weight_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_weights_first_order_moment[i][j] / (std::sqrt(layer.d_weights_second_order_moment[i][j] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
                layer.weights[i][j] -= weight_update;
            }
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer, T first_order_moment_bias_correction, T second_order_moment_bias_correction, const uint32_t batch_size) {
        utils::polyak_update_matrix<T, OUTPUT_DIM, INPUT_DIM>(layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak_update       <T, OUTPUT_DIM>           (layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak_update_squared_matrix<T, OUTPUT_DIM, INPUT_DIM>(layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak_update_squared       <T, OUTPUT_DIM>           (layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(layer, first_order_moment_bias_correction, second_order_moment_bias_correction, batch_size);

    }
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename PARAMETERS, typename RNG>
    FUNCTION_PLACEMENT void init_layer_kaiming(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer, RNG& rng) {
        T negative_slope = std::sqrt(5);
        T gain = std::sqrt(2.0 / (1 + negative_slope * negative_slope));
        T fan = INPUT_DIM;
        T std = gain / std::sqrt(fan);
        T weight_bound = std::sqrt(3.0) * std;
        auto weight_distribution = std::uniform_real_distribution<T>(-weight_bound, weight_bound);
        T bias_bound = 1/std::sqrt((T)INPUT_DIM);
        auto bias_distribution = std::uniform_real_distribution<T>(-bias_bound, bias_bound);

        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.biases[i] = bias_distribution(rng);
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.weights[i][j] = weight_distribution(rng);
            }
        }
    }
}

#endif