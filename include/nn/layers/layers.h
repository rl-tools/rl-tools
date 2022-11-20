#ifndef LAYER_IN_C_NN_LAYERS
#define LAYER_IN_C_NN_LAYERS
#include <random>
#include "../activation_functions.h"
#include "../../utils/polyak.h"

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c::nn::layers {
    using namespace layer_in_c::nn::activation_functions;
    template<typename T, int T_INPUT_DIM, int T_OUTPUT_DIM, ActivationFunction>
    struct Layer{
        static constexpr int INPUT_DIM = T_INPUT_DIM;
        static constexpr int OUTPUT_DIM = T_OUTPUT_DIM;
        T weights[OUTPUT_DIM][INPUT_DIM];
        T biases [OUTPUT_DIM];
    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION>
    struct LayerBackward: public Layer<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{
        T output   [OUTPUT_DIM];
    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION>
    struct LayerBackwardGradient: public LayerBackward<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{
        T d_weights[OUTPUT_DIM][INPUT_DIM];
        T d_biases [OUTPUT_DIM];
    };
    template <typename T>
    struct DefaultSGDParameters{
    public:
        static constexpr T ALPHA   = 0.001;

    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION, typename PARAMETERS>
    struct LayerBackwardSGD: public LayerBackward<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{};

    template <typename T>
    struct DefaultAdamParameters{
    public:
        static constexpr T ALPHA   = 0.001;
        static constexpr T BETA_1  = 0.9;
        static constexpr T BETA_2  = 0.999;
        static constexpr T EPSILON = 1e-7;

    };
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction ACTIVATION_FUNCTION, typename PARAMETERS>
    struct LayerBackwardAdam: public LayerBackwardGradient<T, INPUT_DIM, OUTPUT_DIM, ACTIVATION_FUNCTION>{
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
            for(int j = 0; j < INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
            }
        }
    }

    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN>
    FUNCTION_PLACEMENT void backward(LayerBackwardGradient<T, INPUT_DIM, OUTPUT_DIM, FN>& layer, const T input[INPUT_DIM], const T d_output[OUTPUT_DIM], T d_input[INPUT_DIM]) {
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
    FUNCTION_PLACEMENT void zero_gradient(LayerBackwardGradient<T, INPUT_DIM, OUTPUT_DIM, FN>& layer) {
        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.d_biases[i] = 0;
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.d_weights[i][j] = 0;
            }
        }
    }
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename PARAMETERS>
    FUNCTION_PLACEMENT void gradient_descent(LayerBackwardSGD<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer){
        for(int i = 0; i < OUTPUT_DIM; i++) {
            layer.biases[i] -= PARAMETERS::ALPHA * layer.d_biases[i];
            for(int j = 0; j < INPUT_DIM; j++) {
                layer.weights[i][j] -= PARAMETERS::ALPHA * layer.d_weights[i][j];
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
    FUNCTION_PLACEMENT void gradient_descent(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer, T first_order_moment_bias_correction, T second_order_moment_bias_correction){
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
    FUNCTION_PLACEMENT void update_layer(LayerBackwardAdam<T, INPUT_DIM, OUTPUT_DIM, FN, PARAMETERS>& layer, T first_order_moment_bias_correction, T second_order_moment_bias_correction) {
        utils::polyak_update_matrix<T, OUTPUT_DIM, INPUT_DIM>(layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak_update       <T, OUTPUT_DIM>           (layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak_update_squared_matrix<T, OUTPUT_DIM, INPUT_DIM>(layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak_update_squared       <T, OUTPUT_DIM>           (layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(layer, first_order_moment_bias_correction, second_order_moment_bias_correction);

    }
    template<typename T, int INPUT_DIM, int OUTPUT_DIM, ActivationFunction FN, typename RNG>
    FUNCTION_PLACEMENT void init_layer_kaiming(Layer<T, INPUT_DIM, OUTPUT_DIM, FN>& layer, RNG& rng) {
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