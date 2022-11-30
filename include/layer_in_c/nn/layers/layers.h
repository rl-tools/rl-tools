#ifndef LAYER_IN_C_NN_LAYERS
#define LAYER_IN_C_NN_LAYERS
#include "../activation_functions.h"
#include "../../utils/polyak.h"

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c::nn::layers {
    using namespace layer_in_c::nn::activation_functions;

    template<typename T_T, int T_INPUT_DIM, int T_OUTPUT_DIM, ActivationFunction T_ACTIVATION_FUNCTION>
    struct LayerSpec {
        typedef T_T T;
        static constexpr int INPUT_DIM = T_INPUT_DIM;
        static constexpr int OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
    };
    template<typename T_SPEC>
    struct Layer{
        typedef T_SPEC SPEC;
        typename SPEC::T weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T biases [SPEC::OUTPUT_DIM];
    };
    template<typename SPEC>
    struct LayerBackward: public Layer<SPEC>{
        typename SPEC::T output   [SPEC::OUTPUT_DIM];
    };
    template<typename SPEC>
    struct LayerBackwardGradient: public LayerBackward<SPEC>{
        typename SPEC::T d_weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases [SPEC::OUTPUT_DIM];
    };
    template <typename T>
    struct DefaultSGDParameters{
    public:
        static constexpr T ALPHA = 0.001;

    };
    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardSGD: public LayerBackwardGradient<SPEC>{};

    template <typename T>
    struct DefaultAdamParameters{
    public:
        static constexpr T ALPHA   = 0.001;
        static constexpr T BETA_1  = 0.9;
        static constexpr T BETA_2  = 0.999;
        static constexpr T EPSILON = 1e-7;

    };
    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardAdam: public LayerBackwardGradient<SPEC>{
        typename SPEC::T d_weights_first_order_moment [SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_weights_second_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases_first_order_moment  [SPEC::OUTPUT_DIM];
        typename SPEC::T d_biases_second_order_moment [SPEC::OUTPUT_DIM];
    };


    template<typename T, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(const Layer<SPEC>& layer, const T input[SPEC::INPUT_DIM], T output[SPEC::OUTPUT_DIM]) {
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.biases[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                output[i] += layer.weights[i][j] * input[j];
            }
            output[i] = activation<T, SPEC::ACTIVATION_FUNCTION>(output[i]);
        }
    }

    template<typename T, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(LayerBackward<SPEC>& layer, const T input[SPEC::INPUT_DIM]) {
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.output[i] = layer.biases[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.output[i] += layer.weights[i][j] * input[j];
            }
            layer.output[i] = activation<T, SPEC::ACTIVATION_FUNCTION>(layer.output[i]);
        }
    }

    template<typename SPEC>
    FUNCTION_PLACEMENT void backward(LayerBackward<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(int i = 0; i < SPEC::OUTPUT_DIM; i++) {
            typename SPEC::T d_pre_activation = d_activation_d_x<typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.output[i]) * d_output[i];
            for(int j = 0; j < SPEC::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
            }
        }
    }

    template<typename LS>
    FUNCTION_PLACEMENT void backward(LayerBackwardGradient<LS>& layer, const typename LS::T input[LS::INPUT_DIM], const typename LS::T d_output[LS::OUTPUT_DIM], typename LS::T d_input[LS::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            typename LS::T d_pre_activation = d_activation_d_x<typename LS::T, LS::ACTIVATION_FUNCTION>(layer.output[i]) * d_output[i];
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
    FUNCTION_PLACEMENT void zero_gradient(LayerBackwardGradient<LS>& layer) {
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.d_biases[i] = 0;
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.d_weights[i][j] = 0;
            }
        }
    }
    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(LayerBackwardSGD<LS, PARAMETERS>& layer){
        for(int i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] -= PARAMETERS::ALPHA * layer.d_biases[i];
            for(int j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] -= PARAMETERS::ALPHA * layer.d_weights[i][j];
            }
        }
    }

    template<typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(LayerBackwardAdam<LS, PARAMETERS>& layer) {
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
    FUNCTION_PLACEMENT void gradient_descent(LayerBackwardAdam<LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction){
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
    FUNCTION_PLACEMENT void update_layer(LayerBackwardAdam<LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction) {
        utils::polyak_update_matrix<typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak_update       <typename LS::T, LS::OUTPUT_DIM>               (layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak_update_squared_matrix<typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak_update_squared       <typename LS::T, LS::OUTPUT_DIM>               (layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(layer, first_order_moment_bias_correction, second_order_moment_bias_correction);

    }
    template<typename LS, auto RANDOM_UNIFORM, typename RNG>
    FUNCTION_PLACEMENT void init_layer_kaiming(Layer<LS>& layer, RNG& rng) {
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